#!/usr/bin/env python3
"""Run a safe, resumable DeepSeek V4 Flash multi-hop teacher pilot.

The pilot is paired exactly with the existing Qwen2.5-7B multi-hop pilot:
three sources, the same 20 question IDs per source, two trajectories per
question, seed 42, and at most three searches.  The teacher uses official
OpenAI-compatible function calling for the local ``search`` tool.  Completed
API responses are flushed to ``interactions.jsonl`` immediately, so an
interrupted run can replay paid successful calls before issuing new ones.

Only these two predicates determine eligibility:

    eligible = strict_project_format and final_answer_qa_em

The API key is read only from ``DEEPSEEK_API_KEY`` and is never included in
request dumps, output records, exceptions, or the summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from generate_multihop_sft_teacher_pilot import (
    DATASET_SOURCES,
    MAX_SEARCHES,
    _format_information,
    _json_safe,
    _row_question_record,
    answer_em,
    assemble_trajectory,
    normalize_answer,
    parse_turn,
    validate_trajectory,
)


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_DATASET_PATH = "data/multihopqa_search_mixed_402020_20260830/train.parquet"
DEFAULT_REFERENCE_SUMMARY = "data/teacher_sft_pilot_qwen2.5_7b_multihop/summary.json"
DEFAULT_OUTPUT_DIR = "data/teacher_sft_pilot_deepseek_v4_flash_multihop_fixed"
DEFAULT_RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_SEED = 42
DEFAULT_QUESTIONS_PER_SOURCE = 20
DEFAULT_SAMPLES_PER_QUESTION = 2
DEFAULT_MAX_SEARCHES = 3
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_CONTINUATIONS = 2
DEFAULT_API_TIMEOUT = 180.0
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_BACKOFF = 1.0
DEFAULT_RETRY_BACKOFF_MAX = 30.0

SYSTEM_PROMPT = (
    "You answer multi-hop questions with a search engine. Use the search tool "
    "when evidence is needed. You may call search more than once in one turn, "
    "but each call must use one focused query. After tool results, either call "
    "search again with a focused query or provide a short final answer. Do not "
    "invent tool results."
)
SEARCH_TOOL_NAME = "search"
RECOGNIZED_TAGS = ("think", "search", "information", "answer")
TAG_PATTERN = re.compile(r"</?(?:think|search|information|answer)>")
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
SEARCH_TAG_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)


class DeepSeekAPIError(RuntimeError):
    """An API error whose message is already redacted."""

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def _redact_text(value: Any, api_key: str | None = None) -> str:
    text = str(value)
    if api_key:
        text = text.replace(api_key, "[REDACTED_API_KEY]")
    text = re.sub(
        r"(?i)(bearer\s+|api[_ -]?key\s*[:=]\s*)([^\s,\"']+)",
        r"\1[REDACTED_API_KEY]",
        text,
    )
    return text


def _redact(value: Any, api_key: str | None = None) -> Any:
    """Make a JSON-safe value and recursively remove credential material."""

    value = _json_safe(value)
    if isinstance(value, str):
        return _redact_text(value, api_key)
    if isinstance(value, Mapping):
        return {str(key): _redact(item, api_key) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, api_key) for item in value]
    return value


def _json_line(value: Any, api_key: str | None = None) -> str:
    return json.dumps(_redact(value, api_key), ensure_ascii=False, separators=(",", ":"))


def _safe_error(exc: BaseException, api_key: str | None = None) -> str:
    return _redact_text(f"{type(exc).__name__}: {exc}", api_key)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reference_ids(reference_summary: Path) -> list[str]:
    if not reference_summary.is_file():
        raise FileNotFoundError(f"reference summary does not exist: {reference_summary}")
    payload = _load_json(reference_summary)
    ids = payload.get("dataset", {}).get("selected_question_ids")
    if not isinstance(ids, list) or len(ids) != 60 or len(set(map(str, ids))) != 60:
        raise ValueError("reference summary must contain exactly 60 unique selected_question_ids")
    selected = [str(item) for item in ids]
    counts = Counter(item.split(":", 1)[0] for item in selected)
    expected = Counter({source: 20 for source in DATASET_SOURCES})
    if counts != expected:
        raise ValueError(f"reference IDs are not 20 per source: {dict(counts)}")

    # The prior pilot is part of the pairing contract.  If its candidate file
    # is present, verify both sample indices for every reference question.
    candidate_path = reference_summary.parent / "candidates.jsonl"
    if candidate_path.is_file():
        candidate_ids: list[str] = []
        with candidate_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    candidate_ids.append(str(record.get("candidate_id", "")))
        expected_candidates = {
            f"{question_id}__sample_{sample_index}"
            for question_id in selected
            for sample_index in range(DEFAULT_SAMPLES_PER_QUESTION)
        }
        if set(candidate_ids) != expected_candidates or len(candidate_ids) != 120:
            raise ValueError("reference candidates.jsonl does not contain the exact 120 paired candidates")
    return selected


def _read_paired_questions(dataset_path: Path, selected_ids: Sequence[str]) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(dataset_path)
    required = {"data_source", "prompt", "reward_model"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    by_id: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        record = _row_question_record(row, row_index)
        question_id = str(record["question_id"])
        if question_id in by_id:
            raise ValueError(f"dataset has duplicate question ID: {question_id}")
        by_id[question_id] = record
    missing_ids = [question_id for question_id in selected_ids if question_id not in by_id]
    if missing_ids:
        raise ValueError(f"reference IDs missing from dataset: {missing_ids[:5]}")
    records = [by_id[question_id] for question_id in selected_ids]
    counts = Counter(str(item["data_source"]) for item in records)
    if counts != Counter({source: 20 for source in DATASET_SOURCES}):
        raise ValueError(f"paired dataset source counts are wrong: {dict(counts)}")
    return records


def _search_tool_schema(strict: bool = True) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": SEARCH_TOOL_NAME,
        "description": "Search the local evidence index with one focused query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _build_payload(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    allow_search: bool,
    strict_tools: bool,
    thinking_enabled: bool,
    reasoning_effort_enabled: bool,
    seed_enabled: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _json_safe(list(messages)),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if allow_search:
        payload["tools"] = [_search_tool_schema(strict=strict_tools)]
        payload["tool_choice"] = "auto"
    if thinking_enabled:
        payload["thinking"] = {"type": "enabled"}
    if reasoning_effort_enabled:
        payload["reasoning_effort"] = "high"
    if seed_enabled:
        payload["seed"] = seed
    return payload


def _optional_feature_to_disable(error_text: str) -> str | None:
    lowered = error_text.lower()
    optional_names = (
        ("reasoning_effort", "reasoning_effort"),
        ("thinking", "thinking"),
        ("strict", "strict_tools"),
        ("seed", "seed"),
    )
    if not any(token in lowered for token in ("unsupported", "unknown", "invalid", "not allowed", "unexpected")):
        return None
    for token, feature in optional_names:
        if token in lowered:
            return feature
    return None


class DeepSeekClient:
    def __init__(self, api_key: str, args: argparse.Namespace) -> None:
        self.api_key = api_key
        self.endpoint = args.base_url.rstrip("/") + "/chat/completions"
        self.model = args.model
        self.args = args
        self._feature_lock = threading.Lock()
        self.strict_tools = True
        self.thinking_enabled = True
        self.reasoning_effort_enabled = True
        self.seed_enabled = True

    def _feature_state(self) -> dict[str, bool]:
        with self._feature_lock:
            return {
                "strict_tools": self.strict_tools,
                "thinking_enabled": self.thinking_enabled,
                "reasoning_effort_enabled": self.reasoning_effort_enabled,
                "seed_enabled": self.seed_enabled,
            }

    def _disable_feature(self, feature: str) -> bool:
        with self._feature_lock:
            if feature == "strict_tools" and self.strict_tools:
                self.strict_tools = False
            elif feature == "thinking" and self.thinking_enabled:
                self.thinking_enabled = False
            elif feature == "reasoning_effort" and self.reasoning_effort_enabled:
                self.reasoning_effort_enabled = False
            elif feature == "seed" and self.seed_enabled:
                self.seed_enabled = False
            else:
                return False
        return True

    def _request_once(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], float, int]:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: DeepSeekAPIError | None = None
        started = time.monotonic()
        for attempt in range(self.args.retry_attempts):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=dict(payload),
                    timeout=self.args.api_timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = DeepSeekAPIError(_safe_error(exc, self.api_key), retryable=True)
                if attempt + 1 < self.args.retry_attempts:
                    delay = min(
                        self.args.retry_backoff_max,
                        self.args.retry_backoff * (2**attempt),
                    )
                    time.sleep(delay)
                    continue
                raise last_error

            status = int(response.status_code)
            if 200 <= status < 300:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise DeepSeekAPIError(
                        f"API returned non-JSON success response: {_safe_error(exc, self.api_key)}",
                        status_code=status,
                    ) from exc
                if not isinstance(body, dict):
                    raise DeepSeekAPIError("API returned a non-object JSON response", status_code=status)
                return body, time.monotonic() - started, attempt + 1

            try:
                error_body = response.json()
                error_text = _redact_text(json.dumps(error_body, ensure_ascii=False), self.api_key)
            except ValueError:
                error_text = _redact_text(response.text[:800], self.api_key)
            retryable = status == 429 or 500 <= status <= 599
            last_error = DeepSeekAPIError(
                f"DeepSeek HTTP {status}: {error_text[:800]}",
                status_code=status,
                retryable=retryable,
            )
            if retryable and attempt + 1 < self.args.retry_attempts:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after), self.args.retry_backoff_max) if retry_after else min(
                        self.args.retry_backoff_max,
                        self.args.retry_backoff * (2**attempt),
                    )
                except ValueError:
                    delay = min(self.args.retry_backoff_max, self.args.retry_backoff * (2**attempt))
                time.sleep(max(0.0, delay))
                continue
            raise last_error
        raise last_error or DeepSeekAPIError("DeepSeek request failed without a response")

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        allow_search: bool,
        seed: int,
    ) -> dict[str, Any]:
        # A 400 from an OpenAI-compatible deployment can mean that an optional
        # newer field is not supported.  Disable only the named optional field
        # and retry; transient errors use bounded exponential backoff above.
        for _ in range(5):
            features = self._feature_state()
            payload = _build_payload(
                messages,
                model=self.model,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                seed=seed,
                allow_search=allow_search,
                strict_tools=features["strict_tools"],
                thinking_enabled=features["thinking_enabled"],
                reasoning_effort_enabled=features["reasoning_effort_enabled"],
                seed_enabled=features["seed_enabled"],
            )
            try:
                body, latency, attempts = self._request_once(payload)
                return {
                    "body": _redact(body, self.api_key),
                    "latency_seconds": round(latency, 6),
                    "retry_attempts": attempts,
                    "features": features,
                }
            except DeepSeekAPIError as exc:
                feature = _optional_feature_to_disable(str(exc)) if exc.status_code == 400 else None
                if feature and self._disable_feature(feature):
                    continue
                raise
        raise DeepSeekAPIError("exhausted optional API feature fallbacks")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return "" if content is None else str(content).strip()


def _normalize_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            normalized.append({"index": index, "raw": _json_safe(call)})
            continue
        function = call.get("function", {})
        if not isinstance(function, Mapping):
            function = {}
        normalized.append(
            {
                "index": index,
                "id": str(call.get("id") or f"call_{index}"),
                "type": call.get("type", "function"),
                "function": {
                    "name": str(function.get("name", "")),
                    "arguments": function.get("arguments", ""),
                },
            }
        )
    return normalized


def _assistant_message(message: Mapping[str, Any], tool_calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    reasoning = message.get("reasoning_content")
    if reasoning is not None:
        result["reasoning_content"] = reasoning
    if tool_calls:
        # Rebuild the tool-call list from the normalized calls.  This gives
        # every subsequent tool message an exactly matching ID, even when an
        # OpenAI-compatible endpoint omits optional fields in its response.
        result["tool_calls"] = [
            {
                "id": str(call.get("id") or f"call_{call.get('index', index)}"),
                "type": str(call.get("type") or "function"),
                "function": {
                    "name": str((call.get("function") or {}).get("name", "")),
                    "arguments": _json_safe((call.get("function") or {}).get("arguments", "")),
                },
            }
            for index, call in enumerate(tool_calls)
            if isinstance(call, Mapping)
        ]
    return _json_safe(result)


def _concise_think(reasoning: Any, fallback: str) -> str:
    text = _content_text(reasoning)
    if not text:
        text = fallback
    text = re.sub(r"\s+", " ", text).strip()
    for tag in RECOGNIZED_TAGS:
        text = text.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
    text = text.strip()
    return text[:800] if text else "Use the available evidence to answer the question."


def _extract_final_answer(content: str) -> tuple[str | None, str | None, list[str]]:
    reasons: list[str] = []
    answer_matches = [match.strip() for match in ANSWER_TAG_PATTERN.findall(content) if match.strip()]
    think_matches = [match.strip() for match in THINK_TAG_PATTERN.findall(content) if match.strip()]
    search_matches = [match.strip() for match in SEARCH_TAG_PATTERN.findall(content) if match.strip()]
    if answer_matches:
        if search_matches:
            reasons.append("final_content_contains_search_tag")
        return answer_matches[-1], think_matches[-1] if think_matches else None, reasons
    if not content.strip():
        reasons.append("empty_final_content")
        return None, None, reasons
    if search_matches:
        reasons.append("final_content_contains_search_without_answer_tag")
    return content.strip(), think_matches[-1] if think_matches else None, reasons


def _parse_search_arguments(arguments: Any) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if isinstance(arguments, Mapping):
        payload = arguments
    else:
        try:
            payload = json.loads(str(arguments))
        except (TypeError, ValueError, json.JSONDecodeError):
            reasons.append("search_tool_arguments_not_valid_json")
            return None, reasons
    if not isinstance(payload, Mapping):
        reasons.append("search_tool_arguments_not_object")
        return None, reasons
    query = str(payload.get("query", "")).strip()
    if not query:
        reasons.append("empty_search_query")
        return None, reasons
    extra = set(payload) - {"query"}
    if extra:
        reasons.append("search_tool_arguments_contain_extra_fields")
    return query, reasons


def _project_turn_records(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten API responses into the project's one-search-per-turn format.

    The API may return multiple tool calls in one assistant message.  The
    conversation history must retain that assistant message as one message,
    while the training trajectory needs one ``search``/``information`` pair
    for every executed call.  ``project_turns`` is therefore the explicit
    bridge between the two representations.
    """

    projected: list[dict[str, Any]] = []
    for record in state.get("turn_records", []):
        explicit = record.get("project_turns")
        if isinstance(explicit, list) and explicit:
            projected.extend(dict(item) for item in explicit if isinstance(item, Mapping))
            continue

        search_calls = record.get("search_calls")
        if isinstance(search_calls, list) and search_calls:
            for call in search_calls:
                if not isinstance(call, Mapping) or not call.get("executed"):
                    continue
                raw_output = str(call.get("raw_output", "")).strip()
                if not raw_output:
                    continue
                projected.append(
                    {
                        "turn_index": record.get("turn_index", len(projected)),
                        "raw_output": raw_output,
                        "parse": call.get("parse", parse_turn(raw_output)),
                        "action": "search",
                        "query": call.get("query"),
                        "information": call.get("information"),
                        "generation": record.get("generation", {}),
                    }
                )
            continue

        projected.append(dict(record))
    return projected


def _assemble_project_trajectory(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for record in _project_turn_records(state):
        raw = str(record.get("raw_output", "")).strip()
        if raw:
            parts.append(raw)
        if record.get("information") is not None:
            parts.append(f"<information>{str(record.get('information') or '').strip()}</information>")
    return "\n\n".join(parts)


def _continuation_message(state: Mapping[str, Any], force_answer: bool) -> dict[str, str]:
    if force_answer:
        instruction = (
            "The previous assistant response was cut off before a usable final answer. "
            "Continue from the existing conversation and provide only a concise final answer; "
            "no more searches are allowed."
        )
    else:
        instruction = (
            "The previous assistant response was cut off before a usable final answer or search call. "
            "Continue from exactly where it stopped. If evidence is insufficient, call search once; "
            "otherwise provide only a concise final answer."
        )
    return {"role": "user", "content": instruction}


def _initial_messages(question: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                "Use the search tool for evidence. After each result, either search again or answer. "
                "When answering, provide only a concise final answer; do not include XML tags."
            ),
        },
    ]


def _next_user_message(state: Mapping[str, Any], force_answer: bool) -> dict[str, str]:
    history = _assemble_project_trajectory(state)
    if force_answer:
        instruction = (
            "No more searches are allowed. Use the observations and provide only a concise final answer."
        )
    else:
        instruction = (
            "Use the observations. If they are insufficient, call the search tool once with one focused "
            "query. Otherwise provide only a concise final answer."
        )
    return {
        "role": "user",
        "content": (
            f"Question: {state['question']}\n\n{instruction}\n\n"
            f"Searches already used: {state.get('search_count', 0)}\n\n"
            f"Conversation and actual observations so far:\n{history or '(none)'}"
        ),
    }


def _new_state(item: Mapping[str, Any], sample_index: int, seed: int) -> dict[str, Any]:
    return {
        **dict(item),
        "candidate_id": f"{item['question_id']}__sample_{sample_index}",
        "sample_index": sample_index,
        "seed": seed,
        "status": "active",
        "ended_with_answer": False,
        "search_count": 0,
        "final_answer": None,
        "turn_records": [],
        "retrieval_outputs": [],
        "state_failure_reasons": [],
        "conversion_valid": True,
        "_messages": _initial_messages(str(item["question"])),
        "_pending_search_turns": [],
        "force_answer": False,
        "continuation_count": 0,
        "truncation_exhausted": False,
        "skip_next_user_prompt": False,
    }


def _add_failure(state: dict[str, Any], *reasons: str) -> None:
    for reason in reasons:
        if reason and reason not in state["state_failure_reasons"]:
            state["state_failure_reasons"].append(reason)


def _api_event_from_record(state: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "api_response",
        "candidate_id": state["candidate_id"],
        "question_id": state["question_id"],
        "sample_index": state["sample_index"],
        "seed": state["seed"],
        "turn_index": record["turn_index"],
        "assistant_message": record["assistant_message"],
        "reasoning_content": record.get("reasoning_content"),
        "tool_calls": record.get("tool_calls", []),
        "search_calls": record.get("search_calls", []),
        "project_turns": record.get("project_turns", []),
        "final_content": record.get("final_content"),
        "final_answer": record.get("final_answer"),
        "finish_reason": record.get("finish_reason"),
        "usage": record.get("usage", {}),
        "latency_seconds": record.get("latency_seconds"),
        "retry_attempts": record.get("retry_attempts"),
        "features": record.get("features", {}),
        "response_metadata": record.get("response_metadata", {}),
        "raw_output": record.get("raw_output", ""),
        "parse": record.get("parse", {}),
        "conversion_reasons": record.get("conversion_reasons", []),
        "action": record.get("action"),
        "query": record.get("query"),
        "continuation_index": record.get("continuation_index", 0),
        "continuation_requested": record.get("continuation_requested", False),
        "continuation_message": record.get("continuation_message"),
        "force_answer": record.get("force_answer", False),
        "conversion_valid": record.get("conversion_valid", True),
        "generation": record.get("generation", {}),
        "truncation_exhausted": record.get("truncation_exhausted", False),
    }


def _retrieval_event_from_record(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
    call: Mapping[str, Any],
    search_call_index: int,
) -> dict[str, Any]:
    return {
        "event_type": "retrieval_observation",
        "candidate_id": state["candidate_id"],
        "question_id": state["question_id"],
        "sample_index": state["sample_index"],
        "turn_index": record["turn_index"],
        "search_call_index": search_call_index,
        "tool_call_id": call.get("tool_call_id"),
        "query": call.get("query"),
        "topk": call.get("retrieval_topk"),
        "result": call.get("retrieval_result"),
        "information": call.get("information"),
        "error": call.get("retrieval_error"),
        "executed": bool(call.get("executed")),
        "skip_reason": call.get("skip_reason"),
        "tool_message": call.get("tool_message"),
    }


def _record_from_api_event(event: Mapping[str, Any]) -> dict[str, Any]:
    search_calls = event.get("search_calls")
    if not isinstance(search_calls, list):
        search_calls = []
        if event.get("action") == "search":
            tool_calls = event.get("tool_calls") or []
            call_id = None
            if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], Mapping):
                call_id = tool_calls[0].get("id")
            search_calls = [
                {
                    "tool_call_id": call_id or "call_0",
                    "query": event.get("query"),
                    "valid": True,
                    "executed": False,
                    "raw_output": event.get("raw_output", ""),
                    "parse": event.get("parse", {}),
                }
            ]
    return {
        "turn_index": int(event.get("turn_index", 0)),
        "assistant_message": event.get("assistant_message", {}),
        "reasoning_content": event.get("reasoning_content"),
        "tool_calls": event.get("tool_calls", []),
        "search_calls": search_calls,
        "project_turns": event.get("project_turns", []),
        "final_content": event.get("final_content"),
        "finish_reason": event.get("finish_reason"),
        "usage": event.get("usage", {}),
        "latency_seconds": event.get("latency_seconds"),
        "retry_attempts": event.get("retry_attempts"),
        "features": event.get("features", {}),
        "response_metadata": event.get("response_metadata", {}),
        "raw_output": event.get("raw_output", ""),
        "parse": event.get("parse", {}),
        "conversion_reasons": event.get("conversion_reasons", []),
        "action": event.get("action"),
        "query": event.get("query"),
        "final_answer": event.get("final_answer"),
        "continuation_index": int(event.get("continuation_index", 0) or 0),
        "continuation_requested": bool(event.get("continuation_requested", False)),
        "continuation_message": event.get("continuation_message"),
        "force_answer": bool(event.get("force_answer", False)),
        "conversion_valid": bool(event.get("conversion_valid", True)),
        "truncation_exhausted": bool(event.get("truncation_exhausted", False)),
        "information": None,
        "retrieval_result": None,
        "retrieval_topk": None,
        "retrieval_error": None,
        "tool_message": None,
        "generation": {
            "hit_max_new_tokens": event.get("finish_reason") == "length",
            "max_new_tokens": event.get("generation", {}).get("max_new_tokens")
            if isinstance(event.get("generation"), Mapping)
            else None,
        },
    }


def _replay_state(
    item: Mapping[str, Any],
    sample_index: int,
    seed: int,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    state = _new_state(item, sample_index, seed)
    pending_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for event in events:
        event_type = event.get("event_type")
        if event_type == "api_response":
            record = _record_from_api_event(event)
            turn_index = record["turn_index"]
            state["turn_records"].append(record)
            state["_messages"].append(record["assistant_message"])
            state["conversion_valid"] = state["conversion_valid"] and bool(record.get("conversion_valid", True))
            state["truncation_exhausted"] = state["truncation_exhausted"] or bool(
                record.get("truncation_exhausted", False)
            )
            if record.get("force_answer"):
                state["force_answer"] = True
            if record.get("action") == "answer":
                state["status"] = "answered"
                state["ended_with_answer"] = bool(record.get("final_answer") or record.get("final_content"))
                state["final_answer"] = record.get("final_answer") or record.get("final_content")
            elif record.get("action") == "failure" and not record.get("continuation_requested"):
                state["status"] = "failed"
            if record.get("continuation_requested"):
                continuation_message = record.get("continuation_message")
                if isinstance(continuation_message, Mapping):
                    state["_messages"].append(dict(continuation_message))
                state["continuation_count"] = max(
                    state.get("continuation_count", 0),
                    int(record.get("continuation_index", 0) or 0),
                )
                state["skip_next_user_prompt"] = True
                state["status"] = "active"
            for call in record.get("search_calls", []):
                if isinstance(call, Mapping):
                    # Keep the object stored in the record itself so replayed
                    # retrieval events populate the same search-call records
                    # later used to assemble the project trajectory.
                    call_ref = call if isinstance(call, dict) else dict(call)
                    call_id = str(call_ref.get("tool_call_id") or call_ref.get("id") or "call_0")
                    call_ref["tool_call_id"] = call_id
                    call_ref.setdefault("processed", False)
                    pending_by_key[(turn_index, call_id)] = call_ref
        elif event_type == "retrieval_observation":
            turn_index = int(event.get("turn_index", -1))
            tool_call_id = str(event.get("tool_call_id") or "")
            key = (turn_index, tool_call_id)
            call = pending_by_key.get(key)
            if call is None:
                candidates = [
                    (candidate_key, candidate)
                    for candidate_key, candidate in pending_by_key.items()
                    if candidate_key[0] == turn_index and not candidate.get("processed")
                ]
                if len(candidates) == 1:
                    key, call = candidates[0]
            if call is None:
                _add_failure(state, "orphaned_retrieval_event")
                continue
            call["information"] = event.get("information")
            call["retrieval_result"] = event.get("result")
            call["retrieval_topk"] = event.get("topk")
            call["retrieval_error"] = event.get("error")
            call["tool_message"] = event.get("tool_message")
            call["executed"] = bool(event.get("executed", call.get("executed", False)))
            call["skip_reason"] = event.get("skip_reason")
            call["processed"] = True
            state["retrieval_outputs"].append(
                {
                    "search_turn_index": turn_index,
                    "search_call_index": event.get("search_call_index"),
                    "tool_call_id": call.get("tool_call_id"),
                    "query": event.get("query"),
                    "topk": event.get("topk"),
                    "result": event.get("result"),
                    "information": event.get("information"),
                    "error": event.get("error"),
                    "executed": bool(event.get("executed", call.get("executed", False))),
                    "skip_reason": event.get("skip_reason"),
                }
            )
            if call.get("executed"):
                state["search_count"] += 1
            tool_message = event.get("tool_message")
            if isinstance(tool_message, Mapping):
                state["_messages"].append(dict(tool_message))
            if event.get("error"):
                state["status"] = "failed"

    state["_pending_search_turns"] = [
        call for call in pending_by_key.values() if not call.get("processed")
    ]
    if state["_pending_search_turns"]:
        state["status"] = "active"
    return state


def _extract_choice(body: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], str | None, Mapping[str, Any]]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise DeepSeekAPIError("API response has no usable choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise DeepSeekAPIError("API response choice has no message object")
    finish_reason = choice.get("finish_reason")
    usage = body.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    metadata = {key: value for key, value in body.items() if key not in {"choices", "usage"}}
    return message, choice, str(finish_reason) if finish_reason is not None else None, {
        "usage": _json_safe(usage),
        "metadata": _json_safe(metadata),
    }


def _convert_response(
    state: Mapping[str, Any],
    body: Mapping[str, Any],
    *,
    finish_reason: str | None,
    latency_seconds: float,
    retry_attempts: int,
    features: Mapping[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    message, _choice, _finish_reason, response_parts = _extract_choice(body)
    tool_calls = _normalize_tool_calls(message)
    content = _content_text(message.get("content"))
    reasoning = message.get("reasoning_content")
    assistant = _assistant_message(message, tool_calls)
    conversion_reasons: list[str] = []
    action = "failure"
    query: str | None = None
    final_answer: str | None = None
    raw_output = ""
    think_fallback = "Review the question and available evidence."
    search_calls: list[dict[str, Any]] = []
    conversion_valid = True

    if finish_reason == "length":
        conversion_reasons.append("generation_truncated")
    if len(tool_calls) > 1:
        conversion_reasons.append("multiple_tool_calls_in_one_turn")
    if tool_calls:
        for call in tool_calls:
            function = call.get("function", {})
            name = str(function.get("name", "")) if isinstance(function, Mapping) else ""
            call_reasons: list[str] = []
            if name != SEARCH_TOOL_NAME:
                call_reasons.append("unexpected_tool_name")
            call_query: str | None = None
            if name == SEARCH_TOOL_NAME:
                call_query, argument_reasons = _parse_search_arguments(
                    function.get("arguments") if isinstance(function, Mapping) else None
                )
                call_reasons.extend(argument_reasons)
            valid = bool(name == SEARCH_TOOL_NAME and call_query and not call_reasons)
            call_raw = (
                f"<think>{_concise_think(reasoning, think_fallback)}</think>"
                f"<search>{call_query}</search>"
                if valid and call_query
                else ""
            )
            call_record = {
                "tool_call_id": str(call.get("id") or f"call_{call.get('index', len(search_calls))}"),
                "query": call_query,
                "valid": valid,
                "executed": False,
                "processed": False,
                "skip_reason": None,
                "raw_output": call_raw,
                "parse": parse_turn(call_raw) if call_raw else {
                    "action": "failure",
                    "strict": False,
                    "query": call_query,
                    "answer": None,
                    "reasons": call_reasons,
                },
                "information": None,
                "retrieval_result": None,
                "retrieval_topk": None,
                "retrieval_error": None,
                "tool_message": None,
                "retrieval_seconds": None,
                "reasons": call_reasons,
            }
            search_calls.append(call_record)
            conversion_reasons.extend(call_reasons)
            if not valid:
                conversion_valid = False
        valid_calls = [call for call in search_calls if call.get("valid")]
        if valid_calls:
            action = "search"
            query = str(valid_calls[0]["query"])
            raw_output = str(valid_calls[0]["raw_output"])
    else:
        # A plain content fragment returned with finish_reason=length is not
        # safe to treat as the final answer.  Only an explicitly closed
        # answer tag is considered complete in that case; otherwise the
        # caller will request a bounded continuation.
        if finish_reason == "length" and not ANSWER_TAG_PATTERN.search(content):
            conversion_reasons.append("length_without_complete_answer")
            final_answer = None
            content_think = None
        else:
            final_answer, content_think, final_reasons = _extract_final_answer(content)
            conversion_reasons.extend(final_reasons)
            if final_answer:
                action = "answer"
                raw_output = (
                    f"<think>{_concise_think(reasoning or content_think, 'The evidence is sufficient.')}</think>"
                    f"<answer>{final_answer}</answer>"
                )

    parsed = parse_turn(raw_output) if raw_output else {
        "action": "failure",
        "strict": False,
        "query": query,
        "answer": final_answer,
        "reasons": list(conversion_reasons),
    }
    generation = {
        "hit_max_new_tokens": finish_reason == "length",
        "max_new_tokens": max_new_tokens,
    }
    return {
        "turn_index": len(state.get("turn_records", [])),
        "assistant_message": assistant,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "final_content": content,
        "finish_reason": finish_reason,
        "usage": response_parts["usage"],
        "latency_seconds": latency_seconds,
        "retry_attempts": retry_attempts,
        "features": dict(features),
        "response_metadata": response_parts["metadata"],
        "raw_output": raw_output,
        "parse": parsed,
        "conversion_reasons": list(dict.fromkeys(conversion_reasons)),
        "action": action,
        "query": query,
        "final_answer": final_answer,
        "search_calls": search_calls,
        "project_turns": [],
        "continuation_index": int(state.get("continuation_count", 0)),
        "continuation_requested": False,
        "continuation_message": None,
        "force_answer": bool(state.get("force_answer", False)),
        "conversion_valid": conversion_valid,
        "information": None,
        "retrieval_result": None,
        "retrieval_topk": None,
        "retrieval_error": None,
        "tool_message": None,
        "generation": generation,
    }


def _finalize_state(state: dict[str, Any]) -> None:
    validation_records: list[dict[str, Any]] = []
    for record in _project_turn_records(state):
        copied = dict(record)
        generation = dict(copied.get("generation", {}))
        # A length-terminated intermediate response is recoverable when a
        # later continuation produces the final answer.  The base validator
        # treats every length flag as fatal, so hide only those recoverable
        # flags from the project-format check while retaining them in the raw
        # API record for auditability.
        if state.get("ended_with_answer") and not state.get("truncation_exhausted"):
            generation["hit_max_new_tokens"] = False
        copied["generation"] = generation
        validation_records.append(copied)
    validation_state = dict(state)
    validation_state["turn_records"] = validation_records
    format_correct, format_reasons = validate_trajectory(validation_state, max_searches=DEFAULT_MAX_SEARCHES)
    if not state.get("conversion_valid", True):
        format_correct = False
        format_reasons = list(format_reasons) + ["api_to_xml_conversion_failed"]
    answer = state.get("final_answer")
    answer_correct = answer_em(answer, state.get("gold_answers", []))
    failure_reasons = list(state.get("state_failure_reasons", []))
    failure_reasons.extend(format_reasons)
    for record in state.get("turn_records", []):
        failure_reasons.extend(
            reason
            for reason in record.get("conversion_reasons", [])
            if reason not in {"generation_truncated", "length_without_complete_answer"}
        )
        if record.get("generation", {}).get("hit_max_new_tokens") and state.get("truncation_exhausted"):
            failure_reasons.append("generation_truncated")
    if answer is None:
        failure_reasons.append("final_answer_extraction_failed")
    elif not answer_correct:
        failure_reasons.append("final_answer_em_failed")
    state["format_correct"] = bool(format_correct)
    state["answer_correct"] = bool(answer_correct)
    state["eligible"] = bool(format_correct and answer_correct)
    state["format_failure_reasons"] = list(dict.fromkeys(format_reasons))
    state["failure_reasons"] = list(dict.fromkeys(failure_reasons))
    state["trajectory"] = _assemble_project_trajectory(state)


def _retrieve_one(args: argparse.Namespace, query: str) -> tuple[Any, str | None, str]:
    from generate_multihop_sft_teacher_pilot import _retrieve_batch

    started = time.monotonic()
    try:
        result = _retrieve_batch(
            args.retrieval_url,
            [query],
            args.retrieval_topk,
            args.retrieval_timeout,
        )[0]
        return result, None, f"{time.monotonic() - started:.6f}"
    except Exception as exc:
        return None, _safe_error(exc), f"{time.monotonic() - started:.6f}"


class JSONLAppender:
    def __init__(self, path: Path, *, append: bool, api_key: str | None) -> None:
        self.path = path
        self.api_key = api_key
        self.lock = threading.Lock()
        self.handle = path.open("a" if append else "w", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        line = _json_line(record, self.api_key) + "\n"
        with self.lock:
            self.handle.write(line)
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        with self.lock:
            self.handle.close()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            records.append(value)
    return records


def _records_for_output(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for state in states:
        output.append(
            {
                "candidate_id": state["candidate_id"],
                "question_id": state["question_id"],
                "row_index": state["row_index"],
                "data_source": state["data_source"],
                "question": state["question"],
                "sample_index": state["sample_index"],
                "seed": state["seed"],
                "gold_answers": state["gold_answers"],
                "search_count": state["search_count"],
                "status": state["status"],
                "ended_with_answer": state["ended_with_answer"],
                "raw_generation_turns": state["turn_records"],
                "retrieval_outputs": state["retrieval_outputs"],
                "trajectory": state.get("trajectory", ""),
                "extracted_answer": state.get("final_answer"),
                "format_correct": state.get("format_correct", False),
                "answer_correct": state.get("answer_correct", False),
                "eligible": state.get("eligible", False),
                "format_failure_reasons": state.get("format_failure_reasons", []),
                "failure_reasons": state.get("failure_reasons", []),
            }
        )
    return output


def _aggregate_numeric(values: Iterable[Any]) -> dict[str, float | int]:
    totals: dict[str, float | int] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            key = prefix or "value"
            totals[key] = totals.get(key, 0) + value
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)

    for value in values:
        visit("", value)
    return totals


def _reported_billing_fields(values: Iterable[Any]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                full_key = f"{prefix}.{key}" if prefix else str(key)
                if re.search(r"cost|price|bill|currency", full_key, re.IGNORECASE):
                    result[full_key] = _json_safe(item)
                else:
                    visit(full_key, item)

    for value in values:
        visit("", value)
    return result or None


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(records)
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record["question_id"])].append(record)
    answer_count = sum(bool(record.get("answer_correct")) for record in records)
    format_count = sum(bool(record.get("format_correct")) for record in records)
    eligible_count = sum(bool(record.get("eligible")) for record in records)
    answer_question_count = sum(any(item.get("answer_correct") for item in group) for group in by_question.values())
    eligible_question_count = sum(any(item.get("eligible") for item in group) for group in by_question.values())
    failures: Counter[str] = Counter()
    search_counts: Counter[str] = Counter()
    usages: list[Any] = []
    latencies: list[float] = []
    feature_counts: Counter[str] = Counter()
    for record in records:
        failures.update(str(item) for item in record.get("failure_reasons", []))
        search_counts[str(record.get("search_count", 0))] += 1
        for turn in record.get("raw_generation_turns", []):
            usages.append(turn.get("usage", {}))
            if isinstance(turn.get("latency_seconds"), (int, float)):
                latencies.append(float(turn["latency_seconds"]))
            for feature, enabled in (turn.get("features") or {}).items():
                feature_counts[f"{feature}={bool(enabled)}"] += 1
    usage_totals = _aggregate_numeric(usages)
    return {
        "trajectory_count": denominator,
        "answer_pass_at_1": {"numerator": answer_count, "denominator": denominator, "rate": answer_count / denominator if denominator else 0.0},
        "answer_em": {"numerator": answer_count, "denominator": denominator, "rate": answer_count / denominator if denominator else 0.0},
        "format": {"numerator": format_count, "denominator": denominator, "rate": format_count / denominator if denominator else 0.0},
        "format_rate": format_count / denominator if denominator else 0.0,
        "eligible": {"numerator": eligible_count, "denominator": denominator, "rate": eligible_count / denominator if denominator else 0.0},
        "eligible_rate": eligible_count / denominator if denominator else 0.0,
        "answer_pass_at_2_observed": {"numerator": answer_question_count, "denominator": len(by_question), "rate": answer_question_count / len(by_question) if by_question else 0.0},
        "question_level_answer_pass_at_2_observed_any_success": {"numerator": answer_question_count, "denominator": len(by_question), "rate": answer_question_count / len(by_question) if by_question else 0.0},
        "eligible_pass_at_2_observed": {"numerator": eligible_question_count, "denominator": len(by_question), "rate": eligible_question_count / len(by_question) if by_question else 0.0},
        "question_level_eligible_pass_at_2_observed_any_success": {"numerator": eligible_question_count, "denominator": len(by_question), "rate": eligible_question_count / len(by_question) if by_question else 0.0},
        "search_count_distribution": dict(sorted(search_counts.items(), key=lambda item: int(item[0]))),
        "failure_reason_counts": dict(failures.most_common()),
        "token_usage_totals": usage_totals,
        "reported_billing_fields": _reported_billing_fields(usages),
        "latency_seconds": {
            "sum": round(sum(latencies), 6),
            "mean_per_api_response": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
            "api_response_count": len(latencies),
        },
        "feature_usage": dict(feature_counts),
    }


def _candidate_state_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(record)
    state["state_failure_reasons"] = list(record.get("failure_reasons", []))
    state["conversion_valid"] = bool(record.get("format_correct", False))
    state["_messages"] = []
    state["_pending_search_turns"] = []
    return state


def _tool_message_for_call(call: Mapping[str, Any]) -> dict[str, str]:
    tool_call_id = str(call.get("tool_call_id") or "call_0")
    if call.get("executed"):
        content = str(call.get("information") or "")
    else:
        reason = str(call.get("skip_reason") or "search call was not executed")
        content = f"Search not executed: {reason}."
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": SEARCH_TOOL_NAME,
        "content": content,
    }


def _process_pending_searches(
    state: dict[str, Any],
    args: argparse.Namespace,
    event_writer: JSONLAppender,
) -> None:
    """Execute planned calls and append one tool result for every call ID."""

    pending = list(state.get("_pending_search_turns", []))
    if not pending:
        return
    records_by_turn = {
        int(record.get("turn_index", -1)): record
        for record in state.get("turn_records", [])
    }
    for search_call_index, call in enumerate(pending):
        if not isinstance(call, dict):
            continue
        parent = records_by_turn.get(int(call.get("turn_index", -1)))
        if parent is None:
            _add_failure(state, "orphaned_pending_search_call")
            continue

        if call.get("execute") and call.get("valid"):
            result, retrieval_error, retrieval_seconds = _retrieve_one(args, str(call.get("query") or ""))
            information = _format_information(result) if retrieval_error is None else ""
            call.update(
                {
                    "executed": True,
                    "processed": True,
                    "information": information,
                    "retrieval_result": result,
                    "retrieval_topk": args.retrieval_topk,
                    "retrieval_error": retrieval_error,
                    "retrieval_seconds": retrieval_seconds,
                }
            )
            state["search_count"] += 1
            if retrieval_error is not None:
                _add_failure(state, "retrieval_failed", retrieval_error)
            project_turn = {
                "turn_index": parent["turn_index"],
                "raw_output": call.get("raw_output", ""),
                "parse": call.get("parse", {}),
                "action": "search",
                "query": call.get("query"),
                "information": information,
                "generation": parent.get("generation", {}),
            }
            parent.setdefault("project_turns", []).append(project_turn)
            state["retrieval_outputs"].append(
                {
                    "search_turn_index": parent["turn_index"],
                    "search_call_index": search_call_index,
                    "tool_call_id": call.get("tool_call_id"),
                    "query": call.get("query"),
                    "topk": args.retrieval_topk,
                    "result": result,
                    "information": information,
                    "error": retrieval_error,
                    "executed": True,
                    "skip_reason": None,
                }
            )
        else:
            call.setdefault("processed", True)
            call["tool_message"] = None

        tool_message = _tool_message_for_call(call)
        call["tool_message"] = tool_message
        state["_messages"].append(tool_message)
        event_writer.append(_retrieval_event_from_record(state, parent, call, search_call_index))

    state["_pending_search_turns"] = []
    if any(call.get("skip_reason") == "search_budget_exhausted" for call in pending):
        state["force_answer"] = True
        _add_failure(state, "tool_call_budget_clipped")
    if any(call.get("retrieval_error") for call in pending):
        state["status"] = "failed"


def _run_candidate(
    item: Mapping[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    client: DeepSeekClient,
    event_writer: JSONLAppender,
    prior_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    state = (
        _replay_state(item, sample_index, args.seed, prior_events)
        if prior_events
        else _new_state(item, sample_index, args.seed)
    )
    if state.get("status") in {"answered", "failed"} and not state.get("_pending_search_turns"):
        _finalize_state(state)
        return state

    while state["status"] == "active":
        if state.get("_pending_search_turns"):
            _process_pending_searches(state, args, event_writer)
            continue

        allow_search = not state.get("force_answer", False) and state["search_count"] < args.max_searches
        if state.pop("skip_next_user_prompt", False):
            # The continuation request was already appended immediately after
            # the truncated assistant message.  Adding another user message
            # here would create an invalid/duplicated continuation history.
            pass
        else:
            state["_messages"].append(_next_user_message(state, force_answer=not allow_search))
        try:
            response = client.complete(
                state["_messages"],
                allow_search=allow_search,
                seed=args.seed,
            )
            body = response["body"]
            message, _choice, finish_reason, _parts = _extract_choice(body)
            record = _convert_response(
                state,
                body,
                finish_reason=finish_reason,
                latency_seconds=response["latency_seconds"],
                retry_attempts=response["retry_attempts"],
                features=response["features"],
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            _add_failure(state, "deepseek_api_failed", _safe_error(exc, client.api_key))
            state["status"] = "failed"
            break

        state["turn_records"].append(record)
        state["_messages"].append(record["assistant_message"])
        _add_failure(
            state,
            *(
                reason
                for reason in record.get("conversion_reasons", [])
                if reason not in {"generation_truncated", "length_without_complete_answer"}
            ),
        )
        state["conversion_valid"] = state["conversion_valid"] and bool(record.get("conversion_valid", True))

        if record["action"] == "search" and record.get("search_calls"):
            remaining = max(0, args.max_searches - state["search_count"])
            planned_calls: list[dict[str, Any]] = []
            for call in record["search_calls"]:
                if not isinstance(call, dict):
                    continue
                call["turn_index"] = record["turn_index"]
                if call.get("valid") and remaining > 0:
                    call["execute"] = True
                    remaining -= 1
                elif call.get("valid"):
                    call["execute"] = False
                    call["skip_reason"] = "search_budget_exhausted"
                    record.setdefault("conversion_reasons", []).append("tool_call_budget_clipped")
                else:
                    call["execute"] = False
                    call["skip_reason"] = "invalid_search_tool_call"
                planned_calls.append(call)
            if any(call.get("skip_reason") == "search_budget_exhausted" for call in planned_calls):
                state["force_answer"] = True
            record["force_answer"] = state.get("force_answer", False)
            state["_pending_search_turns"] = planned_calls
        elif record["action"] == "answer" and record.get("final_content"):
            answer = record.get("final_answer")
            if not answer:
                answer, _content_think, _reasons = _extract_final_answer(str(record.get("final_content", "")))
            state["status"] = "answered"
            state["ended_with_answer"] = bool(answer)
            state["final_answer"] = answer
        elif record["finish_reason"] == "length" and state.get("continuation_count", 0) < args.max_continuations:
            state["continuation_count"] += 1
            continuation_message = _continuation_message(state, force_answer=not allow_search)
            record["continuation_index"] = state["continuation_count"]
            record["continuation_requested"] = True
            record["continuation_message"] = continuation_message
            state["_messages"].append(continuation_message)
            state["skip_next_user_prompt"] = True
            state["status"] = "active"
        elif record["finish_reason"] == "length":
            state["truncation_exhausted"] = True
            _add_failure(state, "continuation_exhausted", "generation_truncated")
            state["conversion_valid"] = False
            record["truncation_exhausted"] = True
            state["status"] = "failed"
        else:
            state["conversion_valid"] = False
            _add_failure(state, "no_valid_search_or_final_answer")
            state["status"] = "failed"

        # Persist the API response only after continuation/budget decisions
        # are attached, so resume can reconstruct the exact valid message
        # history without another paid request.
        event_writer.append(_api_event_from_record(state, record))

    _finalize_state(state)
    return state


def _write_json(path: Path, value: Any, api_key: str | None = None) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_redact(value, api_key), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _prepare_output_dir(output_dir: Path, *, overwrite: bool, resume: bool) -> None:
    files = [output_dir / name for name in ("interactions.jsonl", "candidates.jsonl", "accepted.jsonl", "summary.json")]
    existing = [path for path in files if path.exists()]
    if existing and not overwrite and not resume:
        raise FileExistsError(
            "output files already exist; pass --resume to continue or --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _self_test_reference_and_metrics() -> dict[str, Any]:
    reference_ids = _reference_ids(Path(DEFAULT_REFERENCE_SUMMARY))
    candidate_ids = [
        f"{question_id}__sample_{sample_index}"
        for question_id in reference_ids
        for sample_index in range(DEFAULT_SAMPLES_PER_QUESTION)
    ]
    assert len(candidate_ids) == 120 and len(set(candidate_ids)) == 120
    assert Counter(question_id.split(":", 1)[0] for question_id in reference_ids) == Counter(
        {source: 20 for source in DATASET_SOURCES}
    )

    item = {
        "question_id": "hotpotqa:self_test",
        "row_index": 0,
        "data_source": "hotpotqa",
        "question": "Which city is in France?",
        "gold_answers": ["Paris"],
    }
    state = _new_state(item, 0, DEFAULT_SEED)
    search_record = {
        "turn_index": 0,
        "assistant_message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_0"}]},
        "reasoning_content": "Find the city.",
        "tool_calls": [{"id": "call_0", "function": {"name": "search", "arguments": '{"query":"city France"}'}}],
        "final_content": "",
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "latency_seconds": 0.1,
        "retry_attempts": 1,
        "features": {},
        "response_metadata": {},
        "raw_output": "<think>Find the city.</think><search>city France</search>",
        "parse": parse_turn("<think>Find the city.</think><search>city France</search>"),
        "conversion_reasons": [],
        "action": "search",
        "query": "city France",
        "information": "Doc 1(Title: Paris) Paris is in France.",
        "generation": {"hit_max_new_tokens": False},
    }
    answer_record = {
        "turn_index": 1,
        "assistant_message": {"role": "assistant", "content": "Paris"},
        "reasoning_content": "The document identifies Paris.",
        "tool_calls": [],
        "final_content": "Paris",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        "latency_seconds": 0.2,
        "retry_attempts": 1,
        "features": {},
        "response_metadata": {},
        "raw_output": "<think>The document identifies Paris.</think><answer>Paris</answer>",
        "parse": parse_turn("<think>The document identifies Paris.</think><answer>Paris</answer>"),
        "conversion_reasons": [],
        "action": "answer",
        "query": None,
        "information": None,
        "generation": {"hit_max_new_tokens": False},
    }
    state["turn_records"] = [search_record, answer_record]
    state["search_count"] = 1
    state["status"] = "answered"
    state["ended_with_answer"] = True
    state["final_answer"] = "Paris"
    state["retrieval_outputs"] = [{"search_turn_index": 0, "query": "city France", "topk": 3, "result": [], "information": search_record["information"], "error": None}]
    _finalize_state(state)
    assert state["format_correct"] and state["answer_correct"] and state["eligible"]
    records = _records_for_output([state, {**state, "candidate_id": "second", "answer_correct": False, "eligible": False}])
    metrics = _metrics(records)
    assert metrics["answer_pass_at_1"]["numerator"] == 1
    assert metrics["answer_pass_at_2_observed"]["numerator"] == 1
    assert metrics["eligible_pass_at_2_observed"]["numerator"] == 1

    parity_cases = ["The U.S. Constitution", "An apple, a pear!", "Mixed CASE -- text", "C++ and Python"]
    from verl.utils.reward_score.qa_em import normalize_answer as verl_normalize_answer

    for case in parity_cases:
        assert normalize_answer(case) == verl_normalize_answer(case), case

    sentinel = "TEST_SECRET_SENTINEL_7B3F"
    payload = _build_payload(
        [{"role": "user", "content": "question"}],
        model=DEFAULT_MODEL,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        allow_search=True,
        strict_tools=True,
        thinking_enabled=True,
        reasoning_effort_enabled=True,
        seed_enabled=True,
    )
    serialized = _json_line({"payload": payload, "error": f"Bearer {sentinel}"}, sentinel)
    assert sentinel not in serialized
    assert sentinel not in _json_line(payload, sentinel)
    assert "DEEPSEEK_API_KEY" not in serialized

    # The following tests exercise the API-response state machine without any
    # network or retriever call.  They intentionally use only in-memory mocks.
    class _MemoryEventWriter:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def append(self, record: Mapping[str, Any]) -> None:
            self.events.append(_json_safe(record))

    class _MockClient:
        api_key = "MOCK_KEY_MUST_NOT_BE_SERIALIZED"

        def __init__(self, bodies: Sequence[Mapping[str, Any]]) -> None:
            self.bodies = list(bodies)
            self.calls: list[dict[str, Any]] = []

        def complete(self, messages: Sequence[Mapping[str, Any]], *, allow_search: bool, seed: int) -> dict[str, Any]:
            assert self.bodies, "mock API ran out of responses"
            self.calls.append({"messages": _json_safe(list(messages)), "allow_search": allow_search, "seed": seed})
            return {
                "body": dict(self.bodies.pop(0)),
                "latency_seconds": 0.001,
                "retry_attempts": 1,
                "features": {},
            }

    def _mock_body(message: Mapping[str, Any], finish_reason: str) -> dict[str, Any]:
        return {
            "choices": [{"index": 0, "message": dict(message), "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }

    def _mock_args(*, max_searches: int = 3, max_continuations: int = 2) -> argparse.Namespace:
        return argparse.Namespace(
            max_searches=max_searches,
            max_continuations=max_continuations,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            retrieval_url="mock://retriever",
            retrieval_topk=3,
            retrieval_timeout=1.0,
            seed=DEFAULT_SEED,
        )

    def _mock_item(question_id: str = "hotpotqa:mock") -> dict[str, Any]:
        return {
            "question_id": question_id,
            "row_index": 0,
            "data_source": "hotpotqa",
            "question": "Which city is in France?",
            "gold_answers": ["Paris"],
        }

    original_retrieve_one = _retrieve_one

    def _mock_retrieve_one(args: argparse.Namespace, query: str) -> tuple[Any, str | None, str]:
        return ([{"document": {"contents": f"{query}\nEvidence for {query}."}}], None, "0.001")

    globals()["_retrieve_one"] = _mock_retrieve_one
    try:
        multi_calls = [
            {
                "id": "call_alpha",
                "type": "function",
                "function": {"name": SEARCH_TOOL_NAME, "arguments": '{"query":"alpha"}'},
            },
            {
                "id": "call_beta",
                "type": "function",
                "function": {"name": SEARCH_TOOL_NAME, "arguments": '{"query":"beta"}'},
            },
        ]
        multi_client = _MockClient(
            [
                _mock_body({"role": "assistant", "content": None, "reasoning_content": "Search both.", "tool_calls": multi_calls}, "tool_calls"),
                _mock_body({"role": "assistant", "content": "Paris", "reasoning_content": "Answer from evidence."}, "stop"),
            ]
        )
        multi_writer = _MemoryEventWriter()
        multi_state = _run_candidate(_mock_item(), 0, _mock_args(), multi_client, multi_writer, [])
        retrieval_events = [event for event in multi_writer.events if event.get("event_type") == "retrieval_observation"]
        assert [event["tool_call_id"] for event in retrieval_events] == ["call_alpha", "call_beta"]
        assert multi_state["search_count"] == 2
        assert multi_state["eligible"]
        assert multi_state["trajectory"].index("<search>alpha</search>") < multi_state["trajectory"].index("<information>")
        assert multi_state["trajectory"].index("<search>beta</search>") > multi_state["trajectory"].index("<information>")
        second_messages = multi_client.calls[1]["messages"]
        assert [message.get("tool_call_id") for message in second_messages if message.get("role") == "tool"] == [
            "call_alpha",
            "call_beta",
        ]
        assert [message.get("role") for message in second_messages[-3:]] == ["tool", "tool", "user"]

        budget_calls = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": SEARCH_TOOL_NAME, "arguments": json.dumps({"query": f"q{index}"})},
            }
            for index in range(4)
        ]
        budget_client = _MockClient(
            [
                _mock_body({"role": "assistant", "content": None, "tool_calls": budget_calls}, "tool_calls"),
                _mock_body({"role": "assistant", "content": "Paris"}, "stop"),
            ]
        )
        budget_writer = _MemoryEventWriter()
        budget_state = _run_candidate(_mock_item("hotpotqa:budget"), 0, _mock_args(), budget_client, budget_writer, [])
        budget_events = [event for event in budget_writer.events if event.get("event_type") == "retrieval_observation"]
        assert [event["tool_call_id"] for event in budget_events] == ["call_0", "call_1", "call_2", "call_3"]
        assert sum(bool(event.get("executed")) for event in budget_events) == 3
        assert budget_state["search_count"] == 3
        assert budget_state["force_answer"]
        assert "tool_call_budget_clipped" in budget_state["failure_reasons"]
        assert budget_client.calls[1]["allow_search"] is False
        assert [message.get("tool_call_id") for message in budget_client.calls[1]["messages"] if message.get("role") == "tool"] == [
            "call_0",
            "call_1",
            "call_2",
            "call_3",
        ]

        continuation_client = _MockClient(
            [
                _mock_body({"role": "assistant", "content": "partial"}, "length"),
                _mock_body({"role": "assistant", "content": "Paris"}, "stop"),
            ]
        )
        continuation_writer = _MemoryEventWriter()
        continuation_state = _run_candidate(
            _mock_item("hotpotqa:continuation"),
            0,
            _mock_args(),
            continuation_client,
            continuation_writer,
            [],
        )
        assert continuation_state["status"] == "answered"
        assert continuation_state["continuation_count"] == 1
        assert not continuation_state["truncation_exhausted"]
        assert "generation_truncated" not in continuation_state["failure_reasons"]
        assert continuation_state["_messages"][3]["role"] == "assistant"
        assert continuation_state["_messages"][4]["role"] == "user"
        assert continuation_state["_messages"][5]["role"] == "assistant"

        exhausted_client = _MockClient(
            [_mock_body({"role": "assistant", "content": "partial"}, "length") for _ in range(3)]
        )
        exhausted_writer = _MemoryEventWriter()
        exhausted_state = _run_candidate(
            _mock_item("hotpotqa:exhausted"),
            0,
            _mock_args(max_continuations=2),
            exhausted_client,
            exhausted_writer,
            [],
        )
        assert exhausted_state["status"] == "failed"
        assert exhausted_state["continuation_count"] == 2
        assert exhausted_state["truncation_exhausted"]
        assert "continuation_exhausted" in exhausted_state["failure_reasons"]
        assert "generation_truncated" in exhausted_state["failure_reasons"]
    finally:
        globals()["_retrieve_one"] = original_retrieve_one

    return {
        "status": "passed",
        "checks": [
            "exact_reference_60_ids_and_120_candidate_ids",
            "per_source_20_question_sampling",
            "tool_to_project_xml_trajectory_conversion",
            "strict_project_trajectory_validation",
            "qa_em_normalization_parity",
            "metrics_pass_at_1_and_observed_pass_at_2",
            "redaction_and_no_key_serialization",
            "multiple_tool_calls_and_tool_result_id_order",
            "tool_call_budget_clipping_and_forced_final_answer",
            "length_continuation_success_and_history_integrity",
            "continuation_exhaustion_failure",
            "no_external_api_or_retriever_call",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--reference-summary", default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--questions-per-source", type=int, default=DEFAULT_QUESTIONS_PER_SOURCE)
    parser.add_argument("--samples-per-question", type=int, default=DEFAULT_SAMPLES_PER_QUESTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--retrieval-url", default=DEFAULT_RETRIEVAL_URL)
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--retrieval-timeout", type=float, default=90.0)
    parser.add_argument("--max-searches", type=int, default=DEFAULT_MAX_SEARCHES)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-continuations", type=int, default=DEFAULT_MAX_CONTINUATIONS)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--api-timeout", type=float, default=DEFAULT_API_TIMEOUT)
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument("--retry-backoff-max", type=float, default=DEFAULT_RETRY_BACKOFF_MAX)
    parser.add_argument("--resume", action="store_true", help="Resume existing incremental output files.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly truncate this pilot's output files.")
    parser.add_argument("--self-test", action="store_true", help="Run local tests without API or retriever calls.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.model != DEFAULT_MODEL:
        raise ValueError(f"this pilot requires the default model {DEFAULT_MODEL!r}")
    if args.questions_per_source != 20 or args.samples_per_question != 2 or args.seed != 42 or args.max_searches != 3:
        raise ValueError("paired pilot requires 20 questions/source, 2 trajectories/question, seed 42, and max 3 searches")
    if args.retrieval_topk != 3:
        raise ValueError("paired pilot requires retrieval topk=3")
    if args.concurrency < 1 or args.retry_attempts < 1 or args.max_new_tokens < 1 or args.max_continuations < 0:
        raise ValueError(
            "concurrency, retry attempts, and max-new-tokens must be positive; "
            "max-continuations cannot be negative"
        )


def _require_api_key() -> str:
    try:
        api_key = os.environ["DEEPSEEK_API_KEY"]
    except KeyError as exc:
        raise RuntimeError("DEEPSEEK_API_KEY is absent; refusing to make any network/API call") from exc
    if not api_key.strip():
        raise RuntimeError("DEEPSEEK_API_KEY is present but empty; refusing to make any network/API call")
    return api_key


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    api_key = _require_api_key()
    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)
    reference_summary = Path(args.reference_summary)
    selected_ids = _reference_ids(reference_summary)
    items = _read_paired_questions(Path(args.dataset_path), selected_ids)

    candidate_path = output_dir / "candidates.jsonl"
    accepted_path = output_dir / "accepted.jsonl"
    interaction_path = output_dir / "interactions.jsonl"
    existing_candidates = _load_jsonl(candidate_path)
    existing_ids = {str(record.get("candidate_id", "")) for record in existing_candidates}
    expected_ids = {
        f"{item['question_id']}__sample_{sample_index}"
        for item in items
        for sample_index in range(args.samples_per_question)
    }
    if not existing_ids.issubset(expected_ids):
        raise ValueError("existing candidates.jsonl contains IDs outside the paired pilot")
    events = _load_jsonl(interaction_path)
    events_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        candidate_id = str(event.get("candidate_id", ""))
        if candidate_id in expected_ids:
            events_by_candidate[candidate_id].append(event)
    accepted_existing = _load_jsonl(accepted_path)
    accepted_ids = {str(record.get("candidate_id", "")) for record in accepted_existing}

    append_raw = bool(existing_candidates or events or accepted_existing) and not args.overwrite
    event_writer = JSONLAppender(interaction_path, append=append_raw, api_key=api_key)
    candidate_writer = JSONLAppender(candidate_path, append=append_raw, api_key=api_key)
    accepted_writer = JSONLAppender(accepted_path, append=append_raw, api_key=api_key)
    client = DeepSeekClient(api_key, args)
    pending: list[tuple[dict[str, Any], int]] = []
    for item in items:
        for sample_index in range(args.samples_per_question):
            candidate_id = f"{item['question_id']}__sample_{sample_index}"
            if candidate_id not in existing_ids:
                pending.append((item, sample_index))

    started = time.monotonic()
    completed_states: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    _run_candidate,
                    item,
                    sample_index,
                    args,
                    client,
                    event_writer,
                    events_by_candidate.get(f"{item['question_id']}__sample_{sample_index}", []),
                ): (item, sample_index)
                for item, sample_index in pending
            }
            for future in as_completed(futures):
                item, sample_index = futures[future]
                try:
                    state = future.result()
                except Exception as exc:
                    state = _new_state(item, sample_index, args.seed)
                    _add_failure(state, "worker_failed", _safe_error(exc, api_key))
                    state["status"] = "failed"
                    _finalize_state(state)
                record = _records_for_output([state])[0]
                candidate_writer.append(record)
                if record["eligible"] and record["candidate_id"] not in accepted_ids:
                    accepted_writer.append(record)
                    accepted_ids.add(record["candidate_id"])
                completed_states.append(state)
    finally:
        event_writer.close()
        candidate_writer.close()
        accepted_writer.close()

    all_candidates = _load_jsonl(candidate_path)
    by_id: dict[str, dict[str, Any]] = {}
    for record in all_candidates:
        candidate_id = str(record.get("candidate_id", ""))
        if candidate_id in by_id:
            raise ValueError(f"duplicate candidate in output: {candidate_id}")
        by_id[candidate_id] = record
    if set(by_id) != expected_ids:
        raise RuntimeError(f"pilot incomplete: wrote {len(by_id)} of {len(expected_ids)} candidates")
    records = [by_id[candidate_id] for candidate_id in sorted(expected_ids)]
    accepted = [record for record in records if record.get("eligible")]
    per_dataset = {
        source: _metrics([record for record in records if record.get("data_source") == source])
        for source in DATASET_SOURCES
    }
    summary = {
        "status": "completed",
        "api_key_present": True,
        "api": {
            "model": args.model,
            "base_url": args.base_url,
            "endpoint": args.base_url.rstrip("/") + "/chat/completions",
            "thinking_requested": True,
            "reasoning_effort_requested": "high",
            "strict_search_tool_requested": True,
            "concurrency": args.concurrency,
            "retry_policy": {
                "attempts": args.retry_attempts,
                "backoff_seconds": args.retry_backoff,
                "backoff_max_seconds": args.retry_backoff_max,
                "retry_statuses": [429, "5xx", "timeouts", "connection_errors"],
            },
        },
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "dataset_path": args.dataset_path,
            "reference_summary": args.reference_summary,
            "output_dir": args.output_dir,
            "questions_per_source": args.questions_per_source,
            "samples_per_question": args.samples_per_question,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "retrieval_url": args.retrieval_url,
            "retrieval_topk": args.retrieval_topk,
            "max_searches": args.max_searches,
            "max_new_tokens": args.max_new_tokens,
            "max_continuations": args.max_continuations,
            "format_filter": "project strict XML trajectory after conversion: 1-3 search/information rounds then think/answer",
            "answer_filter": "project QA-EM exact-match normalization against any gold answer",
            "eligibility": "format_correct AND answer_correct only",
        },
        "dataset": {
            "selected_unique_questions": len(selected_ids),
            "selected_by_source": dict(Counter(item["data_source"] for item in items)),
            "selected_question_ids": selected_ids,
        },
        "candidates": {
            "expected": len(expected_ids),
            "written": len(records),
            "accepted_written": len(accepted),
            "resumed_existing": len(existing_ids),
            "newly_completed": len(completed_states),
        },
        "metrics": {
            "overall": _metrics(records),
            "per_dataset": per_dataset,
        },
        "timing_seconds": {
            "new_candidate_wall_time": round(time.monotonic() - started, 3),
            "mean_new_candidate_wall_time": round((time.monotonic() - started) / len(completed_states), 3) if completed_states else 0.0,
        },
        "output_files": {
            "interactions": str(interaction_path),
            "candidates": str(candidate_path),
            "accepted": str(accepted_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary, api_key)
    print(json.dumps(_redact(summary, api_key), ensure_ascii=False, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        try:
            print(json.dumps(_self_test_reference_and_metrics(), ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"SELF-TEST FAILED: {_safe_error(exc)}", file=sys.stderr)
            return 1
        return 0
    try:
        run_pilot(args)
    except Exception as exc:
        print(f"PILOT FAILED: {_safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
