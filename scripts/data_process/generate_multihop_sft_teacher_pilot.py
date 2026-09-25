#!/usr/bin/env python3
"""Generate and score a small multi-hop teacher-trajectory pilot.

The pilot uses the local Qwen2.5-7B-Instruct checkpoint as a response teacher.
Each candidate is generated as an iterative ReAct trajectory: a teacher turn
contains either one ``<think>...<search>...</search>`` action or one final
``<think>...<answer>...</answer>`` action.  A real retriever is called after
every valid search action and its returned documents are appended as an
``<information>`` observation before the next teacher turn.

Only two predicates decide eligibility:

    eligible = format_correct and answer_correct

The script retains failed candidates, raw teacher turns, retrieval outputs or
errors, and failure reasons so that the pilot can be audited rather than only
reporting the accepted trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_MODEL_PATH = "/home/luwa/Documents/Tree-GRPO/models/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_PATH = "data/multihopqa_search_mixed_402020_20260830/train.parquet"
DEFAULT_OUTPUT_DIR = "data/teacher_sft_pilot_qwen2.5_7b_multihop"
DATASET_SOURCES = ("hotpotqa", "2wikimultihopqa", "musique")
MAX_SEARCHES = 3

SYSTEM_PROMPT = "You are a helpful assistant that answers questions using a search engine."
RECOGNIZED_TAGS = ("think", "search", "information", "answer")
TAG_PATTERN = re.compile(r"</?(?:think|search|information|answer)>")
SEARCH_BODY_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_BODY_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SEARCH_TURN_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*"
    r"<search>(?P<query>.*?)</search>\s*$",
    re.DOTALL,
)
ANSWER_TURN_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)


def _json_safe(value: Any) -> Any:
    """Convert parquet/numpy values into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _answers_list(value: Any) -> list[str]:
    value = _json_safe(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    value = str(value).strip()
    return [value] if value else []


def normalize_answer(text: str) -> str:
    """Match ``verl.utils.reward_score.qa_em.normalize_answer`` exactly."""

    text = str(text).lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(answer: str | None, golden_answers: Sequence[str]) -> bool:
    if answer is None or not str(answer).strip():
        return False
    normalized = normalize_answer(answer)
    return any(normalized == normalize_answer(gold) for gold in golden_answers)


def _question_key(question: str) -> str:
    return " ".join(str(question).split()).casefold()


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    value = _json_safe(value)
    return value.get(key, default) if isinstance(value, Mapping) else default


def extract_question(prompt_value: Any) -> str:
    """Extract the question from the mixed parquet prompt message."""

    prompt_value = _json_safe(prompt_value)
    messages = prompt_value if isinstance(prompt_value, list) else [prompt_value]
    content = ""
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = str(message.get("content", ""))
            break
    if not content and messages and isinstance(messages[0], Mapping):
        content = str(messages[0].get("content", ""))
    if "Question:" in content:
        return content.split("Question:", 1)[1].strip()
    return content.strip()


def _row_question_record(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    source = str(row.get("data_source", "")).strip()
    question = extract_question(row.get("prompt"))
    reward_model = _json_safe(row.get("reward_model", {}))
    ground_truth = _mapping_value(reward_model, "ground_truth", {})
    gold_answers = _answers_list(_mapping_value(ground_truth, "target", []))
    extra_info = _json_safe(row.get("extra_info", {}))
    source_index = _mapping_value(extra_info, "index")
    question_id = str(source_index) if source_index is not None else f"{source}:{row_index}"
    return {
        "row_index": row_index,
        "question_id": question_id,
        "data_source": source,
        "question": question,
        "gold_answers": gold_answers,
        "extra_info": extra_info,
    }


def read_stratified_questions(
    dataset_path: str,
    questions_per_source: int = 20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], int]:
    """Select a deterministic, duplicate-free sample from each source."""

    import pandas as pd

    frame = pd.read_parquet(dataset_path)
    required = {"data_source", "prompt", "reward_model"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    rows = frame.to_dict(orient="records")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        record = _row_question_record(row, row_index)
        if record["data_source"] in DATASET_SOURCES:
            by_source[record["data_source"]].append(record)

    selected: list[dict[str, Any]] = []
    for source_index, source in enumerate(DATASET_SOURCES):
        candidates = by_source[source]
        order = list(range(len(candidates)))
        random.Random(seed + source_index).shuffle(order)
        seen: set[str] = set()
        source_selected: list[dict[str, Any]] = []
        for index in order:
            record = candidates[index]
            key = _question_key(record["question"])
            if not key or key in seen:
                continue
            seen.add(key)
            source_selected.append(record)
            if len(source_selected) == questions_per_source:
                break
        if len(source_selected) != questions_per_source:
            raise ValueError(
                f"Could select only {len(source_selected)} unique {source} questions; "
                f"requested {questions_per_source}"
            )
        selected.extend(source_selected)

    return selected, len(rows)


def _build_chat_prompt(tokenizer: Any, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        return f"{SYSTEM_PROMPT}\n\n{user_content}\n\nAssistant:\n"


def parse_turn(text: str) -> dict[str, Any]:
    """Classify a teacher turn while retaining malformed output for auditing."""

    text = str(text or "").strip()
    search_match = SEARCH_TURN_PATTERN.fullmatch(text)
    if search_match and search_match.group("query").strip():
        return {
            "action": "search",
            "strict": True,
            "query": search_match.group("query").strip(),
            "answer": None,
            "reasons": [],
        }

    answer_match = ANSWER_TURN_PATTERN.fullmatch(text)
    if answer_match and answer_match.group("answer").strip():
        return {
            "action": "answer",
            "strict": True,
            "query": None,
            "answer": answer_match.group("answer").strip(),
            "reasons": [],
        }

    answer_matches = ANSWER_BODY_PATTERN.findall(text)
    search_matches = SEARCH_BODY_PATTERN.findall(text)
    nonempty_answer = answer_matches[-1].strip() if answer_matches and answer_matches[-1].strip() else None
    nonempty_query = search_matches[0].strip() if search_matches and search_matches[0].strip() else None

    if nonempty_answer is not None:
        reasons = ["answer_turn_not_exact_think_then_answer"]
        if nonempty_query is not None:
            reasons.append("answer_turn_contains_search_tag")
        return {
            "action": "answer",
            "strict": False,
            "query": nonempty_query,
            "answer": nonempty_answer,
            "reasons": reasons,
        }
    if nonempty_query is not None:
        reasons = ["search_turn_not_exact_think_then_search"]
        if ANSWER_BODY_PATTERN.search(text):
            reasons.append("search_turn_contains_answer_tag")
        return {
            "action": "search",
            "strict": False,
            "query": nonempty_query,
            "answer": None,
            "reasons": reasons,
        }

    reasons = ["no_nonempty_answer_or_search_action"]
    if any(f"<{tag}>" in text for tag in RECOGNIZED_TAGS):
        reasons.append("unclosed_or_empty_recognized_tag")
    if text:
        reasons.append("turn_not_parseable")
    return {
        "action": "failure",
        "strict": False,
        "query": None,
        "answer": None,
        "reasons": reasons,
    }


def _tag_counts(text: str) -> Counter[str]:
    return Counter(TAG_PATTERN.findall(text or ""))


def validate_trajectory(state: Mapping[str, Any], max_searches: int = MAX_SEARCHES) -> tuple[bool, list[str]]:
    """Validate 1--3 strict search rounds followed by a strict answer turn."""

    reasons: list[str] = []
    records = list(state.get("turn_records", []))
    if state.get("status") != "answered" or not state.get("ended_with_answer"):
        reasons.append("trajectory_did_not_end_with_nonempty_answer")
    if not records:
        reasons.append("no_generation_turn")
        return False, reasons

    final_record = records[-1]
    final_parse = final_record.get("parse", {})
    if final_parse.get("action") != "answer":
        reasons.append("final_turn_is_not_answer")
    if not final_parse.get("strict", False):
        reasons.extend(final_parse.get("reasons", ["final_answer_turn_malformed"]))
    if not final_parse.get("answer"):
        reasons.append("empty_final_answer")

    search_records = records[:-1] if final_parse.get("action") == "answer" else records
    search_count = sum(1 for record in search_records if record.get("parse", {}).get("action") == "search")
    if search_count < 1:
        reasons.append("requires_at_least_one_search_round")
    if search_count > max_searches:
        reasons.append("too_many_search_rounds")

    for record in search_records:
        parsed = record.get("parse", {})
        if parsed.get("action") != "search":
            reasons.append("non_search_turn_before_final_answer")
            continue
        if not parsed.get("strict", False):
            reasons.extend(parsed.get("reasons", ["search_turn_malformed"]))
        if record.get("information") is None:
            reasons.append("missing_information_after_search")
        if TAG_PATTERN.search(str(record.get("information") or "")):
            reasons.append("control_tag_inside_information")

        expected = Counter({"<think>": 1, "</think>": 1, "<search>": 1, "</search>": 1})
        actual = _tag_counts(str(record.get("raw_output", "")))
        for tag, count in expected.items():
            if actual[tag] != count:
                reasons.append(f"unexpected_tag_count_{tag}")
        for tag in ("<answer>", "</answer>", "<information>", "</information>"):
            if actual[tag]:
                reasons.append(f"unexpected_tag_{tag}")

    expected_final = Counter({"<think>": 1, "</think>": 1, "<answer>": 1, "</answer>": 1})
    actual_final = _tag_counts(str(final_record.get("raw_output", "")))
    for tag, count in expected_final.items():
        if actual_final[tag] != count:
            reasons.append(f"unexpected_final_tag_count_{tag}")
    for tag in ("<search>", "</search>", "<information>", "</information>"):
        if actual_final[tag]:
            reasons.append(f"unexpected_final_tag_{tag}")

    for record in records:
        generation = record.get("generation", {})
        if generation.get("hit_max_new_tokens"):
            reasons.append("generation_hit_max_new_tokens")

    deduplicated = list(dict.fromkeys(reasons))
    return not deduplicated, deduplicated


def assemble_trajectory(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for record in state.get("turn_records", []):
        raw = str(record.get("raw_output", "")).strip()
        if raw:
            parts.append(raw)
        if record.get("information") is not None:
            parts.append(f"<information>{str(record.get('information') or '').strip()}</information>")
    return "\n\n".join(parts)


def _format_information(retrieval_result: Any) -> str:
    """Format the real retriever response as the project's document context."""

    if retrieval_result is None:
        return ""
    if not isinstance(retrieval_result, list):
        return str(retrieval_result)
    formatted = ""
    for index, item in enumerate(retrieval_result):
        if isinstance(item, Mapping):
            document = item.get("document", item)
            if isinstance(document, Mapping):
                content = document.get("contents", document.get("content", ""))
                if not content:
                    content = json.dumps(document, ensure_ascii=False, sort_keys=True)
            else:
                content = str(document)
        else:
            content = str(item)
        lines = str(content).splitlines()
        title = lines[0] if lines else ""
        body = "\n".join(lines[1:])
        formatted += f"Doc {index + 1}(Title: {title}) {body}\n"
    return formatted


def _retrieve_batch(
    retrieval_url: str,
    queries: Sequence[str],
    topk: int,
    timeout: float,
) -> list[Any]:
    import requests

    response = requests.post(
        retrieval_url,
        json={"queries": list(queries), "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("result") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or len(results) != len(queries):
        raise RuntimeError(
            f"retriever returned {type(results).__name__} with length "
            f"{len(results) if isinstance(results, list) else 'n/a'}; expected {len(queries)}"
        )
    return results


class TeacherGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        expected = str(args.expected_cuda_visible).replace(" ", "")
        if not visible:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be set explicitly, for example CUDA_VISIBLE_DEVICES=4")
        if "," in visible or (expected and visible.replace(" ", "") != expected):
            raise RuntimeError(
                f"refusing unsafe GPU selection: CUDA_VISIBLE_DEVICES={visible!r}; expected one device {expected!r}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"CUDA is unavailable or not single-GPU: available={torch.cuda.is_available()} "
                f"device_count={torch.cuda.device_count()}"
            )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        except Exception as exc:
            raise RuntimeError(f"could not inspect visible GPU memory safely: {exc}") from exc
        if free_bytes < args.min_free_gib * (1024**3):
            raise RuntimeError(
                f"visible GPU has only {free_bytes / (1024**3):.2f} GiB free; "
                f"minimum safe free memory is {args.min_free_gib:.2f} GiB"
            )

        if Path(args.model_path).resolve() != Path(REQUIRED_MODEL_PATH).resolve():
            raise RuntimeError(f"only the required local teacher is allowed: {REQUIRED_MODEL_PATH}")
        if not Path(args.model_path).is_dir():
            raise FileNotFoundError(f"local teacher model does not exist: {args.model_path}")

        self.device = torch.device("cuda:0")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(0)
        # PyTorch 2.6 may reject an explicit logical device after
        # CUDA_VISIBLE_DEVICES remapping even though cuda:0 allocations work.
        torch.cuda.reset_peak_memory_stats()
        self.prompt_max_length = args.prompt_max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True

    def generate(
        self,
        user_prompts: Sequence[str],
        max_new_tokens: int,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        prompts = [_build_chat_prompt(self.tokenizer, prompt) for prompt in user_prompts]
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            untruncated_lengths = [
                len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                for prompt in prompt_batch
            ]
            encoded = self.tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.prompt_max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_width = encoded["input_ids"].shape[1]
            for index, sequence in enumerate(generated):
                new_ids = sequence[prompt_width:].tolist()
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and eos_id in new_ids:
                    generated_token_count = new_ids.index(eos_id) + 1
                    hit_max_new_tokens = False
                else:
                    generated_token_count = len(new_ids)
                    hit_max_new_tokens = generated_token_count >= max_new_tokens
                text = self.tokenizer.decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                outputs.append(
                    {
                        "text": text,
                        "generated_token_count": generated_token_count,
                        "hit_max_new_tokens": hit_max_new_tokens,
                        "input_token_count": untruncated_lengths[index],
                        "input_was_truncated": untruncated_lengths[index] > self.prompt_max_length,
                    }
                )
            del encoded, generated
            self.torch.cuda.empty_cache()
        return outputs

    def memory_stats(self) -> dict[str, Any]:
        allocated = self.torch.cuda.max_memory_allocated()
        reserved = self.torch.cuda.max_memory_reserved()
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(self.device),
            "peak_memory_allocated_bytes": int(allocated),
            "peak_memory_reserved_bytes": int(reserved),
            "peak_memory_allocated_gib": round(allocated / (1024**3), 3),
            "peak_memory_reserved_gib": round(reserved / (1024**3), 3),
        }


def _turn_prompt(state: Mapping[str, Any], force_answer: bool) -> str:
    history = assemble_trajectory(state)
    question = state["question"]
    if force_answer:
        instruction = (
            "No more searches are allowed because the maximum of three searches has been used. "
            "Use all observations and output exactly one final answer turn in this form, with no "
            "extra text: <think>brief reasoning</think><answer>short final answer</answer>."
        )
    elif not history:
        instruction = (
            "You must search at least once. On this turn output exactly one search action in this "
            "form, with no extra text: <think>brief reasoning</think><search>one focused query</search>."
        )
    else:
        instruction = (
            "Read the observations below. If they are sufficient, output exactly one final answer "
            "turn: <think>brief reasoning</think><answer>short final answer</answer>. Otherwise "
            "output exactly one new search action: <think>brief reasoning</think><search>one focused "
            "query</search>. Do not output both actions and do not add extra text."
        )
    return (
        f"Question: {question}\n\n"
        f"{instruction}\n\n"
        f"Searches already used: {state.get('search_count', 0)}\n\n"
        f"Conversation and actual observations so far:\n"
        f"{history if history else '(none)'}"
    )


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
    }


def _add_failure(state: dict[str, Any], *reasons: str) -> None:
    for reason in reasons:
        if reason and reason not in state["state_failure_reasons"]:
            state["state_failure_reasons"].append(reason)


def _finalize_state(state: dict[str, Any]) -> None:
    format_correct, format_reasons = validate_trajectory(state)
    answer = state.get("final_answer")
    answer_correct = answer_em(answer, state.get("gold_answers", []))
    failure_reasons = list(state.get("state_failure_reasons", []))
    failure_reasons.extend(format_reasons)
    if answer is None:
        failure_reasons.append("final_answer_extraction_failed")
    elif not answer_correct:
        failure_reasons.append("final_answer_em_failed")
    if any(record.get("generation", {}).get("input_was_truncated") for record in state["turn_records"]):
        failure_reasons.append("input_prompt_truncated")
    state["format_correct"] = bool(format_correct)
    state["answer_correct"] = bool(answer_correct)
    state["eligible"] = bool(format_correct and answer_correct)
    state["format_failure_reasons"] = list(dict.fromkeys(format_reasons))
    state["failure_reasons"] = list(dict.fromkeys(failure_reasons))
    state["trajectory"] = assemble_trajectory(state)


def run_rollouts(
    args: argparse.Namespace,
    generator: TeacherGenerator,
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], float, float]:
    states = [
        _new_state(item, sample_index, args.seed)
        for item in items
        for sample_index in range(args.samples_per_question)
    ]
    active = set(range(len(states)))
    generation_seconds = 0.0
    retrieval_seconds = 0.0
    while active:
        active_indices = sorted(active)
        prompts = [
            _turn_prompt(states[index], states[index]["search_count"] >= args.max_searches)
            for index in active_indices
        ]
        force_answer = [states[index]["search_count"] >= args.max_searches for index in active_indices]
        generation_started = time.monotonic()
        generated = generator.generate(prompts, args.turn_max_new_tokens, args.generation_batch_size)
        generation_seconds += time.monotonic() - generation_started
        if len(generated) != len(active_indices):
            raise RuntimeError("teacher returned a different number of turns than active candidates")

        pending_retrievals: list[tuple[int, str, bool]] = []
        for local_index, state_index in enumerate(active_indices):
            state = states[state_index]
            generation = generated[local_index]
            raw_output = str(generation["text"]).strip()
            parsed = parse_turn(raw_output)
            record = {
                "turn_index": len(state["turn_records"]),
                "input_prompt": prompts[local_index],
                "raw_output": raw_output,
                "generation": {
                    "generated_token_count": generation["generated_token_count"],
                    "hit_max_new_tokens": generation["hit_max_new_tokens"],
                    "input_token_count": generation["input_token_count"],
                    "input_was_truncated": generation["input_was_truncated"],
                    "max_new_tokens": args.turn_max_new_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                "forced_final_answer": force_answer[local_index],
                "parse": parsed,
                "information": None,
            }
            state["turn_records"].append(record)
            _add_failure(state, *parsed.get("reasons", []))
            if generation["hit_max_new_tokens"]:
                _add_failure(state, "generation_hit_max_new_tokens")

            if parsed["action"] == "answer":
                state["status"] = "answered"
                state["ended_with_answer"] = bool(parsed.get("answer"))
                state["final_answer"] = parsed.get("answer")
                active.remove(state_index)
                continue

            if parsed["action"] == "search" and parsed.get("query"):
                if state["search_count"] >= args.max_searches:
                    _add_failure(state, "search_attempted_after_max_searches")
                    state["status"] = "failed"
                    active.remove(state_index)
                    continue
                pending_retrievals.append(
                    (state_index, str(parsed["query"]), bool(parsed.get("strict")))
                )
                continue

            _add_failure(state, "no_nonempty_answer_or_valid_search")
            if generation["hit_max_new_tokens"]:
                _add_failure(state, "generation_truncated")
            state["status"] = "failed"
            active.remove(state_index)

        if not pending_retrievals:
            continue

        retrieval_started = time.monotonic()
        try:
            retrieval_results = _retrieve_batch(
                args.retrieval_url,
                [query for _, query, _ in pending_retrievals],
                args.retrieval_topk,
                args.retrieval_timeout,
            )
            retrieval_error = None
        except Exception as exc:
            retrieval_results = [None] * len(pending_retrievals)
            retrieval_error = f"{type(exc).__name__}: {exc}"
        retrieval_seconds += time.monotonic() - retrieval_started

        for (state_index, query, strict_search), result in zip(pending_retrievals, retrieval_results):
            state = states[state_index]
            record = state["turn_records"][-1]
            state["search_count"] += 1
            information = _format_information(result) if retrieval_error is None else ""
            record["information"] = information
            retrieval_record = {
                "search_turn_index": record["turn_index"],
                "query": query,
                "topk": args.retrieval_topk,
                "result": result,
                "information": information,
                "error": retrieval_error,
            }
            state["retrieval_outputs"].append(retrieval_record)
            if retrieval_error is not None:
                _add_failure(state, "retrieval_failed", retrieval_error)
                state["status"] = "failed"
                active.remove(state_index)
            elif not strict_search:
                _add_failure(state, "malformed_search_turn_after_retrieval")
                state["status"] = "failed"
                active.remove(state_index)

    for state in states:
        _finalize_state(state)
    return states, generation_seconds, retrieval_seconds


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def _metrics_for_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(records)
    format_count = sum(bool(record.get("format_correct")) for record in records)
    answer_count = sum(bool(record.get("answer_correct")) for record in records)
    eligible_count = sum(bool(record.get("eligible")) for record in records)
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record["question_id"])].append(record)
    question_answer_success = sum(
        any(item.get("answer_correct") for item in group) for group in by_question.values()
    )
    question_eligible_success = sum(
        any(item.get("eligible") for item in group) for group in by_question.values()
    )
    failures: Counter[str] = Counter()
    search_counts: Counter[str] = Counter()
    for record in records:
        failures.update(record.get("failure_reasons", []))
        search_counts[str(record.get("search_count", 0))] += 1
    return {
        "trajectory_count": denominator,
        "format_correct": _metric(format_count, denominator),
        "answer_em": _metric(answer_count, denominator),
        "eligible": _metric(eligible_count, denominator),
        "format_rate": format_count / denominator if denominator else 0.0,
        "answer_em_rate": answer_count / denominator if denominator else 0.0,
        "eligible_rate": eligible_count / denominator if denominator else 0.0,
        "question_level_answer_pass_at_2_observed_any_success": _metric(
            question_answer_success, len(by_question)
        ),
        "question_level_eligible_pass_at_2_observed_any_success": _metric(
            question_eligible_success, len(by_question)
        ),
        "search_count_distribution": dict(sorted(search_counts.items(), key=lambda item: int(item[0]))),
        "failure_reason_counts": dict(failures.most_common()),
    }


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
                "trajectory": state["trajectory"],
                "extracted_answer": state["final_answer"],
                "format_correct": state["format_correct"],
                "answer_correct": state["answer_correct"],
                "eligible": state["eligible"],
                "format_failure_reasons": state["format_failure_reasons"],
                "failure_reasons": state["failure_reasons"],
            }
        )
    return output


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def run_self_test() -> dict[str, Any]:
    def state_from_turns(turns: Sequence[tuple[str, str | None]], status: str = "answered") -> dict[str, Any]:
        records = []
        for index, (raw, information) in enumerate(turns):
            parsed = parse_turn(raw)
            records.append(
                {
                    "turn_index": index,
                    "raw_output": raw,
                    "parse": parsed,
                    "information": information,
                    "generation": {"hit_max_new_tokens": False},
                }
            )
        return {
            "status": status,
            "ended_with_answer": status == "answered",
            "turn_records": records,
        }

    one_search = state_from_turns(
        [
            ("<think>Find the entity.</think><search>first query</search>", "Doc 1 Paris\n"),
            ("<think>The document gives it.</think><answer>Paris</answer>", None),
        ]
    )
    valid, reasons = validate_trajectory(one_search)
    assert valid and not reasons
    assert assemble_trajectory(one_search).count("<information>") == 1

    three_searches = state_from_turns(
        [
            ("<think>one</think><search>query one</search>", "one"),
            ("<think>two</think><search>query two</search>", "two"),
            ("<think>three</think><search>query three</search>", "three"),
            ("<think>done</think><answer>Paris</answer>", None),
        ]
    )
    valid_three, reasons_three = validate_trajectory(three_searches)
    assert valid_three and not reasons_three

    direct_answer = state_from_turns(
        [("<think>Recall it.</think><answer>Paris</answer>", None)]
    )
    direct_valid, direct_reasons = validate_trajectory(direct_answer)
    assert not direct_valid and "requires_at_least_one_search_round" in direct_reasons

    malformed = state_from_turns(
        [
            ("<think>one</think><search>query</search><search>extra</search>", "docs"),
            ("<think>done</think><answer>Paris</answer>", None),
        ]
    )
    malformed_valid, malformed_reasons = validate_trajectory(malformed)
    assert not malformed_valid and malformed_reasons

    truncated = parse_turn("<think>unfinished")
    assert truncated["action"] == "failure"
    assert parse_turn("<think>x</think><answer></answer>")["action"] == "failure"
    assert answer_em("The Paris", ["Paris"])
    assert not answer_em("London", ["Paris"])

    from verl.utils.reward_score.qa_em import normalize_answer as verl_normalize_answer

    parity_cases = [
        "The U.S. Constitution",
        "An apple, a pear!",
        "  Mixed CASE -- text  ",
        "C++ and Python",
        "the answer",
    ]
    for case in parity_cases:
        assert normalize_answer(case) == verl_normalize_answer(case), case

    return {
        "status": "passed",
        "checks": [
            "one_search_round_valid",
            "three_search_rounds_valid",
            "zero_search_round_rejected",
            "extra_recognized_tag_rejected",
            "truncated_or_empty_answer_rejected",
            "answer_em_filter",
            "verl_qa_em_normalization_parity",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=REQUIRED_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--questions-per-source", type=int, default=20)
    parser.add_argument("--samples-per-question", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--retrieval-timeout", type=float, default=90.0)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--turn-max-new-tokens", type=int, default=192)
    parser.add_argument("--prompt-max-length", type=int, default=8192)
    parser.add_argument("--max-searches", type=int, default=MAX_SEARCHES)
    parser.add_argument("--expected-cuda-visible", default="4")
    parser.add_argument("--min-free-gib", type=float, default=18.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run CPU-only validator tests.")
    parser.add_argument("--self-test", action="store_true", help="Alias for --dry-run validator tests.")
    return parser


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.questions_per_source != 20 or args.samples_per_question != 2 or args.max_searches != 3:
        raise ValueError("the requested pilot requires 20 questions/source, 2 trajectories/question, and max 3 searches")
    output_dir = Path(args.output_dir)
    output_paths = [output_dir / name for name in ("candidates.jsonl", "accepted.jsonl", "summary.json")]
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing pilot outputs; pass --overwrite: " + ", ".join(existing)
            )

    selected, dataset_rows = read_stratified_questions(
        args.dataset_path,
        questions_per_source=args.questions_per_source,
        seed=args.seed,
    )
    candidate_count = len(selected) * args.samples_per_question
    generator = TeacherGenerator(args)
    rollout_result = run_rollouts(args, generator, selected)
    states, generation_seconds, retrieval_seconds = rollout_result
    records = _records_for_output(states)
    if len(records) != candidate_count:
        raise RuntimeError(f"expected {candidate_count} candidates, produced {len(records)}")
    accepted = [record for record in records if record["eligible"]]

    per_dataset: dict[str, Any] = {}
    for source in DATASET_SOURCES:
        source_records = [record for record in records if record["data_source"] == source]
        source_questions = len({record["question_id"] for record in source_records})
        metrics = _metrics_for_records(source_records)
        metrics["selected_question_count"] = source_questions
        per_dataset[source] = metrics

    summary = {
        "status": "completed",
        "config": {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "output_dir": args.output_dir,
            "questions_per_source": args.questions_per_source,
            "samples_per_question": args.samples_per_question,
            "seed": args.seed,
            "temperature": 0.7,
            "top_p": 0.9,
            "retrieval_url": args.retrieval_url,
            "retrieval_topk": args.retrieval_topk,
            "max_searches": args.max_searches,
            "turn_max_new_tokens": args.turn_max_new_tokens,
            "prompt_max_length": args.prompt_max_length,
            "format_filter": "1-3 strict think/search/information rounds followed by strict think/answer, no extra recognized tags",
            "answer_filter": "verl qa_em exact-match normalization against any gold answer",
            "eligibility": "format_correct AND answer_correct only",
        },
        "dataset": {
            "rows": dataset_rows,
            "selected_unique_questions": len(selected),
            "selected_by_source": dict(Counter(item["data_source"] for item in selected)),
            "selected_question_ids": [item["question_id"] for item in selected],
        },
        "candidates": {
            "expected": candidate_count,
            "written": len(records),
            "accepted_written": len(accepted),
        },
        "metrics": {
            "overall": _metrics_for_records(records),
            "per_dataset": per_dataset,
        },
        "timing_seconds": {
            "generation": round(generation_seconds, 3),
            "retrieval": round(retrieval_seconds, 3),
            "total": round(time.monotonic() - started, 3),
            "mean_per_candidate": round((time.monotonic() - started) / len(records), 3),
        },
        "gpu": generator.memory_stats(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_paths[0], records)
    _write_jsonl(output_paths[1], accepted)
    with output_paths[2].open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run or args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return 0
    try:
        run_pilot(args)
    except Exception as exc:
        print(f"PILOT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
