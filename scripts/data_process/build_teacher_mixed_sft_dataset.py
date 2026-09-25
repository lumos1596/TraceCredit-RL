#!/usr/bin/env python3
"""Validate and build the 1,000+500 mixed teacher SFT dataset.

The builder consumes only accepted production JSONL records.  It validates the
stored XML trajectory again, converts it to the alternating chat format used
by ``verl.utils.dataset.SFTDataset``, performs a deterministic stratified
90/10 split, and fails on any duplicate, leakage, quota, or 2560-token error.
The information observations remain user messages, so the existing dataset
mask supervises only assistant turns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SINGLE = "/home/luwa/Documents/Tree-GRPO/data/teacher_singlehop_qwen7b_prod_20260911/accepted.jsonl"
DEFAULT_MULTI = "/home/luwa/Documents/Tree-GRPO/data/teacher_multihop_dsflash_prod_20260911/accepted.jsonl"
DEFAULT_OUTPUT = "/home/luwa/Documents/Tree-GRPO/data/teacher_sft_mixed_1500_20260911"
DEFAULT_TOKENIZER = "/home/luwa/Documents/Tree-GRPO/models/Qwen2.5-3B-Instruct"
MAX_LENGTH = 2560
SOURCES = ("nq", "hotpotqa", "2wikimultihopqa", "musique")
QUOTAS = {"nq": 1000, "hotpotqa": 200, "2wikimultihopqa": 175, "musique": 125}
COMPRESSION_MARKER = "[... information compressed ...]"

TAG_RE = re.compile(r"</?(?:think|search|information|answer)>", re.IGNORECASE)
SEARCH_BLOCK_RE = re.compile(
    r"\s*<think>(.*?)</think>\s*<search>(.*?)</search>\s*"
    r"<information>(.*?)</information>",
    re.IGNORECASE | re.DOTALL,
)
FINAL_BLOCK_RE = re.compile(
    r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
INFORMATION_MESSAGE_RE = re.compile(
    r"^<information>(.*?)</information>$", re.IGNORECASE | re.DOTALL
)
ASSISTANT_SEARCH_MESSAGE_RE = re.compile(
    r"^<think>(.*?)</think><search>(.*?)</search>$",
    re.IGNORECASE | re.DOTALL,
)
ASSISTANT_FINAL_MESSAGE_RE = re.compile(
    r"^<think>(.*?)</think><answer>(.*?)</answer>$",
    re.IGNORECASE | re.DOTALL,
)

INITIAL_INSTRUCTION = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> and it "
    "will return the top searched results between <information> and </information>. "
    "You can search as many times as you want. If no further external knowledge is "
    "needed, provide the answer inside <answer> and </answer>, without detailed "
    "illustrations. Question: {question}"
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def question_key(question: str) -> str:
    return " ".join(str(question).split()).casefold()


def normalize_answer(text: str) -> str:
    import string

    text = str(text).lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(answer: str, gold_answers: Sequence[str]) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and any(normalized == normalize_answer(gold) for gold in gold_answers)


def parse_trajectory_parts(
    trajectory: str, *, expected_searches: tuple[int, int]
) -> tuple[list[dict[str, str]], str, str]:
    """Parse strict project XML while preserving every captured body exactly."""

    text = str(trajectory or "")
    cursor = 0
    rounds: list[dict[str, str]] = []
    while True:
        match = SEARCH_BLOCK_RE.match(text, cursor)
        if match is None:
            break
        think, query, information = match.groups()
        if not query.strip():
            raise ValueError("empty search query")
        if TAG_RE.search(think) or TAG_RE.search(query) or TAG_RE.search(information):
            raise ValueError("recognized control tag nested in a trajectory field")
        rounds.append({"think": think, "query": query, "information": information})
        cursor = match.end()
    final = FINAL_BLOCK_RE.match(text, cursor)
    if final is None:
        raise ValueError("trajectory does not end with exactly think/answer")
    final_think, answer = final.groups()
    if not answer.strip():
        raise ValueError("empty final answer")
    if TAG_RE.search(final_think) or TAG_RE.search(answer):
        raise ValueError("recognized control tag nested in final answer fields")
    low, high = expected_searches
    if not low <= len(rounds) <= high:
        raise ValueError(f"expected {low}-{high} searches, found {len(rounds)}")
    return rounds, final_think, answer


def parse_trajectory(
    trajectory: str, *, expected_searches: tuple[int, int]
) -> tuple[list[dict[str, str]], str]:
    """Compatibility wrapper returning search rounds plus final answer."""

    rounds, _final_think, answer = parse_trajectory_parts(
        trajectory, expected_searches=expected_searches
    )
    return rounds, answer


def trajectory_to_messages(
    question: str,
    trajectory: str,
    *,
    expected_searches: tuple[int, int] = (1, 3),
) -> tuple[list[dict[str, str]], str, int]:
    rounds, final_think, answer = parse_trajectory_parts(
        trajectory, expected_searches=expected_searches
    )
    messages: list[dict[str, str]] = [
        {"role": "user", "content": INITIAL_INSTRUCTION.format(question=question)},
    ]
    for item in rounds:
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{item['think']}</think><search>{item['query']}</search>",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"<information>{item['information']}</information>",
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": f"<think>{final_think}</think><answer>{answer}</answer>",
        }
    )
    return messages, answer, len(rounds)


def _information_body(content: str) -> str | None:
    match = INFORMATION_MESSAGE_RE.fullmatch(str(content))
    return None if match is None else match.group(1)


def _information_indices(messages: Sequence[Mapping[str, str]]) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if str(message.get("role", "")).lower() == "user"
        and _information_body(str(message.get("content", ""))) is not None
    ]


def _replace_information_bodies(
    messages: Sequence[Mapping[str, str]], bodies: Mapping[int, str]
) -> list[dict[str, str]]:
    replaced: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        copy = {"role": str(message["role"]), "content": str(message["content"])}
        if index in bodies:
            if _information_body(copy["content"]) is None:
                raise ValueError(f"message {index} is not a complete information block")
            copy["content"] = f"<information>{bodies[index]}</information>"
        replaced.append(copy)
    return replaced


def messages_to_trajectory(messages: Sequence[Mapping[str, str]]) -> str:
    """Serialize the final chat messages into the project XML trajectory form."""

    if not messages or str(messages[0].get("role", "")).lower() != "user":
        raise ValueError("chat messages must start with the initial user message")
    rounds: list[tuple[str, str, str]] = []
    index = 1
    while index < len(messages) - 1:
        assistant = messages[index]
        information = messages[index + 1]
        if str(assistant.get("role", "")).lower() != "assistant":
            raise ValueError(f"message {index} is not an assistant search turn")
        if str(information.get("role", "")).lower() != "user":
            raise ValueError(f"message {index + 1} is not a user information turn")
        search_match = ASSISTANT_SEARCH_MESSAGE_RE.fullmatch(str(assistant.get("content", "")))
        information_body = _information_body(str(information.get("content", "")))
        if search_match is None or information_body is None:
            raise ValueError("malformed alternating search/information messages")
        rounds.append((search_match.group(1), search_match.group(2), information_body))
        index += 2
    if index != len(messages) - 1:
        raise ValueError("chat messages have an incomplete search/information pair")
    final = messages[-1]
    if str(final.get("role", "")).lower() != "assistant":
        raise ValueError("chat messages must end with an assistant answer turn")
    final_match = ASSISTANT_FINAL_MESSAGE_RE.fullmatch(str(final.get("content", "")))
    if final_match is None:
        raise ValueError("malformed final assistant message")
    return "".join(
        f"<think>{think}</think><search>{query}</search><information>{information}</information>"
        for think, query, information in rounds
    ) + f"<think>{final_match.group(1)}</think><answer>{final_match.group(2)}</answer>"


def _coerce_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _token_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=False
    )
    return _coerce_token_ids(encoded)


def _text_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize an information body for auditable statistics.

    The production tokenizer is callable, which gives the exact body token
    count.  The fallback keeps the offline FakeTokenizer useful without
    introducing a second tokenizer implementation.
    """

    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except (AttributeError, TypeError):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=False,
        )
    return _coerce_token_ids(encoded)


def _chat_token_length(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> int:
    return len(_token_ids(tokenizer, messages))


def _compress_information_body(text: str, retained_characters: int) -> str:
    """Retain a deterministic prefix and suffix with a neutral marker."""

    text = str(text)
    retained_characters = max(0, min(int(retained_characters), len(text)))
    if retained_characters >= len(text):
        return text
    if not text:
        return text
    prefix_count = (retained_characters + 1) // 2
    suffix_count = retained_characters // 2
    prefix = text[:prefix_count]
    suffix = text[len(text) - suffix_count :] if suffix_count else ""
    pieces = [part for part in (prefix, COMPRESSION_MARKER, suffix) if part]
    return " ".join(pieces)


def _assert_only_information_changed(
    original: Sequence[Mapping[str, str]], final: Sequence[Mapping[str, str]]
) -> None:
    if len(original) != len(final):
        raise ValueError("compression changed the number of chat messages")
    for index, (before, after) in enumerate(zip(original, final)):
        if str(before.get("role", "")) != str(after.get("role", "")):
            raise ValueError(f"compression changed message role at index {index}")
        before_body = _information_body(str(before.get("content", "")))
        after_body = _information_body(str(after.get("content", "")))
        if before_body is None:
            if str(before.get("content", "")) != str(after.get("content", "")):
                raise ValueError(
                    f"compression changed non-information message content at index {index}"
                )
        elif after_body is None:
            raise ValueError(f"compression corrupted information wrapper at index {index}")
        if after_body is not None and TAG_RE.search(after_body):
            raise ValueError(f"compression emitted a nested control tag at index {index}")


def _compression_stats(
    original_bodies: Sequence[str],
    final_bodies: Sequence[str],
    tokenizer: Any,
    *,
    original_length: int,
    final_length: int,
) -> dict[str, Any]:
    original_characters = sum(len(body) for body in original_bodies)
    final_characters = sum(len(body) for body in final_bodies)
    original_tokens = sum(len(_text_token_ids(tokenizer, body)) for body in original_bodies)
    final_tokens = sum(len(_text_token_ids(tokenizer, body)) for body in final_bodies)
    return {
        "original_token_length": int(original_length),
        "final_token_length": int(final_length),
        "compressed": original_bodies != final_bodies,
        "information_blocks": len(original_bodies),
        "information_blocks_compressed": sum(
            before != after for before, after in zip(original_bodies, final_bodies)
        ),
        "information_characters_original": original_characters,
        "information_characters_final": final_characters,
        "information_characters_removed": max(0, original_characters - final_characters),
        "information_tokens_original": original_tokens,
        "information_tokens_final": final_tokens,
        "information_tokens_removed": max(0, original_tokens - final_tokens),
        "compression_marker": COMPRESSION_MARKER,
    }


def compress_messages_to_budget(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    max_length: int = MAX_LENGTH,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Compress only information bodies until the exact chat template fits.

    A common retention ratio is searched over all information turns, so long
    observations share the reduction.  Each candidate is measured through the
    configured tokenizer's chat template; the final safety loop re-measures
    after every reduction and therefore never relies on character counts as a
    proxy for the actual training length.
    """

    original = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    original_length = _chat_token_length(tokenizer, original)
    indices = _information_indices(original)
    original_bodies = [str(_information_body(original[index]["content"])) for index in indices]

    if original_length <= max_length:
        metadata = _compression_stats(
            original_bodies,
            original_bodies,
            tokenizer,
            original_length=original_length,
            final_length=original_length,
        )
        return original, metadata

    if not indices:
        raise ValueError(
            "fixed non-information content alone exceeds max_length="
            f"{max_length}; no user-role <information> body can be compressed"
        )

    empty = _replace_information_bodies(original, {index: "" for index in indices})
    empty_length = _chat_token_length(tokenizer, empty)
    if empty_length > max_length:
        raise ValueError(
            "fixed non-information content alone exceeds max_length="
            f"{max_length} even with empty information bodies "
            f"(chat length={empty_length})"
        )

    def candidate(ratio: float) -> tuple[list[dict[str, str]], int, list[str]]:
        bodies = {
            index: _compress_information_body(body, int(len(body) * ratio))
            for index, body in zip(indices, original_bodies)
        }
        result = _replace_information_bodies(original, bodies)
        return result, _chat_token_length(tokenizer, result), [bodies[index] for index in indices]

    # A zero-retention candidate still carries an explicit marker.  It should
    # normally fit whenever the fixed content fits; if it cannot, silently
    # dropping the marker would violate the auditability contract.
    best_messages, best_length, best_bodies = candidate(0.0)
    if best_length > max_length:
        raise ValueError(
            "information compression marker cannot fit within max_length="
            f"{max_length} after fixed content was fitted "
            f"(chat length={best_length})"
        )
    best_ratio = 0.0
    lower, upper = 0.0, 1.0
    for _ in range(40):
        ratio = (lower + upper) / 2.0
        trial_messages, trial_length, trial_bodies = candidate(ratio)
        if trial_length <= max_length:
            lower = ratio
            best_ratio = ratio
            best_messages, best_length, best_bodies = (
                trial_messages,
                trial_length,
                trial_bodies,
            )
        else:
            upper = ratio

    # Guard against tokenizer edge cases that make the ratio search locally
    # non-monotonic.  Repeatedly lower retention and remeasure the full chat.
    safety_ratio = best_ratio
    safety_iterations = 0
    while best_length > max_length and safety_iterations < 64:
        safety_ratio *= 0.9
        best_messages, best_length, best_bodies = candidate(safety_ratio)
        safety_iterations += 1
    if best_length > max_length:
        raise ValueError(
            "iterative information compression could not satisfy "
            f"max_length={max_length}; final chat length={best_length}"
        )

    _assert_only_information_changed(original, best_messages)
    metadata = _compression_stats(
        original_bodies,
        best_bodies,
        tokenizer,
        original_length=original_length,
        final_length=best_length,
    )
    return best_messages, metadata


def _gold_answers(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if value is not None and str(value).strip() else []


def _record_to_row(record: Mapping[str, Any], source: str, source_path: Path) -> dict[str, Any]:
    question = str(record.get("question", ""))
    question_id = str(record.get("question_id", "")).strip()
    trajectory = str(record.get("trajectory", ""))
    gold_answers = _gold_answers(record.get("gold_answers", []))
    if not question.strip() or not question_id or not trajectory or not gold_answers:
        raise ValueError(f"accepted record lacks required fields: {record.get('candidate_id')}")
    if not bool(record.get("format_correct")) or not bool(record.get("answer_correct")) or not bool(record.get("eligible")):
        raise ValueError(f"accepted record is not marked eligible: {record.get('candidate_id')}")
    expected = (1, 1) if source == "nq" else (1, 3)
    messages, answer, search_count = trajectory_to_messages(question, trajectory, expected_searches=expected)
    if not answer_em(answer, gold_answers):
        raise ValueError(f"accepted record fails independent QA-EM validation: {record.get('candidate_id')}")
    trajectory_hash = sha256_text(trajectory)
    messages_json = canonical(messages)
    teacher_model = str(record.get("teacher_model") or (
        "Qwen2.5-7B-Instruct" if source == "nq" else "deepseek-v4-flash"
    ))
    teacher_type = str(record.get("teacher_type") or ("local_hf" if source == "nq" else "deepseek_api"))
    original_hash = sha256_text(trajectory)
    return {
        "messages": messages_json,
        "question": question,
        "question_id": question_id,
        "question_key": question_key(question),
        "data_source": source,
        "teacher_model": teacher_model,
        "teacher_type": teacher_type,
        "source_record": str(source_path),
        "source_candidate_id": str(record.get("candidate_id", "")),
        "sample_index": int(record.get("sample_index", 0)),
        "gold_answers": gold_answers,
        "trajectory": trajectory,
        "original_trajectory": trajectory,
        "trajectory_sha256": trajectory_hash,
        "original_trajectory_sha256": original_hash,
        "messages_sha256": sha256_text(messages_json),
        "search_count": search_count,
    }


def load_and_validate_inputs(single_path: Path, multi_path: Path) -> list[dict[str, Any]]:
    if not single_path.is_file() or not multi_path.is_file():
        raise FileNotFoundError(f"missing accepted input: {single_path if not single_path.is_file() else multi_path}")
    single_records = jsonl(single_path)
    multi_records = jsonl(multi_path)
    if len(single_records) != QUOTAS["nq"]:
        raise ValueError(f"single-hop accepted quota is {len(single_records)}, expected {QUOTAS['nq']}")
    multi_counts = Counter(str(record.get("data_source")) for record in multi_records)
    expected_multi = Counter({key: value for key, value in QUOTAS.items() if key != "nq"})
    if multi_counts != expected_multi:
        raise ValueError(f"multi-hop accepted quotas are {dict(multi_counts)}, expected {dict(expected_multi)}")

    rows = [_record_to_row(record, "nq", single_path) for record in single_records]
    rows.extend(
        _record_to_row(record, str(record["data_source"]), multi_path)
        for record in multi_records
    )
    if len(rows) != 1500:
        raise ValueError(f"combined accepted rows are {len(rows)}, expected 1500")

    validate_unique_rows(rows)
    return rows


def validate_unique_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject question or complete-trajectory duplicates before splitting."""

    seen_questions: dict[str, str] = {}
    seen_trajectories: dict[str, str] = {}
    for row in rows:
        qkey = str(row["question_key"])
        if qkey in seen_questions:
            raise ValueError(f"duplicate question across SFT sources: {qkey}")
        seen_questions[qkey] = str(row["question_id"])
        thash = str(row["trajectory_sha256"])
        if thash in seen_trajectories:
            raise ValueError(
                f"duplicate full trajectory: {row['source_candidate_id']} and {seen_trajectories[thash]}"
            )
        seen_trajectories[thash] = str(row["source_candidate_id"])


def stratified_split(
    rows: Sequence[dict[str, Any]],
    seed: int,
    val_ratio: float = 0.1,
    *,
    expected_train_size: int | None = 1350,
    expected_validation_size: int | None = 150,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if abs(val_ratio - 0.1) > 1e-9:
        raise ValueError("this experiment requires an exact 90/10 split")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["data_source"]].append(dict(row))
    if expected_train_size is not None and expected_validation_size is not None:
        if expected_train_size + expected_validation_size != len(rows):
            raise ValueError(
                "expected split sizes do not add up to the input size: "
                f"{expected_train_size}+{expected_validation_size}!={len(rows)}"
            )
        total_validation = expected_validation_size
    else:
        total_validation = int(round(len(rows) * val_ratio))
    floors = {source: int(len(group) * val_ratio) for source, group in groups.items()}
    remainder = total_validation - sum(floors.values())
    ranked = sorted(
        groups,
        key=lambda source: (-(len(groups[source]) * val_ratio - floors[source]), source),
    )
    for source in ranked[:remainder]:
        floors[source] += 1
    validation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for index, source in enumerate(sorted(groups)):
        group = list(groups[source])
        random.Random(seed + index).shuffle(group)
        count = floors[source]
        validation.extend(group[:count])
        train.extend(group[count:])
    random.Random(seed + 1000).shuffle(train)
    random.Random(seed + 1001).shuffle(validation)
    if expected_train_size is not None and len(train) != expected_train_size:
        raise ValueError(f"split sizes are train={len(train)}, expected train={expected_train_size}")
    if expected_validation_size is not None and len(validation) != expected_validation_size:
        raise ValueError(f"split sizes are train={len(train)}, validation={len(validation)}")
    if {row["question_key"] for row in train} & {row["question_key"] for row in validation}:
        raise ValueError("question leakage between train and validation")
    return train, validation


def _apply_row_compression(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    messages = json.loads(row["messages"])
    original_messages = [dict(message) for message in messages]
    final_messages, metadata = compress_messages_to_budget(
        original_messages, tokenizer, max_length=MAX_LENGTH
    )
    _assert_only_information_changed(original_messages, final_messages)
    row["messages"] = canonical(final_messages)
    row.update(metadata)
    row["token_length"] = metadata["final_token_length"]
    row["supervised_turn_count"] = sum(
        message.get("role") == "assistant" and bool(message.get("content"))
        for message in final_messages
    )
    if "question" in row and "trajectory" in row:
        original_trajectory = str(row.get("original_trajectory", row["trajectory"]))
        # This is the final serialized trajectory.  The raw input remains in
        # original_trajectory and is independently hashed below.
        final_trajectory = messages_to_trajectory(final_messages)
        row["trajectory"] = final_trajectory
        row["trajectory_sha256"] = sha256_text(final_trajectory)
        row["original_trajectory"] = original_trajectory
        row.setdefault("original_trajectory_sha256", sha256_text(original_trajectory))
    row["messages_sha256"] = sha256_text(row["messages"])
    return row


def aggregate_compression_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "compressed_rows": 0,
            "compression_rate": 0.0,
            "original_tokens": 0,
            "final_tokens": 0,
            "tokens_removed": 0,
            "information_characters_removed": 0,
            "information_tokens_removed": 0,
            "information_blocks_compressed": 0,
        }
    original_tokens = sum(int(row.get("original_token_length", 0)) for row in rows)
    final_tokens = sum(int(row.get("final_token_length", row.get("token_length", 0))) for row in rows)
    return {
        "rows": len(rows),
        "compressed_rows": sum(bool(row.get("compressed")) for row in rows),
        "compression_rate": sum(bool(row.get("compressed")) for row in rows) / len(rows),
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "tokens_removed": max(0, original_tokens - final_tokens),
        "information_characters_removed": sum(
            int(row.get("information_characters_removed", 0)) for row in rows
        ),
        "information_tokens_removed": sum(
            int(row.get("information_tokens_removed", 0)) for row in rows
        ),
        "information_blocks_compressed": sum(
            int(row.get("information_blocks_compressed", 0)) for row in rows
        ),
    }


def validate_tokenization(rows: Sequence[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    lengths: list[int] = []
    role_counts: Counter[str] = Counter()
    for row in rows:
        if "final_token_length" not in row:
            _apply_row_compression(row, tokenizer)
        messages = json.loads(row["messages"])
        length = _chat_token_length(tokenizer, messages)
        if length > MAX_LENGTH:
            raise ValueError(
                f"tokenized example exceeds {MAX_LENGTH}: {row['source_candidate_id']} has {length}"
            )
        if int(row.get("final_token_length", length)) != length:
            raise ValueError(
                f"stored final token length disagrees with tokenizer for "
                f"{row.get('source_candidate_id', '<unknown>')}: "
                f"stored={row.get('final_token_length')} actual={length}"
            )
        if not any(message.get("role") == "assistant" and message.get("content") for message in messages):
            raise ValueError(f"no assistant turn to supervise: {row['source_candidate_id']}")
        role_counts.update(str(message.get("role")) for message in messages)
        row["token_length"] = length
        row["supervised_turn_count"] = sum(
            message.get("role") == "assistant" and bool(message.get("content"))
            for message in messages
        )
        lengths.append(length)
    return {
        "max_token_length": max(lengths) if lengths else 0,
        "min_token_length": min(lengths) if lengths else 0,
        "mean_token_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "role_counts": dict(role_counts),
        "compression": aggregate_compression_stats(rows),
    }


def validate_sft_dataset_masks(train_path: Path, validation_path: Path, tokenizer_path: str) -> None:
    """Run the exact existing SFTDataset mask/length path on both outputs."""

    from transformers import AutoTokenizer

    from verl.utils.dataset.sft_dataset import SFTDataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    for path in (train_path, validation_path):
        dataset = SFTDataset(path, tokenizer, prompt_key="messages", response_key=None, max_length=MAX_LENGTH, truncation="error")
        for index in range(len(dataset)):
            sample = dataset[index]
            if int(sample["loss_mask"].sum().item()) <= 0:
                raise ValueError(f"empty loss mask in {path}:{index}")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = {
        "messages": [row["messages"] for row in rows],
        "question": [row["question"] for row in rows],
        "question_id": [row["question_id"] for row in rows],
        "data_source": [row["data_source"] for row in rows],
        "teacher_model": [row["teacher_model"] for row in rows],
        "teacher_type": [row["teacher_type"] for row in rows],
        "trajectory": [row["trajectory"] for row in rows],
        "original_trajectory": [row["original_trajectory"] for row in rows],
        "trajectory_sha256": [row["trajectory_sha256"] for row in rows],
        "original_trajectory_sha256": [row["original_trajectory_sha256"] for row in rows],
        "messages_sha256": [row["messages_sha256"] for row in rows],
        "original_token_length": [row["original_token_length"] for row in rows],
        "final_token_length": [row["final_token_length"] for row in rows],
        "compressed": [row["compressed"] for row in rows],
        "information_characters_removed": [
            row["information_characters_removed"] for row in rows
        ],
        "information_tokens_removed": [row["information_tokens_removed"] for row in rows],
        "information_blocks_compressed": [
            row["information_blocks_compressed"] for row in rows
        ],
    }
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table(columns), path)
    except ImportError:
        import pandas as pd

        pd.DataFrame(columns).to_parquet(path, index=False)


def write_manifest(path: Path, train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for split, rows in (("train", train), ("validation", validation)):
            for index, row in enumerate(rows):
                manifest = {
                    "split": split,
                    "row_index": index,
                    "question_id": row["question_id"],
                    "question_sha256": sha256_text(row["question_key"]),
                    "data_source": row["data_source"],
                    "teacher_model": row["teacher_model"],
                    "teacher_type": row["teacher_type"],
                    "source_record": row["source_record"],
                    "source_candidate_id": row["source_candidate_id"],
                    "sample_index": row["sample_index"],
                    "trajectory_sha256": row["trajectory_sha256"],
                    "original_trajectory_sha256": row["original_trajectory_sha256"],
                    "messages_sha256": row["messages_sha256"],
                    "original_token_length": row["original_token_length"],
                    "final_token_length": row["final_token_length"],
                    "compressed": row["compressed"],
                    "information_characters_original": row[
                        "information_characters_original"
                    ],
                    "information_characters_final": row["information_characters_final"],
                    "information_characters_removed": row[
                        "information_characters_removed"
                    ],
                    "information_tokens_original": row["information_tokens_original"],
                    "information_tokens_final": row["information_tokens_final"],
                    "information_tokens_removed": row["information_tokens_removed"],
                    "information_blocks": row["information_blocks"],
                    "information_blocks_compressed": row[
                        "information_blocks_compressed"
                    ],
                    "token_length": row["token_length"],
                    "supervised_turn_count": row["supervised_turn_count"],
                }
                handle.write(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    files = [output_dir / name for name in ("train.parquet", "validation.parquet", "stats.json", "manifest.jsonl")]
    if output_dir.exists() and any(path.exists() for path in files) and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite explicitly: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in files:
            if path.is_file():
                path.unlink()


def build(args: argparse.Namespace) -> dict[str, Any]:
    single_path = Path(args.single_accepted)
    multi_path = Path(args.multi_accepted)
    output_dir = Path(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    rows = load_and_validate_inputs(single_path, multi_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    token_stats = validate_tokenization(rows, tokenizer)
    # Compression changes the final serialized trajectory hash.  Re-run the
    # duplicate check after compression as well as on the original inputs.
    validate_unique_rows(rows)
    train, validation = stratified_split(rows, args.seed)
    # Recheck after split so all final records carry the exact metadata.
    validate_tokenization(train, tokenizer)
    validate_tokenization(validation, tokenizer)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    write_parquet(train_path, train)
    write_parquet(validation_path, validation)
    validate_sft_dataset_masks(train_path, validation_path, args.tokenizer_path)
    write_manifest(output_dir / "manifest.jsonl", train, validation)

    source_counts = {
        split: dict(Counter(row["data_source"] for row in records))
        for split, records in (("train", train), ("validation", validation))
    }
    stats = {
        "status": "completed",
        "counts": {"total": len(rows), "train": len(train), "validation": len(validation)},
        "quotas": dict(QUOTAS),
        "split": {"method": "stratified", "ratio": "90/10", "seed": args.seed, "question_overlap": 0},
        "source_counts": source_counts,
        "teacher_counts": {
            split: dict(Counter(row["teacher_model"] for row in records))
            for split, records in (("train", train), ("validation", validation))
        },
        "tokenization": {"tokenizer": args.tokenizer_path, "max_length": MAX_LENGTH, **token_stats},
        "compression": aggregate_compression_stats(rows),
        "masking": {
            "information_role": "user",
            "supervised_roles": ["assistant"],
            "validated_with": "verl.utils.dataset.SFTDataset",
            "truncation": "error",
        },
        "deduplication": {"question_keys": len({row["question_key"] for row in rows}), "trajectory_hashes": len({row["trajectory_sha256"] for row in rows})},
        "secret_policy": "manifest/stats contain no API keys or raw API authorization fields",
        "output_files": {
            "train": str(train_path),
            "validation": str(validation_path),
            "stats": str(output_dir / "stats.json"),
            "manifest": str(output_dir / "manifest.jsonl"),
        },
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


class FakeTokenizer:
    """Tiny deterministic tokenizer for the offline builder self-test."""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]], *, tokenize: bool, add_generation_prompt: bool) -> list[int]:
        text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        return list(range(max(1, len(text))))


def self_test() -> dict[str, Any]:
    one = "<think>Find evidence.</think><search>author</search><information>Doc</information><think>It says Ada.</think><answer>Ada</answer>"
    messages, answer, searches = trajectory_to_messages("Who?", one, expected_searches=(1, 1))
    assert answer == "Ada" and searches == 1
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2]["content"].startswith("<information>")
    assert messages[2]["role"] == "user"
    fake = FakeTokenizer()
    assert len(_token_ids(fake, messages)) > 0
    long_information = "BEGIN-" + (" useful middle evidence" * 180) + "-END"
    long_trajectory = (
        "<think>  Find evidence.  </think><search> author </search>"
        f"<information>{long_information}</information>"
        "<think> final reasoning </think><answer> Ada </answer>"
    )
    long_messages, _, _ = trajectory_to_messages(
        "  Who is the author?  ", long_trajectory, expected_searches=(1, 1)
    )
    compressed_messages, compression = compress_messages_to_budget(
        long_messages, fake, max_length=800
    )
    compressed_messages_again, compression_again = compress_messages_to_budget(
        long_messages, fake, max_length=800
    )
    assert compression["compressed"]
    assert compression["final_token_length"] <= 800
    assert canonical(compressed_messages) == canonical(compressed_messages_again)
    assert compression == compression_again
    assert compressed_messages[0]["content"] == long_messages[0]["content"]
    assert compressed_messages[1]["content"] == long_messages[1]["content"]
    assert compressed_messages[-1]["content"] == long_messages[-1]["content"]
    compressed_body = str(_information_body(compressed_messages[2]["content"]))
    assert COMPRESSION_MARKER in compressed_body
    assert compressed_body.startswith("BEGIN-") and compressed_body.endswith("-END")
    assert messages_to_trajectory(compressed_messages).startswith("<think>  Find evidence.  ")
    try:
        compress_messages_to_budget(
            [
                {"role": "user", "content": "x" * 4000},
                {"role": "assistant", "content": "ok"},
            ],
            fake,
            max_length=100,
        )
    except ValueError as exc:
        assert "fixed non-information content alone" in str(exc)
    else:
        raise AssertionError("fixed non-information overflow unexpectedly passed")
    invalid = "<think>x</think><search></search><information>i</information><think>x</think><answer>a</answer>"
    test_rows = [
        {
            "messages": canonical(messages),
            "question": "Who?",
            "question_id": "synthetic",
            "question_key": "who?",
            "data_source": "nq",
            "teacher_model": "Qwen2.5-7B-Instruct",
            "teacher_type": "local_hf",
            "source_record": "synthetic",
            "source_candidate_id": "synthetic__sample_0",
            "sample_index": 0,
            "gold_answers": ["Ada"],
            "trajectory_sha256": sha256_text(one),
            "messages_sha256": sha256_text(canonical(messages)),
            "search_count": 1,
            "trajectory": one,
        }
    ]
    validate_tokenization(test_rows, fake)
    assert test_rows[0]["token_length"] > 0 and test_rows[0]["supervised_turn_count"] == 2
    try:
        parse_trajectory(invalid, expected_searches=(1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid trajectory unexpectedly passed")

    synthetic_rows: list[dict[str, Any]] = []
    for source, quota in QUOTAS.items():
        for index in range(quota):
            synthetic_rows.append(
                {
                    **test_rows[0],
                    "question": f"{source} question {index}",
                    "question_id": f"{source}:{index}",
                    "question_key": f"{source} question {index}",
                    "data_source": source,
                    "trajectory_sha256": sha256_text(f"{source}:{index}"),
                    "source_candidate_id": f"{source}:{index}__sample_0",
                }
            )
    validate_unique_rows(synthetic_rows)
    train, val = stratified_split(synthetic_rows, seed=42)
    assert len(train) == 1350 and len(val) == 150
    assert not ({row["question_key"] for row in train} & {row["question_key"] for row in val})
    assert Counter(row["data_source"] for row in train + val) == Counter(
        {source: quota for source, quota in QUOTAS.items()}
    )
    duplicate_question = list(synthetic_rows)
    duplicate_question.append(dict(synthetic_rows[0], question_id="different-id"))
    try:
        validate_unique_rows(duplicate_question)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate question unexpectedly passed")
    return {
        "status": "passed",
        "checks": [
            "trajectory_to_alternating_messages",
            "information_user_role_for_masking",
            "assistant_supervision_nonempty",
            "QA-EM_and_strict_search_format",
            "exact quotas and deterministic stratified 90/10 split",
            "no question leakage and duplicate rejection",
            "tokenizer_length_helper",
            "no_external_call",
        ],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single-accepted", default=DEFAULT_SINGLE)
    p.add_argument("--multi-accepted", default=DEFAULT_MULTI)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return 0
    try:
        build(args)
    except Exception as exc:
        print(f"MIXED SFT BUILD FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
