#!/usr/bin/env python3
"""Construct the bounded Qwen2.5-7B single-hop SFT source set.

The production run samples questions from the local NQ training parquet,
generates two candidates per question with the existing local Qwen teacher,
and retains at most one candidate per question.  A candidate is eligible only
when the existing pilot's strict one-search format and project QA-EM both
pass.  Candidates are appended and fsynced after every question, so an
interrupted run can be resumed without regenerating completed candidates.

This script intentionally requires one visible physical GPU, exposed as
``CUDA_VISIBLE_DEVICES=4`` by default.  It never writes outside its production
output directory and never touches the later SFT/RL stages.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from generate_singlehop_sft_teacher_pilot import (
    ANSWER_INSTRUCTION,
    QUERY_INSTRUCTION,
    TeacherGenerator,
    _answer_em,
    _answers_list,
    _json_safe,
    _passages_to_string,
    _retrieve,
    extract_answer,
    extract_query,
    validate_flat_trajectory,
)


DEFAULT_MODEL_PATH = "/home/luwa/Documents/Tree-GRPO/models/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_PATH = "/home/luwa/Documents/Search-R1/data/nq_search/train.parquet"
DEFAULT_OUTPUT_DIR = "/home/luwa/Documents/Tree-GRPO/data/teacher_singlehop_qwen7b_prod_20260911"
DEFAULT_TARGET_ACCEPTED = 1000
DEFAULT_SAMPLES_PER_QUESTION = 2
DEFAULT_SEED = 42
DEFAULT_RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_RETRIEVAL_TOPK = 3
DEFAULT_GENERATION_BATCH_SIZE = 2
DEFAULT_MAX_QUESTIONS = 5000


def question_key(question: str) -> str:
    return " ".join(str(question).split()).casefold()


def read_questions(dataset_path: str, seed: int) -> tuple[list[dict[str, Any]], int]:
    """Read and deterministically shuffle unique NQ training questions."""

    import pandas as pd

    frame = pd.read_parquet(dataset_path)
    required = {"id", "question", "golden_answers"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    rows = frame.to_dict(orient="records")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index in order:
        row = rows[row_index]
        question = str(row["question"]).strip()
        key = question_key(question)
        if not key or key in seen:
            continue
        gold_answers = _answers_list(row["golden_answers"])
        if not gold_answers:
            continue
        seen.add(key)
        selected.append(
            {
                "row_index": int(row_index),
                "question_id": str(row["id"]),
                "question": question,
                "question_key": key,
                "gold_answers": gold_answers,
                "source_dataset": str(dataset_path),
                "source_split": "train",
            }
        )
    return selected, len(rows)


class JSONLAppender:
    """Append one JSON object atomically enough for process interruption."""

    def __init__(self, path: Path, *, append: bool) -> None:
        self.path = path
        self.handle = path.open("a" if append else "w", encoding="utf-8")
        self.lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock:
            self.handle.write(line)
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        with self.lock:
            self.handle.close()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prepare_output_dir(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    paths = [output_dir / name for name in ("candidates.jsonl", "accepted.jsonl", "summary.json")]
    existing = [path for path in paths if path.exists()]
    if existing and not resume and not overwrite:
        raise FileExistsError(
            "production output exists; pass --resume or explicit --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths:
            if path.is_file():
                path.unlink()


def _candidate_specs(item: Mapping[str, Any], samples_per_question: int) -> list[dict[str, Any]]:
    return [
        {
            **dict(item),
            "candidate_id": f"{item['question_id']}__sample_{sample_index}",
            "sample_index": sample_index,
        }
        for sample_index in range(samples_per_question)
    ]


def generate_candidates(
    specs: Sequence[Mapping[str, Any]],
    generator: TeacherGenerator,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Generate, retrieve, answer, and score a small candidate group."""

    if not specs:
        return []
    query_prompts = [QUERY_INSTRUCTION.format(question=item["question"]) for item in specs]
    raw_query_outputs = generator.generate(
        query_prompts,
        args.query_max_new_tokens,
        args.generation_batch_size,
    )
    if len(raw_query_outputs) != len(specs):
        raise RuntimeError("teacher query generation returned the wrong number of outputs")

    queries = [extract_query(output) for output in raw_query_outputs]
    retrieval_results: list[Any | None] = [None] * len(specs)
    retrieval_errors: list[str | None] = [None] * len(specs)
    nonempty = [index for index, query in enumerate(queries) if query]
    if nonempty:
        try:
            result_values, _ = _retrieve(
                args.retrieval_url,
                [queries[index] for index in nonempty if queries[index]],
                args.retrieval_topk,
                args.retrieval_timeout,
            )
            for index, result in zip(nonempty, result_values):
                retrieval_results[index] = result
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for index in nonempty:
                retrieval_errors[index] = message
    for index, query in enumerate(queries):
        if not query:
            retrieval_errors[index] = "missing_search_query"

    information = [_passages_to_string(result) for result in retrieval_results]
    answer_prompts = [
        ANSWER_INSTRUCTION.format(
            question=item["question"],
            query=queries[index] or "",
            information=information[index],
        )
        for index, item in enumerate(specs)
    ]
    raw_answer_outputs = generator.generate(
        answer_prompts,
        args.answer_max_new_tokens,
        args.generation_batch_size,
    )
    if len(raw_answer_outputs) != len(specs):
        raise RuntimeError("teacher answer generation returned the wrong number of outputs")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(specs):
        raw_query = str(raw_query_outputs[index]).strip()
        raw_answer = str(raw_answer_outputs[index]).strip()
        format_correct, format_reasons, trajectory = validate_flat_trajectory(
            raw_query,
            information[index],
            raw_answer,
        )
        extracted_answer = extract_answer(raw_answer)
        answer_correct = _answer_em(extracted_answer, item["gold_answers"])
        failures = list(format_reasons)
        if extracted_answer is None:
            failures.append("final_answer_extraction_failed")
        elif not answer_correct:
            failures.append("final_answer_em_failed")
        if retrieval_errors[index]:
            failures.append("retrieval_failed")
        records.append(
            {
                "candidate_id": item["candidate_id"],
                "question_id": item["question_id"],
                "question_key": item["question_key"],
                "row_index": item["row_index"],
                "question": item["question"],
                "sample_index": item["sample_index"],
                "seed": args.seed,
                "gold_answers": item["gold_answers"],
                "teacher_model": "Qwen2.5-7B-Instruct",
                "teacher_type": "local_hf",
                "source_dataset": item["source_dataset"],
                "source_split": item["source_split"],
                "raw_query_output": raw_query,
                "extracted_query": queries[index],
                "retrieval_url": args.retrieval_url,
                "retrieval_topk": args.retrieval_topk,
                "retrieval_result": retrieval_results[index],
                "retrieval_error": retrieval_errors[index],
                "information": information[index],
                "raw_answer_output": raw_answer,
                "extracted_answer": extracted_answer,
                "trajectory": trajectory,
                "format_correct": bool(format_correct),
                "answer_correct": bool(answer_correct),
                "eligible": bool(format_correct and answer_correct),
                "format_failure_reasons": format_reasons,
                "failure_reasons": list(dict.fromkeys(failures)),
            }
        )
    return records


def _validate_existing(
    records: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    samples_per_question: int,
) -> None:
    question_by_id = {str(item["question_id"]): item for item in questions}
    seen_ids: set[str] = set()
    for record in records:
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen_ids:
            raise ValueError(f"duplicate or empty candidate_id in resumed output: {candidate_id!r}")
        seen_ids.add(candidate_id)
        question_id = str(record.get("question_id", ""))
        item = question_by_id.get(question_id)
        if item is None:
            raise ValueError(f"resumed candidate is not from the deterministic NQ pool: {question_id}")
        expected_prefix = f"{question_id}__sample_"
        if not candidate_id.startswith(expected_prefix):
            raise ValueError(f"candidate ID does not match question ID: {candidate_id}")
        try:
            sample_index = int(candidate_id[len(expected_prefix) :])
        except ValueError as exc:
            raise ValueError(f"invalid sample index in candidate ID: {candidate_id}") from exc
        if not 0 <= sample_index < samples_per_question:
            raise ValueError(f"sample index out of range: {candidate_id}")
        if str(record.get("question_key", question_key(item["question"]))) != item["question_key"]:
            raise ValueError(f"question key mismatch in resumed candidate: {candidate_id}")


def _select_accepted(
    records: Sequence[Mapping[str, Any]],
    target: int,
    question_order: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select at most one eligible trajectory per question in stable order."""

    by_question: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_question.setdefault(str(record["question_id"]), []).append(record)
    selected: list[dict[str, Any]] = []
    for item in question_order:
        candidates = sorted(
            by_question.get(str(item["question_id"]), []),
            key=lambda record: int(record.get("sample_index", 0)),
        )
        eligible = [record for record in candidates if bool(record.get("eligible"))]
        if eligible:
            selected.append(dict(eligible[0]))
        if len(selected) >= target:
            break
    return selected[:target]


def build_summary(
    records: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    dataset_rows: int,
    question_count: int,
    status: str,
    elapsed_seconds: float,
    generator: TeacherGenerator | None,
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for record in records:
        reason_counts.update(str(reason) for reason in record.get("failure_reasons", []))
    format_count = sum(bool(record.get("format_correct")) for record in records)
    answer_count = sum(bool(record.get("answer_correct")) for record in records)
    eligible_count = sum(bool(record.get("eligible")) for record in records)
    by_question = {str(record.get("question_id")) for record in records}
    return {
        "status": status,
        "config": {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "output_dir": args.output_dir,
            "target_accepted": args.target_accepted,
            "samples_per_question": args.samples_per_question,
            "seed": args.seed,
            "retrieval_url": args.retrieval_url,
            "retrieval_topk": args.retrieval_topk,
            "generation_batch_size": args.generation_batch_size,
            "query_max_new_tokens": args.query_max_new_tokens,
            "answer_max_new_tokens": args.answer_max_new_tokens,
            "max_questions": args.max_questions,
            "format_filter": "exactly one think/search + information + think/answer",
            "answer_filter": "verl qa_em exact-match normalization against any golden answer",
            "selection_policy": "one eligible trajectory per unique question, stable sample order",
        },
        "dataset": {
            "rows": dataset_rows,
            "question_pool_unique": question_count,
            "questions_processed": len(by_question),
        },
        "candidates": {
            "written": len(records),
            "eligible_total": eligible_count,
            "accepted_selected": len(accepted),
            "target_reached": len(accepted) == args.target_accepted,
            "format_correct": format_count,
            "answer_correct": answer_count,
            "questions_with_eligible_candidate": len(
                {str(record.get("question_id")) for record in records if record.get("eligible")}
            ),
        },
        "metrics": {
            "format_rate": format_count / len(records) if records else 0.0,
            "answer_rate": answer_count / len(records) if records else 0.0,
            "eligible_rate": eligible_count / len(records) if records else 0.0,
            "accepted_rate_over_questions": len(accepted) / question_count if question_count else 0.0,
        },
        "failure_reason_counts": dict(reason_counts.most_common()),
        "timing_seconds": {"wall_time": round(elapsed_seconds, 3)},
        "gpu": generator.memory_stats() if generator is not None else None,
        "provenance": {
            "teacher_model": "Qwen2.5-7B-Instruct",
            "teacher_type": "local_hf",
            "production_script": "scripts/data_process/generate_singlehop_sft_teacher_production.py",
            "api_key_persisted": False,
        },
    }


def run_production(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir, resume=args.resume, overwrite=args.overwrite)
    questions, dataset_rows = read_questions(args.dataset_path, args.seed)
    if len(questions) < args.max_questions:
        max_questions = len(questions)
    else:
        max_questions = args.max_questions
    question_order = questions[:max_questions]

    candidate_path = output_dir / "candidates.jsonl"
    accepted_path = output_dir / "accepted.jsonl"
    candidate_records = load_jsonl(candidate_path)
    _validate_existing(candidate_records, question_order, args.samples_per_question)
    candidate_by_id = {str(record["candidate_id"]): record for record in candidate_records}

    accepted = _select_accepted(candidate_records, args.target_accepted, question_order)
    write_jsonl(accepted_path, accepted)
    if len(accepted) >= args.target_accepted:
        summary = build_summary(
            candidate_records,
            accepted,
            args=args,
            dataset_rows=dataset_rows,
            question_count=len(question_order),
            status="target_reached",
            elapsed_seconds=time.monotonic() - started,
            generator=None,
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    appender = JSONLAppender(candidate_path, append=bool(candidate_records))
    generator: TeacherGenerator | None = None
    questions_processed: set[str] = set()
    try:
        for item in question_order:
            if len(accepted) >= args.target_accepted:
                break
            specs = _candidate_specs(item, args.samples_per_question)
            pending = [spec for spec in specs if spec["candidate_id"] not in candidate_by_id]
            if pending:
                if generator is None:
                    generator = TeacherGenerator(args)
                new_records = generate_candidates(pending, generator, args)
                for record in new_records:
                    appender.append(record)
                    candidate_records.append(record)
                    candidate_by_id[str(record["candidate_id"])] = record
            questions_processed.add(str(item["question_id"]))
            accepted = _select_accepted(candidate_records, args.target_accepted, question_order)
            write_jsonl(accepted_path, accepted)
            status = "target_reached" if len(accepted) >= args.target_accepted else "in_progress"
            summary = build_summary(
                candidate_records,
                accepted,
                args=args,
                dataset_rows=dataset_rows,
                question_count=len(questions_processed),
                status=status,
                elapsed_seconds=time.monotonic() - started,
                generator=generator,
            )
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if len(accepted) >= args.target_accepted:
                break
    finally:
        appender.close()

    accepted = _select_accepted(candidate_records, args.target_accepted, question_order)
    write_jsonl(accepted_path, accepted)
    status = "target_reached" if len(accepted) == args.target_accepted else "question_pool_exhausted"
    summary = build_summary(
        candidate_records,
        accepted,
        args=args,
        dataset_rows=dataset_rows,
        question_count=len(questions_processed),
        status=status,
        elapsed_seconds=time.monotonic() - started,
        generator=generator,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _self_test() -> dict[str, Any]:
    item = {
        "question_id": "train:self",
        "question_key": "who is the author",
        "question": "Who is the author?",
        "row_index": 0,
        "gold_answers": ["Ada"],
        "source_dataset": "synthetic",
        "source_split": "train",
    }
    specs = _candidate_specs(item, 2)
    assert [spec["candidate_id"] for spec in specs] == ["train:self__sample_0", "train:self__sample_1"]
    records = [
        {**specs[0], "eligible": True, "sample_index": 0},
        {**specs[1], "eligible": True, "sample_index": 1},
    ]
    selected = _select_accepted(records, 1, [item])
    assert len(selected) == 1 and selected[0]["candidate_id"].endswith("sample_0")
    assert len(_select_accepted(records, 3, [item])) == 1
    assert question_key("  Who   Is the Author? ") == "who is the author?"
    return {
        "status": "passed",
        "checks": [
            "two_stable_candidate_ids_per_question",
            "one_accepted_max_per_question",
            "exact_target_selection_without_overshoot",
            "casefold_question_key",
            "no_model_retriever_or_api_call",
        ],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-accepted", type=int, default=DEFAULT_TARGET_ACCEPTED)
    p.add_argument("--samples-per-question", type=int, default=DEFAULT_SAMPLES_PER_QUESTION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--retrieval-url", default=DEFAULT_RETRIEVAL_URL)
    p.add_argument("--retrieval-topk", type=int, default=DEFAULT_RETRIEVAL_TOPK)
    p.add_argument("--retrieval-timeout", type=float, default=90.0)
    p.add_argument("--generation-batch-size", type=int, default=DEFAULT_GENERATION_BATCH_SIZE)
    p.add_argument("--query-max-new-tokens", type=int, default=192)
    p.add_argument("--answer-max-new-tokens", type=int, default=192)
    p.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS)
    p.add_argument("--expected-cuda-visible", default="4")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.target_accepted <= 0 or args.samples_per_question != 2:
        raise ValueError("target-accepted must be positive and samples-per-question must be exactly 2")
    if args.retrieval_topk != 3:
        raise ValueError("production single-hop generation requires retriever topk=3")
    if args.max_questions <= 0 or args.generation_batch_size <= 0:
        raise ValueError("max-questions and generation-batch-size must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return 0
    try:
        validate_args(args)
        summary = run_production(args)
        if summary["status"] != "target_reached":
            print("WARNING: target was not reached before the deterministic question pool ended", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"PRODUCTION SINGLEHOP FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
