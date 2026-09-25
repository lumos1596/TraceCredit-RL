#!/usr/bin/env python3
"""Construct bounded DeepSeek-Flash multi-hop SFT trajectories.

This is the production counterpart of the fixed DeepSeek pilot.  It reuses
that pilot's API/tool-calling state machine and real local retriever, but
selects a deterministic, non-overlapping training-question pool and continues
source by source until the exact accepted quotas are reached or a configured
candidate budget is exhausted:

    hotpotqa: 200, 2wikimultihopqa: 175, musique: 125

Each question gets two candidate trajectories and at most one is selected for
SFT.  ``candidates.jsonl`` and ``interactions.jsonl`` are append-only and
fsynced; ``accepted.jsonl`` is rebuilt atomically from the candidates after
each batch.  The API key is read only from ``DEEPSEEK_API_KEY`` and is passed
to the reused redacting writer; it is never written to an output file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import generate_multihop_sft_teacher_pilot_deepseek as fixed


DATASET_SOURCES = ("hotpotqa", "2wikimultihopqa", "musique")
TARGETS = {"hotpotqa": 200, "2wikimultihopqa": 175, "musique": 125}
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_DATASET_PATH = "/home/luwa/Documents/Tree-GRPO/data/multihopqa_search_mixed_402020_20260830/train.parquet"
DEFAULT_EXCLUDE_SUMMARIES = (
    "/home/luwa/Documents/Tree-GRPO/data/teacher_sft_pilot_qwen2.5_7b_multihop/summary.json",
    "/home/luwa/Documents/Tree-GRPO/data/teacher_sft_pilot_deepseek_v4_flash_multihop_fixed/summary.json",
)
DEFAULT_OUTPUT_DIR = "/home/luwa/Documents/Tree-GRPO/data/teacher_multihop_dsflash_prod_20260911"
DEFAULT_RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_MAX_CANDIDATES_PER_SOURCE = 6000
DEFAULT_QUESTION_BATCH_SIZE = 8
DEFAULT_MAX_API_FAILURES_PER_RUN = 32
SYSTEMIC_API_STATUS_CODES = frozenset({400, 401, 402, 403, 404})
API_FAILURE_REASON_PREFIX = "deepseek_api_failed"


def question_key(question: str) -> str:
    return " ".join(str(question).split()).casefold()


def load_excluded_question_ids(paths: Sequence[str | os.PathLike[str]]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing pilot summary required to prove no paired-question reuse: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids = payload.get("dataset", {}).get("selected_question_ids")
        if not isinstance(ids, list):
            raise ValueError(f"summary has no dataset.selected_question_ids: {path}")
        excluded.update(str(item) for item in ids)
    if not excluded:
        raise ValueError("the exclusion summaries did not contain any question IDs")
    return excluded


def read_question_pool(
    dataset_path: str,
    seed: int,
    excluded_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build source-stratified, globally question-deduplicated pool."""

    import pandas as pd

    frame = pd.read_parquet(dataset_path)
    required = {"data_source", "prompt", "reward_model", "extra_info"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        record = fixed._row_question_record(row, row_index)
        source = str(record["data_source"])
        if source in DATASET_SOURCES:
            record["question_key"] = question_key(record["question"])
            record["source_dataset"] = str(dataset_path)
            record["source_split"] = "train"
            by_source[source].append(record)

    pool: dict[str, list[dict[str, Any]]] = {source: [] for source in DATASET_SOURCES}
    globally_seen_questions: set[str] = set()
    skipped_excluded = Counter()
    skipped_duplicate = Counter()
    for source_index, source in enumerate(DATASET_SOURCES):
        order = list(range(len(by_source[source])))
        random.Random(seed + source_index).shuffle(order)
        for index in order:
            item = by_source[source][index]
            question_id = str(item["question_id"])
            key = str(item["question_key"])
            if question_id in excluded_ids:
                skipped_excluded[source] += 1
                continue
            if not key or key in globally_seen_questions:
                skipped_duplicate[source] += 1
                continue
            globally_seen_questions.add(key)
            pool[source].append(item)

    stats = {
        "dataset_rows": len(frame),
        "pool_unique_questions": {source: len(pool[source]) for source in DATASET_SOURCES},
        "excluded_paired_question_ids": sorted(excluded_ids),
        "skipped_excluded_by_source": dict(skipped_excluded),
        "skipped_duplicate_question_by_source": dict(skipped_duplicate),
    }
    return pool, stats


def candidate_specs(item: Mapping[str, Any], samples_per_question: int = 2) -> list[dict[str, Any]]:
    if samples_per_question != 2:
        raise ValueError("production multi-hop generation requires exactly two candidates per question")
    return [
        {
            **dict(item),
            "candidate_id": f"{item['question_id']}__sample_{sample_index}",
            "sample_index": sample_index,
        }
        for sample_index in range(samples_per_question)
    ]


def decorate_record(record: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(record)
    result.update(
        {
            "teacher_model": args.model,
            "teacher_type": "deepseek_api",
            "source_dataset": args.dataset_path,
            "source_split": "train",
            "production_script": "scripts/data_process/generate_multihop_sft_teacher_production_deepseek.py",
            "api_key_persisted": False,
        }
    )
    return result


def _failure_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons = record.get("failure_reasons", [])
    if not isinstance(reasons, list):
        return []
    return [str(reason) for reason in reasons]


def has_actual_generation(record: Mapping[str, Any]) -> bool:
    """Whether this row contains at least one actual model response.

    A failed API request can still leave an audit row, but it cannot consume a
    candidate slot unless the model produced a response.  The production
    writer stores all API response turns in ``raw_generation_turns``; the
    fallback checks keep the classifier useful for older hand-written rows.
    """

    turns = record.get("raw_generation_turns")
    if isinstance(turns, list) and turns:
        return True
    return bool(record.get("trajectory")) or record.get("extracted_answer") is not None


def is_api_failure(record: Mapping[str, Any]) -> bool:
    if bool(record.get("api_failure")):
        return True
    return any(
        reason.split(":", 1)[0].strip() == API_FAILURE_REASON_PREFIX
        for reason in _failure_reasons(record)
    )


def is_retryable_api_failure(record: Mapping[str, Any]) -> bool:
    """Identify an API-failure placeholder that did not generate any output."""

    return is_api_failure(record) and not has_actual_generation(record)


def is_effective_candidate(record: Mapping[str, Any]) -> bool:
    """Rows that consume the effective candidate budget."""

    return not is_retryable_api_failure(record)


def _api_status_code(record: Mapping[str, Any]) -> int | None:
    value = record.get("api_failure_status_code")
    if isinstance(value, int):
        return value
    for reason in _failure_reasons(record):
        match = re.search(r"DeepSeek HTTP (\d{3})", reason, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def systemic_api_failure_reason(record: Mapping[str, Any]) -> str | None:
    """Return a stable pause reason for configuration/auth/billing failures."""

    if not is_retryable_api_failure(record):
        return None
    status_code = _api_status_code(record)
    if status_code in SYSTEMIC_API_STATUS_CODES:
        return f"http_{status_code}"
    return None


def _annotate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    api_failure = is_api_failure(result)
    retryable = is_retryable_api_failure(result)
    result.update(
        {
            "actual_generation": has_actual_generation(result),
            "api_failure": api_failure,
            "retryable_api_failure": retryable,
            "api_failure_status_code": _api_status_code(result),
        }
    )
    return result


def validate_existing_candidates(
    records: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
    excluded_ids: set[str],
) -> None:
    item_by_id = {
        str(item["question_id"]): item
        for items in pool.values()
        for item in items
    }
    records_by_candidate_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"duplicate or empty resumed candidate_id: {candidate_id!r}")
        records_by_candidate_id[candidate_id].append(record)
        if "__sample_" not in candidate_id:
            raise ValueError(f"invalid resumed candidate ID: {candidate_id}")
        question_id, sample_text = candidate_id.rsplit("__sample_", 1)
        if question_id in excluded_ids:
            raise ValueError(f"resumed output reuses a paired pilot question: {question_id}")
        item = item_by_id.get(question_id)
        if item is None:
            raise ValueError(f"resumed candidate is outside the deterministic production pool: {question_id}")
        try:
            sample_index = int(sample_text)
        except ValueError as exc:
            raise ValueError(f"invalid sample index in {candidate_id}") from exc
        if sample_index not in (0, 1):
            raise ValueError(f"sample index must be 0 or 1 in {candidate_id}")
        if str(record.get("question_id")) != question_id:
            raise ValueError(f"question_id mismatch in resumed candidate: {candidate_id}")
        if str(record.get("data_source")) != str(item["data_source"]):
            raise ValueError(f"data_source mismatch in resumed candidate: {candidate_id}")
        if record.get("sample_index") is not None and int(record["sample_index"]) != sample_index:
            raise ValueError(f"sample_index mismatch in resumed candidate: {candidate_id}")

    for candidate_id, candidate_records in records_by_candidate_id.items():
        effective_count = sum(is_effective_candidate(record) for record in candidate_records)
        if effective_count > 1:
            raise ValueError(
                "duplicate effective candidate_id in resumed output: "
                f"{candidate_id!r} ({effective_count} effective rows)"
            )


def select_accepted(
    records: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Stable at-most-one-per-question selection, capped per source."""

    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[str(record.get("question_id"))].append(record)
    selected: list[dict[str, Any]] = []
    for source in DATASET_SOURCES:
        count = 0
        for item in pool[source]:
            candidates = sorted(
                by_question.get(str(item["question_id"]), []),
                key=lambda record: int(record.get("sample_index", 0)),
            )
            eligible = [
                record
                for record in candidates
                if is_effective_candidate(record) and bool(record.get("eligible"))
            ]
            if eligible:
                selected.append(dict(eligible[0]))
                count += 1
            if count >= int(targets[source]):
                break
    return selected


def has_unresolved_retryable_slots(
    records: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Mapping[str, int],
) -> bool:
    """Whether an unmet source quota still has a retryable candidate slot."""

    accepted = select_accepted(records, pool, targets)
    effective_ids = {
        str(record.get("candidate_id"))
        for record in records
        if is_effective_candidate(record)
    }
    retryable_ids = {
        str(record.get("candidate_id"))
        for record in records
        if is_retryable_api_failure(record)
    }
    for source in DATASET_SOURCES:
        accepted_source = sum(str(record.get("data_source")) == source for record in accepted)
        if accepted_source >= int(targets[source]):
            continue
        for item in pool[source]:
            for spec in candidate_specs(item):
                candidate_id = str(spec["candidate_id"])
                if candidate_id in retryable_ids and candidate_id not in effective_ids:
                    return True
    return False


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(fixed._redact(record), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_output_dir(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    paths = [output_dir / name for name in ("interactions.jsonl", "candidates.jsonl", "accepted.jsonl", "summary.json")]
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


def source_funnel(
    records: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    source: str,
    target: int,
) -> dict[str, Any]:
    source_records = [record for record in records if str(record.get("data_source")) == source]
    effective_records = [record for record in source_records if is_effective_candidate(record)]
    generated_records = [record for record in source_records if has_actual_generation(record)]
    api_failure_records = [record for record in source_records if is_api_failure(record)]
    retryable_api_failure_records = [record for record in source_records if is_retryable_api_failure(record)]
    source_accepted = [record for record in accepted if str(record.get("data_source")) == source]
    metrics = fixed._metrics(source_records)
    effective_metrics = fixed._metrics(effective_records)
    return {
        "target_accepted": target,
        "candidate_count": len(source_records),
        "audit_rows": len(source_records),
        "effective_attempted": len(effective_records),
        "generated_candidates": len(generated_records),
        "api_failure_rows": len(api_failure_records),
        "retryable_api_failure_rows": len(retryable_api_failure_records),
        "format_correct": sum(bool(record.get("format_correct")) for record in effective_records),
        "answer_correct": sum(bool(record.get("answer_correct")) for record in effective_records),
        "eligible_total": sum(bool(record.get("eligible")) for record in effective_records),
        "accepted_selected": len(source_accepted),
        "target_reached": len(source_accepted) == target,
        "questions_with_candidate": len({str(record.get("question_id")) for record in effective_records}),
        "questions_with_eligible": len({str(record.get("question_id")) for record in effective_records if record.get("eligible")}),
        "metrics": metrics,
        "effective_metrics": effective_metrics,
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
    pool_stats: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    status: str,
    excluded_ids: set[str],
    elapsed_seconds: float,
    stop_reason: str | None = None,
    new_retryable_api_failures: int = 0,
) -> dict[str, Any]:
    per_source = {
        source: source_funnel(records, accepted, source, TARGETS[source])
        for source in DATASET_SOURCES
    }
    return {
        "status": status,
        "api_key_present": True,
        "api": {
            "model": args.model,
            "base_url": args.base_url,
            "thinking_requested": True,
            "reasoning_effort_requested": "high",
            "strict_search_tool_requested": True,
            "concurrency": args.concurrency,
            "api_key_persisted": False,
        },
        "config": {
            "model": args.model,
            "dataset_path": args.dataset_path,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "samples_per_question": args.samples_per_question,
            "retrieval_url": args.retrieval_url,
            "retrieval_topk": args.retrieval_topk,
            "max_searches": args.max_searches,
            "max_new_tokens": args.max_new_tokens,
            "max_continuations": args.max_continuations,
            "question_batch_size": args.question_batch_size,
            "max_candidates_per_source": args.max_candidates_per_source,
            "max_api_failures_per_run": args.max_api_failures_per_run,
            "eligibility": "format_correct AND answer_correct only",
            "format_filter": "project strict XML trajectory: 1-3 search/information rounds then think/answer",
            "answer_filter": "project QA-EM exact-match normalization against any gold answer",
        },
        "targets": dict(TARGETS),
        "dataset": {
            **dict(pool_stats),
            "production_question_ids_by_source": {
                source: [str(item["question_id"]) for item in pool[source]]
                for source in DATASET_SOURCES
            },
            "excluded_paired_question_ids": sorted(excluded_ids),
        },
        "candidates": {
            "written": len(records),
            "audit_rows_written": len(records),
            "effective_attempted": sum(is_effective_candidate(record) for record in records),
            "generated_candidates": sum(has_actual_generation(record) for record in records),
            "api_failure_rows": sum(is_api_failure(record) for record in records),
            "retryable_api_failure_rows": sum(is_retryable_api_failure(record) for record in records),
            "new_retryable_api_failures_this_run": new_retryable_api_failures,
            "accepted_selected": len(accepted),
            "target_total": sum(TARGETS.values()),
            "target_reached": len(accepted) == sum(TARGETS.values())
            and all(per_source[source]["target_reached"] for source in DATASET_SOURCES),
        },
        "per_source": per_source,
        "metrics": fixed._metrics(records),
        "effective_metrics": fixed._metrics(
            [record for record in records if is_effective_candidate(record)]
        ),
        "run_control": {
            "status": status,
            "stop_reason": stop_reason,
            "max_api_failures_per_run": args.max_api_failures_per_run,
        },
        "timing_seconds": {"wall_time": round(elapsed_seconds, 3)},
        "provenance": {
            "teacher_model": args.model,
            "teacher_type": "deepseek_api",
            "production_script": "scripts/data_process/generate_multihop_sft_teacher_production_deepseek.py",
            "secret_fields_written": False,
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return fixed._load_jsonl(path)


def run_production(args: argparse.Namespace) -> dict[str, Any]:
    if args.model != DEFAULT_MODEL:
        raise ValueError(f"this production script is pinned to {DEFAULT_MODEL!r}")
    if args.samples_per_question != 2 or args.retrieval_topk != 3:
        raise ValueError("production requires exactly 2 samples/question and retriever topk=3")
    if (
        args.concurrency < 1
        or args.question_batch_size < 1
        or args.max_candidates_per_source < 2
        or args.max_api_failures_per_run < 1
    ):
        raise ValueError("concurrency, question batch size, candidate budget, or API failure budget is invalid")
    api_key = fixed._require_api_key()
    excluded_ids = load_excluded_question_ids(args.exclude_summary)
    pool, pool_stats = read_question_pool(args.dataset_path, args.seed, excluded_ids)
    for source, target in TARGETS.items():
        if len(pool[source]) < target:
            raise ValueError(f"source {source} has only {len(pool[source])} unique production questions; target is {target}")

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, resume=args.resume, overwrite=args.overwrite)
    interaction_path = output_dir / "interactions.jsonl"
    candidate_path = output_dir / "candidates.jsonl"
    accepted_path = output_dir / "accepted.jsonl"
    existing_candidates = _load_jsonl(candidate_path)
    validate_existing_candidates(existing_candidates, pool, excluded_ids)
    # Only an effective row completes a candidate slot.  Historical API
    # placeholders remain in the audit file but deliberately do not block a
    # retry of the same deterministic candidate ID.
    candidate_by_id = {
        str(record["candidate_id"]): record
        for record in existing_candidates
        if is_effective_candidate(record)
    }
    accepted = select_accepted(existing_candidates, pool, TARGETS)
    write_jsonl_atomic(accepted_path, accepted)

    if len(accepted) == sum(TARGETS.values()) and all(
        sum(str(record.get("data_source")) == source for record in accepted) == TARGETS[source]
        for source in DATASET_SOURCES
    ):
        summary = build_summary(
            existing_candidates,
            accepted,
            pool,
            pool_stats,
            args,
            status="target_reached",
            excluded_ids=excluded_ids,
            elapsed_seconds=0.0,
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    events = _load_jsonl(interaction_path)
    events_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        candidate_id = str(event.get("candidate_id", ""))
        if candidate_id:
            events_by_candidate[candidate_id].append(event)

    append_existing = bool(existing_candidates or events) and not args.overwrite
    event_writer = fixed.JSONLAppender(interaction_path, append=append_existing, api_key=api_key)
    candidate_writer = fixed.JSONLAppender(candidate_path, append=bool(existing_candidates), api_key=api_key)
    client = fixed.DeepSeekClient(api_key, args)
    started = time.monotonic()
    attempted_ids_this_run: set[str] = set()
    new_retryable_api_failures = 0
    try:
        for source in DATASET_SOURCES:
            while True:
                accepted = select_accepted(existing_candidates, pool, TARGETS)
                accepted_source = sum(str(record.get("data_source")) == source for record in accepted)
                if accepted_source >= TARGETS[source]:
                    break
                source_count = sum(
                    str(record.get("data_source")) == source
                    and is_effective_candidate(record)
                    for record in existing_candidates
                )
                if source_count >= args.max_candidates_per_source:
                    break

                pending_specs: list[tuple[dict[str, Any], int]] = []
                for item in pool[source]:
                    specs = candidate_specs(item, args.samples_per_question)
                    missing = [
                        spec
                        for spec in specs
                        if str(spec["candidate_id"]) not in candidate_by_id
                        and str(spec["candidate_id"]) not in attempted_ids_this_run
                    ]
                    if not missing:
                        continue
                    remaining_budget = args.max_candidates_per_source - source_count - len(pending_specs)
                    if remaining_budget <= 0:
                        break
                    pending_specs.extend((spec, 0) for spec in missing[:remaining_budget])
                    if len({str(spec["question_id"]) for spec, _ in pending_specs}) >= args.question_batch_size:
                        break
                if not pending_specs:
                    break

                # A batch may contain more eligible trajectories than the
                # remaining quota.  All generated candidates are retained,
                # while accepted.jsonl is capped deterministically below.
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    futures = {
                        executor.submit(
                            fixed._run_candidate,
                            spec,
                            int(spec["sample_index"]),
                            args,
                            client,
                            event_writer,
                            events_by_candidate.get(str(spec["candidate_id"]), []),
                        ): spec
                        for spec, _ in pending_specs
                    }
                    completed: list[tuple[dict[str, Any], dict[str, Any]]] = []
                    for future in as_completed(futures):
                        spec = futures[future]
                        try:
                            state = future.result()
                        except Exception as exc:
                            state = fixed._new_state(spec, int(spec["sample_index"]), args.seed)
                            fixed._add_failure(state, "worker_failed", fixed._safe_error(exc, api_key))
                            state["status"] = "failed"
                            fixed._finalize_state(state)
                        record = _annotate_record(
                            decorate_record(fixed._records_for_output([state])[0], args)
                        )
                        completed.append((spec, record))
                systemic_reasons: list[str] = []
                for _spec, record in sorted(completed, key=lambda pair: str(pair[1]["candidate_id"])):
                    candidate_writer.append(record)
                    existing_candidates.append(record)
                    candidate_id = str(record["candidate_id"])
                    if is_effective_candidate(record):
                        candidate_by_id[candidate_id] = record
                    elif is_retryable_api_failure(record):
                        attempted_ids_this_run.add(candidate_id)
                        new_retryable_api_failures += 1
                        pause_reason = systemic_api_failure_reason(record)
                        if pause_reason is not None and pause_reason not in systemic_reasons:
                            systemic_reasons.append(pause_reason)

                accepted = select_accepted(existing_candidates, pool, TARGETS)
                write_jsonl_atomic(accepted_path, accepted)
                if systemic_reasons:
                    partial_status = "paused_api_error"
                    partial_reason = "systemic_api_failure:" + ",".join(sorted(systemic_reasons))
                elif new_retryable_api_failures >= args.max_api_failures_per_run:
                    partial_status = "paused_api_failure_budget"
                    partial_reason = (
                        "max_api_failures_per_run_reached:"
                        f"{args.max_api_failures_per_run}"
                    )
                else:
                    partial_status = "in_progress"
                    partial_reason = None
                partial = build_summary(
                    existing_candidates,
                    accepted,
                    pool,
                    pool_stats,
                    args,
                    status=partial_status,
                    excluded_ids=excluded_ids,
                    elapsed_seconds=time.monotonic() - started,
                    stop_reason=partial_reason,
                    new_retryable_api_failures=new_retryable_api_failures,
                )
                (output_dir / "summary.json").write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if systemic_reasons or new_retryable_api_failures >= args.max_api_failures_per_run:
                    return partial
    finally:
        event_writer.close()
        candidate_writer.close()

    accepted = select_accepted(existing_candidates, pool, TARGETS)
    write_jsonl_atomic(accepted_path, accepted)
    all_targets_reached = len(accepted) == sum(TARGETS.values()) and all(
        sum(str(record.get("data_source")) == source for record in accepted) == TARGETS[source]
        for source in DATASET_SOURCES
    )
    if all_targets_reached:
        status = "target_reached"
        stop_reason = None
    elif new_retryable_api_failures >= args.max_api_failures_per_run:
        status = "paused_api_failure_budget"
        stop_reason = f"max_api_failures_per_run_reached:{args.max_api_failures_per_run}"
    elif has_unresolved_retryable_slots(existing_candidates, pool, TARGETS):
        status = "retryable_failures_pending"
        stop_reason = "retryable_api_failure_slots_remain"
    else:
        status = "max_candidates_or_pool_exhausted"
        stop_reason = None
    summary = build_summary(
        existing_candidates,
        accepted,
        pool,
        pool_stats,
        args,
        status=status,
        excluded_ids=excluded_ids,
        elapsed_seconds=time.monotonic() - started,
        stop_reason=stop_reason,
        new_retryable_api_failures=new_retryable_api_failures,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _self_test() -> dict[str, Any]:
    rows = [
        {"question_id": "hotpotqa:paired", "question": "same", "data_source": "hotpotqa"},
        {"question_id": "hotpotqa:new", "question": "New question", "data_source": "hotpotqa"},
        {"question_id": "2wikimultihopqa:new", "question": "Another question", "data_source": "2wikimultihopqa"},
    ]
    excluded = {"hotpotqa:paired"}
    # Exercise the stable ID and question-key logic without reading a parquet
    # file, starting a retriever, or making an API call.
    assert "hotpotqa:paired" in excluded
    assert question_key(" New   Question ") == "new question"
    synthetic_pool = {
        "hotpotqa": [{"question_id": "hotpotqa:new", "data_source": "hotpotqa", "question": "New question"}],
        "2wikimultihopqa": [{"question_id": "2wikimultihopqa:new", "data_source": "2wikimultihopqa", "question": "Another question"}],
        "musique": [],
    }
    records = [
        {"candidate_id": "hotpotqa:new__sample_0", "question_id": "hotpotqa:new", "data_source": "hotpotqa", "sample_index": 0, "eligible": True},
        {"candidate_id": "hotpotqa:new__sample_1", "question_id": "hotpotqa:new", "data_source": "hotpotqa", "sample_index": 1, "eligible": True},
    ]
    assert len(select_accepted(records, synthetic_pool, {"hotpotqa": 1, "2wikimultihopqa": 0, "musique": 0})) == 1
    retryable_placeholder = {
        "candidate_id": "hotpotqa:new__sample_0",
        "question_id": "hotpotqa:new",
        "data_source": "hotpotqa",
        "sample_index": 0,
        "status": "failed",
        "raw_generation_turns": [],
        "failure_reasons": ["deepseek_api_failed", "DeepSeek HTTP 402: insufficient balance"],
        "eligible": False,
    }
    validate_existing_candidates([retryable_placeholder], synthetic_pool, set())
    assert is_retryable_api_failure(retryable_placeholder)
    assert not is_effective_candidate(retryable_placeholder)
    assert systemic_api_failure_reason(retryable_placeholder) == "http_402"
    assert has_unresolved_retryable_slots(
        [retryable_placeholder], synthetic_pool, {"hotpotqa": 1, "2wikimultihopqa": 0, "musique": 0}
    )
    generated_ineligible = {
        **retryable_placeholder,
        "raw_generation_turns": [{"turn_index": 0}],
        "failure_reasons": ["final_answer_em_failed"],
        "trajectory": "<think>x</think>",
    }
    validate_existing_candidates(
        [retryable_placeholder, generated_ineligible], synthetic_pool, set()
    )
    assert not is_retryable_api_failure(generated_ineligible)
    assert is_effective_candidate(generated_ineligible)
    assert not has_unresolved_retryable_slots(
        [retryable_placeholder, generated_ineligible],
        synthetic_pool,
        {"hotpotqa": 1, "2wikimultihopqa": 0, "musique": 0},
    )
    try:
        validate_existing_candidates(
            [generated_ineligible, {**generated_ineligible, "trajectory": "other"}],
            synthetic_pool,
            set(),
        )
    except ValueError as exc:
        assert "duplicate effective candidate_id" in str(exc)
    else:
        raise AssertionError("duplicate effective candidate IDs must be rejected")
    sentinel = "PRODUCTION_SECRET_SENTINEL"
    redacted = fixed._json_line({"authorization": f"Bearer {sentinel}"}, sentinel)
    assert sentinel not in redacted
    assert rows[0]["question_id"] in excluded
    return {
        "status": "passed",
        "checks": [
            "two_candidates_and_one_accepted_per_question",
            "exact_source_quota_cap",
            "retryable_api_failure_resume_slot",
            "systemic_api_failure_fail_fast_reason",
            "duplicate_effective_candidate_rejected",
            "paired_question_exclusion",
            "casefold_question_dedup_key",
            "secret_redaction",
            "no_api_or_retriever_call",
        ],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    p.add_argument("--exclude-summary", action="append", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-question", type=int, default=2)
    p.add_argument("--retrieval-url", default=DEFAULT_RETRIEVAL_URL)
    p.add_argument("--retrieval-topk", type=int, default=3)
    p.add_argument("--retrieval-timeout", type=float, default=90.0)
    p.add_argument("--max-searches", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-continuations", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--question-batch-size", type=int, default=DEFAULT_QUESTION_BATCH_SIZE)
    p.add_argument("--max-candidates-per-source", type=int, default=DEFAULT_MAX_CANDIDATES_PER_SOURCE)
    p.add_argument("--max-api-failures-per-run", type=int, default=DEFAULT_MAX_API_FAILURES_PER_RUN)
    p.add_argument("--api-timeout", type=float, default=180.0)
    p.add_argument("--retry-attempts", type=int, default=5)
    p.add_argument("--retry-backoff", type=float, default=1.0)
    p.add_argument("--retry-backoff-max", type=float, default=30.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return 0
    if args.exclude_summary is None:
        args.exclude_summary = list(DEFAULT_EXCLUDE_SUMMARIES)
    try:
        summary = run_production(args)
        if summary["status"] != "target_reached":
            print("WARNING: one or more multi-hop accepted quotas were not reached", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"PRODUCTION MULTIHOP FAILED: {fixed._safe_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
