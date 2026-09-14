#!/usr/bin/env python3
"""Prepare Search-R1 SFT conversations as deterministic train/validation Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


THINKING_OPEN = re.compile(r"<thinking\b", flags=re.IGNORECASE)
THINKING_CLOSE = re.compile(r"</thinking\s*>", flags=re.IGNORECASE)


def normalize_content(value: str) -> tuple[str, int]:
    """Return text with the Search-R1 thinking tag spelling normalized."""
    text = str(value).strip()
    replacements = len(THINKING_OPEN.findall(text)) + len(THINKING_CLOSE.findall(text))
    text = THINKING_OPEN.sub("<think", text)
    return THINKING_CLOSE.sub("</think>", text), replacements


def normalize_messages(messages: Any) -> tuple[list[dict[str, str]] | None, int]:
    if not isinstance(messages, list) or not messages:
        return None, 0

    normalized: list[dict[str, str]] = []
    replacements = 0
    for message in messages:
        if not isinstance(message, dict):
            return None, replacements
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str):
            return None, replacements
        normalized_content, count = normalize_content(content)
        replacements += count
        normalized.append({"role": role.strip(), "content": normalized_content})

    if normalized[-1]["role"] != "assistant" or not normalized[-1]["content"]:
        return None, replacements
    if not any(message["role"] == "assistant" and message["content"] for message in normalized):
        return None, replacements
    return normalized, replacements


def content_hash(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_records(local_json: Path | None, dataset_name: str | None) -> tuple[list[Any], str]:
    if local_json is not None and local_json.exists():
        with local_json.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {local_json}")
        return records, str(local_json)

    if not dataset_name:
        missing = f"{local_json}" if local_json is not None else "the local JSON input"
        raise FileNotFoundError(f"Local JSON not found ({missing}); provide --dataset-name for the HF fallback")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HF fallback requires the 'datasets' package") from exc

    dataset = load_dataset(dataset_name)
    if hasattr(dataset, "values"):
        split_records = []
        for split in dataset.values():
            split_records.extend(split)
        return split_records, f"hf:{dataset_name}"
    return list(dataset), f"hf:{dataset_name}"


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write structured messages when pyarrow is available, with a portable fallback."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "messages": [row["messages"] for row in rows],
                "content_hash": [row["content_hash"] for row in rows],
            }
        )
        pq.write_table(table, path)
        return
    except ImportError:
        pass

    try:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "messages": [json.dumps(row["messages"], ensure_ascii=False) for row in rows],
                "content_hash": [row["content_hash"] for row in rows],
            }
        )
        frame.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError("Writing Parquet requires pyarrow or pandas with a Parquet engine") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-json",
        type=Path,
        default=Path("data/search_r1_sft/raw/qwen2.5-7b-instruct-sft.json"),
        help="Local JSON array; if absent, --dataset-name is used.",
    )
    parser.add_argument("--dataset-name", help="Hugging Face dataset name used when --local-json is unavailable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_paths = [output_dir / name for name in ("train.parquet", "validation.parquet", "stats.json")]
    if output_dir.exists() and (any(path.exists() for path in output_paths) or any(output_dir.iterdir())) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace it")
    if args.overwrite and output_dir.exists():
        for path in output_paths:
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, source = load_records(args.local_json, args.dataset_name)
    stats: dict[str, Any] = {
        "source": source,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "requested_max_samples": args.max_samples,
        "input_samples": len(records),
        "skipped_malformed": 0,
        "skipped_after_max_samples": 0,
        "normalized_thinking_tags": 0,
    }

    valid_rows: list[dict[str, Any]] = []
    for record in records:
        messages, replacement_count = normalize_messages(
            record.get("messages") if isinstance(record, dict) else None)
        stats["normalized_thinking_tags"] += replacement_count
        if messages is None:
            stats["skipped_malformed"] += 1
            continue
        valid_rows.append({"messages": messages, "content_hash": content_hash(messages)})

    if args.max_samples is not None:
        if len(valid_rows) > args.max_samples:
            stats["skipped_after_max_samples"] = len(valid_rows) - args.max_samples
            valid_rows = valid_rows[: args.max_samples]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        groups.setdefault(row["content_hash"], []).append(row)

    group_keys = list(groups)
    random.Random(args.seed).shuffle(group_keys)
    target_validation = int(len(valid_rows) * args.val_ratio)
    if args.val_ratio > 0 and len(valid_rows) > 1:
        target_validation = max(1, target_validation)
    validation_keys: set[str] = set()
    validation_count = 0
    if target_validation and len(group_keys) > 1:
        for key in group_keys:
            if validation_count + len(groups[key]) <= target_validation or not validation_keys:
                validation_keys.add(key)
                validation_count += len(groups[key])
            if validation_count >= target_validation:
                break
        if validation_count == len(valid_rows):
            validation_keys.remove(group_keys[0])

    validation = [row for key in group_keys if key in validation_keys for row in groups[key]]
    train = [row for key in group_keys if key not in validation_keys for row in groups[key]]
    write_parquet(output_paths[0], train)
    write_parquet(output_paths[1], validation)

    stats.update(
        {
            "valid_samples": len(valid_rows),
            "unique_content_groups": len(groups),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "train_groups": len({row["content_hash"] for row in train}),
            "validation_groups": len({row["content_hash"] for row in validation}),
            "duplicate_samples": len(valid_rows) - len(groups),
            "role_counts": {
                role: sum(
                    message["role"] == role
                    for row in valid_rows
                    for message in row["messages"]
                )
                for role in sorted({
                    message["role"]
                    for row in valid_rows
                    for message in row["messages"]
                })
            },
        }
    )
    with output_paths[2].open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
