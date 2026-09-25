#!/usr/bin/env python3
"""Question-grouped train/validation export for audited Tree-SEED Analyzer SFT."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("accepted input is empty")
    # `raw_response` is retained in the audited provenance file so that a
    # reviewer can reproduce the validator's decision.  It is deliberately
    # not carried into `to_sft_row`; only actual answer-bearing source fields
    # make the audited input itself unsafe.
    forbidden = {"ground_truth", "answer", "answers"}
    for row in rows:
        if not row.get("accepted"):
            raise ValueError(f"non-accepted record found: {row.get('sample_id')}")
        if row.get("rejection_reasons"):
            raise ValueError(f"filtered record has rejection reason: {row.get('sample_id')}")
        if forbidden & set(row):
            raise ValueError(f"sensitive source field present: {row.get('sample_id')}")
        parsed = json.loads(row["response"])
        if set(parsed) != {"episode_summary", "episode_skill"}:
            raise ValueError(f"invalid response schema: {row.get('sample_id')}")


def to_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "response": row["response"],
        "sample_id": row["sample_id"],
        "uid": row["uid"],
        "data_source": row["data_source"],
        "source_score": row["source_score"],
        "source_outcome": row["source_outcome"],
        "model": row["model"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")

    rows = read_jsonl(Path(args.accepted_jsonl))
    validate(rows)
    uids = sorted({str(row["uid"]) for row in rows})
    rng = random.Random(args.seed)
    rng.shuffle(uids)
    val_count = max(1, min(len(uids) - 1, round(len(uids) * args.val_ratio)))
    val_uids = set(uids[:val_count])
    train_rows = [to_sft_row(row) for row in rows if str(row["uid"]) not in val_uids]
    val_rows = [to_sft_row(row) for row in rows if str(row["uid"]) in val_uids]
    if {row["uid"] for row in train_rows} & {row["uid"] for row in val_rows}:
        raise AssertionError("question leakage across split")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "validation.jsonl", val_rows)
    try:
        import pandas as pd

        pd.DataFrame(train_rows).to_parquet(output / "train.parquet", index=False)
        pd.DataFrame(val_rows).to_parquet(output / "validation.parquet", index=False)
    except Exception as exc:
        raise RuntimeError("parquet export failed") from exc

    manifest = {
        "source_accepted_jsonl": str(Path(args.accepted_jsonl)),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "total_questions": len(uids),
        "train_questions": len({row["uid"] for row in train_rows}),
        "validation_questions": len({row["uid"] for row in val_rows}),
        "source_counts": dict(Counter(row["data_source"] for row in rows)),
        "outcome_counts": dict(Counter(row["source_outcome"] for row in rows)),
        "leakage_policy": "Only independently revalidated accepted rows are exported; train/validation split is by uid.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
