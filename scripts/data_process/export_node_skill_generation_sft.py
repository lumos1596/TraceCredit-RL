#!/usr/bin/env python3
"""Export direct, answer-free node-skill generation SFT pairs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_node_skill_consumption_sft import teacher_prompt


FIELDS = {"failure_type", "missing_relation", "next_operation", "stop_condition"}


def read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", required=True); parser.add_argument("--prepared", required=True)
    parser.add_argument("--train-split", required=True); parser.add_argument("--validation-split", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    consumers = {row["event_id"]: row for row in read(args.consumer)}
    prepared = {row["event_id"]: row for row in read(args.prepared)}
    train_uids = {row["uid"] for row in read(args.train_split)}
    val_uids = {row["uid"] for row in read(args.validation_split)}
    if train_uids & val_uids:
        raise ValueError("train/validation UID leakage")
    train, validation = [], []
    for event_id, row in sorted(consumers.items()):
        if float(row.get("value_gap", 0)) < .1 or "[redacted]" in str(row.get("response", "")).casefold():
            continue
        skill = row.get("node_skill")
        if not isinstance(skill, dict) or set(skill) != FIELDS:
            continue
        source = prepared.get(event_id)
        if source is None:
            continue
        event = dict(source)
        event["better_titles"] = []
        event["aliases"] = []
        item = {
            "prompt": teacher_prompt(event),
            "response": json.dumps({key: str(skill[key]).strip() for key in
                                     ("failure_type", "missing_relation", "next_operation", "stop_condition")},
                                    ensure_ascii=False, separators=(",", ":")),
            "task_type": "node_skill_generation", "event_id": event_id,
            "uid": row["uid"], "tree_uid": row.get("tree_uid"),
            "parent_uid": row.get("parent_uid"), "data_source": row.get("data_source"),
            "value_gap": row.get("value_gap"), "teacher_model": row.get("model", "deepseek-chat"),
        }
        if row["uid"] in train_uids:
            train.append(item)
        elif row["uid"] in val_uids:
            validation.append(item)
    if not train or not validation:
        raise ValueError(f"empty split: train={len(train)}, validation={len(validation)}")
    train_ids, val_ids = {row["uid"] for row in train}, {row["uid"] for row in validation}
    if train_ids & val_ids:
        raise ValueError("exported split has UID leakage")
    if any("[redacted]" in row["response"].casefold() for row in train + validation):
        raise ValueError("answer-redaction artifact in target")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write(output / "train.jsonl", train); write(output / "validation.jsonl", validation)
    manifest = {
        "source_consumer": args.consumer, "source_prepared": args.prepared,
        "train_examples": len(train), "validation_examples": len(validation),
        "train_uids": len(train_ids), "validation_uids": len(val_ids), "uid_overlap": 0,
        "task_counts": dict(Counter(row["task_type"] for row in train + validation)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
