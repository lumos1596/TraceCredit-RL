#!/usr/bin/env python3
"""Export strict node-skill pilot mixed with Analyzer-generation SFT data."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strict_consumer(row: dict[str, Any]) -> bool:
    return (
        float(row.get("value_gap", 0.0)) >= .25
        and "[redacted]" not in str(row.get("response", "")).casefold()
        and re.fullmatch(r"<think>.+?</think>\n<search>[^\n<>]+</search>", str(row.get("response", "")), re.DOTALL) is not None
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analyzer-train", required=True); p.add_argument("--analyzer-validation", required=True)
    p.add_argument("--consumer", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--consumer-repeat", type=int, default=16); p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args(); rng = random.Random(args.seed)
    analyzer_train, analyzer_val = read(args.analyzer_train), read(args.analyzer_validation)
    train_uids, val_uids = {r["uid"] for r in analyzer_train}, {r["uid"] for r in analyzer_val}
    if train_uids & val_uids: raise ValueError("Analyzer split has UID leakage")
    consumers = [row for row in read(args.consumer) if strict_consumer(row)]
    consumer_train = [row for row in consumers if row["uid"] in train_uids]
    consumer_val = [row for row in consumers if row["uid"] in val_uids]
    unknown = [row["uid"] for row in consumers if row["uid"] not in train_uids | val_uids]
    if unknown: raise ValueError(f"consumer UIDs absent from Analyzer split: {sorted(set(unknown))}")
    for row in analyzer_train + analyzer_val: row["task_type"] = "analyzer_generation"
    mixed_train = analyzer_train + [dict(row, repeat_index=i) for i in range(args.consumer_repeat) for row in consumer_train]
    mixed_val = analyzer_val + consumer_val
    rng.shuffle(mixed_train); rng.shuffle(mixed_val)
    if {r["uid"] for r in mixed_train} & {r["uid"] for r in mixed_val}: raise ValueError("mixed split UID leakage")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write(output / "train.jsonl", mixed_train); write(output / "validation.jsonl", mixed_val)
    manifest = {
        "analyzer_train": len(analyzer_train), "analyzer_validation": len(analyzer_val),
        "strict_consumer_total": len(consumers), "consumer_train_unique": len(consumer_train),
        "consumer_validation_unique": len(consumer_val), "consumer_repeat": args.consumer_repeat,
        "mixed_train": len(mixed_train), "mixed_validation": len(mixed_val),
        "train_task_counts": dict(Counter(r["task_type"] for r in mixed_train)),
        "validation_task_counts": dict(Counter(r["task_type"] for r in mixed_val)),
        "train_uids": len({r["uid"] for r in mixed_train}), "validation_uids": len({r["uid"] for r in mixed_val}),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
