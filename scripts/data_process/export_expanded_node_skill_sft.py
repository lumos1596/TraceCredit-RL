#!/usr/bin/env python3
"""Build a UID-disjoint expanded Analyzer/skill-consumer SFT mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"<think>.+?</think>\n<search>[^\n<>]+</search>", re.DOTALL)


def read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def heldout_uid(uid: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def valid_consumer(row: dict[str, Any]) -> bool:
    response = str(row.get("response", ""))
    return (
        float(row.get("value_gap", 0.0)) >= 0.1
        and "[redacted]" not in response.casefold()
        and ACTION_RE.fullmatch(response) is not None
        and isinstance(row.get("node_skill"), dict)
    )


def plain_replay(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"])
    prompt = re.sub(r"\n<analyzer_skill>\n.*?\n</analyzer_skill>\n", "\n", prompt, flags=re.DOTALL)
    prompt = re.sub(
        r"Use this answer-free procedural skill.*?do not state a final answer\.\n?",
        "Choose the next action from the current visible state. Return exactly "
        "<think>...</think><search>...</search>; do not state a final answer.\n",
        prompt,
        flags=re.DOTALL,
    )
    result = dict(row)
    result.update(prompt=prompt, task_type="action_replay", source_event_id=row["event_id"])
    result.pop("node_skill", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer-train", required=True)
    parser.add_argument("--analyzer-validation", required=True)
    parser.add_argument("--consumer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--consumer-train", type=int, default=600)
    parser.add_argument("--plain-train", type=int, default=107)
    parser.add_argument("--heldout-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    analyzer_train = read(args.analyzer_train)
    analyzer_val = read(args.analyzer_validation)
    analyzer_train_uids = {str(row["uid"]) for row in analyzer_train}
    analyzer_val_uids = {str(row["uid"]) for row in analyzer_val}
    if analyzer_train_uids & analyzer_val_uids:
        raise ValueError("Analyzer split has UID leakage")

    def is_val(uid: str) -> bool:
        if uid in analyzer_val_uids:
            return True
        if uid in analyzer_train_uids:
            return False
        return heldout_uid(uid, args.seed, args.heldout_fraction)

    consumers = [row for row in read(args.consumer) if valid_consumer(row)]
    # Defensive event-level deduplication after API generation.
    consumers = list({str(row["event_id"]): row for row in consumers}.values())
    consumer_train_pool = [row for row in consumers if not is_val(str(row["uid"]))]
    consumer_val = [row for row in consumers if is_val(str(row["uid"]))]
    rng.shuffle(consumer_train_pool)
    if len(consumer_train_pool) < args.consumer_train:
        raise ValueError(
            f"need {args.consumer_train} train consumer events, "
            f"found {len(consumer_train_pool)}"
        )

    consumer_train = consumer_train_pool[: args.consumer_train]
    # Paired skill/no-skill views explicitly teach conditional skill use while
    # retaining a sizeable UID-disjoint validation set.  These are distinct
    # inputs, not repeated examples.
    plain_sources = consumer_train[: args.plain_train]
    plain_train = [plain_replay(row) for row in plain_sources]
    plain_val = [plain_replay(row) for row in consumer_val[: min(40, len(consumer_val))]]
    for row in analyzer_train + analyzer_val:
        row["task_type"] = "analyzer_generation"

    train = analyzer_train + consumer_train + plain_train
    validation = analyzer_val + consumer_val + plain_val
    rng.shuffle(train)
    rng.shuffle(validation)
    train_uids = {str(row["uid"]) for row in train}
    val_uids = {str(row["uid"]) for row in validation}
    if train_uids & val_uids:
        raise ValueError(f"UID leakage: {sorted(train_uids & val_uids)[:10]}")
    if any("[redacted]" in str(row.get("response", "")).casefold() for row in train + validation):
        raise ValueError("redaction artifact in a training target")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "train.jsonl", train)
    write(output / "validation.jsonl", validation)
    manifest = {
        "seed": args.seed,
        "source_consumer": args.consumer,
        "source_analyzer_train": args.analyzer_train,
        "source_analyzer_validation": args.analyzer_validation,
        "accepted_consumer_total": len(consumers),
        "consumer_train_pool": len(consumer_train_pool),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_uids": len(train_uids),
        "validation_uids": len(val_uids),
        "uid_overlap": 0,
        "train_task_counts": dict(Counter(str(row["task_type"]) for row in train)),
        "validation_task_counts": dict(Counter(str(row["task_type"]) for row in validation)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
