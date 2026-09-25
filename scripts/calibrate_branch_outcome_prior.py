#!/usr/bin/env python3
"""Calibrate expansion priors from frozen random-selector rollout trees."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("**/*trees.jsonl"))
    if not files:
        raise FileNotFoundError(f"no *trees.jsonl files found under {path}")
    for file_path in files:
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {file_path}:{line_number}"
                    ) from exc


def bucket(node: dict[str, Any]) -> str:
    depth = int(node.get("depth", 0) or 0)
    if depth <= 0:
        return "root"
    if not bool(node.get("last_selection_after_search", False)):
        return "pre_search"
    if depth == 1:
        return "after_search_depth1"
    if depth == 2:
        return "after_search_depth2"
    return "after_search_depth3plus"


def calibrate(path: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"selected_parents": 0, "effective_parents": 0}
    )
    all_selected = {"selected_parents": 0, "effective_parents": 0}
    record_count = 0
    for record in iter_records(path):
        record_count += 1
        nodes = {node["node_uid"]: node for node in record.get("nodes", [])}
        selected_uids = dict.fromkeys(
            (record.get("expand_selection_stats") or {}).get("selected_node_uids", [])
        )
        for uid in selected_uids:
            node = nodes[uid]
            children = [
                child
                for child in nodes.values()
                if child.get("parent_node_uid") == uid
            ]
            effective = any(
                abs(float(child.get("edge_credit_normalized", 0.0) or 0.0)) > 1e-12
                for child in children
            )
            name = bucket(node)
            counts[name]["selected_parents"] += 1
            counts[name]["effective_parents"] += int(effective)
            all_selected["selected_parents"] += 1
            all_selected["effective_parents"] += int(effective)

    global_rate = (
        all_selected["effective_parents"] / all_selected["selected_parents"]
        if all_selected["selected_parents"]
        else 0.0
    )
    output: dict[str, Any] = {
        "source": str(path),
        "records": record_count,
        "global": {**all_selected, "effective_rate": global_rate},
        "buckets": {},
    }
    for name in (
        "root",
        "pre_search",
        "after_search_depth1",
        "after_search_depth2",
        "after_search_depth3plus",
    ):
        entry = counts[name]
        rate = (
            entry["effective_parents"] / entry["selected_parents"]
            if entry["selected_parents"]
            else global_rate
        )
        output["buckets"][name] = {**entry, "effective_rate": rate}
    output["recommended_config"] = {
        f"expand_outcome_prior_{name}": data["effective_rate"]
        for name, data in output["buckets"].items()
    }
    output["recommended_config"]["expand_outcome_prior_concentration_power"] = 2.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = calibrate(args.input)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
