#!/usr/bin/env python3
"""Compare branch-credit signal efficiency from Tree-GRPO rollout dumps.

The primary metric is parent-level: among parents with at least two retained
children, how many have non-zero correctness-based branch credit?  Edge-level
coverage and selection diagnostics are reported as secondary metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Totals:
    records: int = 0
    parents: int = 0
    zero_variance_parents: int = 0
    sibling_edges: int = 0
    nonzero_sibling_edges: int = 0
    selections: int = 0
    selection_duplicates: int = 0
    selected_depth_sum: float = 0.0
    selected_after_search_sum: float = 0.0
    selected_uncertainty_sum: float = 0.0
    selected_outcome_prior_sum: float = 0.0
    selection_stats_records: int = 0
    files: list[str] = field(default_factory=list)

    def add_record(self, record: dict[str, Any]) -> None:
        stats = record.get("branch_credit_stats") or {}
        parents = int(stats.get("parents_with_multiple_children", 0) or 0)
        zero = int(stats.get("zero_variance_parents", 0) or 0)
        sibling_edges = int(stats.get("edges_with_siblings", 0) or 0)

        nodes = record.get("nodes") or []
        nonzero_edges = sum(
            1
            for node in nodes
            if node.get("parent_node_uid") is not None
            and len(_children_of(nodes, node.get("parent_node_uid"))) > 1
            and _finite_nonzero(node.get("edge_credit_normalized", 0.0))
        )

        self.records += 1
        self.parents += parents
        self.zero_variance_parents += min(zero, parents)
        self.sibling_edges += sibling_edges
        self.nonzero_sibling_edges += nonzero_edges

        selection = record.get("expand_selection_stats") or {}
        count = int(selection.get("selected_count", 0) or 0)
        if count:
            self.selections += count
            self.selection_duplicates += int(selection.get("duplicate_count", 0) or 0)
            self.selected_depth_sum += float(selection.get("mean_selected_depth", 0.0)) * count
            self.selected_after_search_sum += float(
                selection.get("fraction_selected_after_search", 0.0)
            ) * count
            self.selected_uncertainty_sum += float(
                selection.get("mean_selection_uncertainty", 0.0)
            ) * count
            self.selected_outcome_prior_sum += float(
                selection.get("mean_selection_outcome_prior", 0.0)
            ) * count
            self.selection_stats_records += 1

    def as_dict(self) -> dict[str, Any]:
        effective_parents = self.parents - self.zero_variance_parents
        return {
            "records": self.records,
            "parents_with_multiple_children": self.parents,
            "effective_branching_parents": effective_parents,
            "effective_parent_rate": _ratio(effective_parents, self.parents),
            "zero_variance_parent_rate": _ratio(self.zero_variance_parents, self.parents),
            "sibling_edges": self.sibling_edges,
            "nonzero_sibling_edges": self.nonzero_sibling_edges,
            "nonzero_sibling_edge_rate": _ratio(
                self.nonzero_sibling_edges, self.sibling_edges
            ),
            "selected_count": self.selections,
            "selection_duplicate_rate": _ratio(
                self.selection_duplicates, self.selections
            ),
            "mean_selected_depth": _ratio(self.selected_depth_sum, self.selections),
            "fraction_selected_after_search": _ratio(
                self.selected_after_search_sum, self.selections
            ),
            "mean_selection_uncertainty": _ratio(
                self.selected_uncertainty_sum, self.selections
            ),
            "mean_selection_outcome_prior": _ratio(
                self.selected_outcome_prior_sum, self.selections
            ),
            "selection_stats_records": self.selection_stats_records,
            "file_count": len(self.files),
            "first_file": self.files[0] if self.files else None,
            "last_file": self.files[-1] if self.files else None,
        }


def _children_of(nodes: list[dict[str, Any]], parent_uid: Any) -> list[dict[str, Any]]:
    return [node for node in nodes if node.get("parent_node_uid") == parent_uid]


def _finite_nonzero(value: Any, eps: float = 1e-12) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and abs(number) > eps


def _ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def iter_jsonl_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(path.glob("**/*trees.jsonl"))


def summarize(path: Path) -> dict[str, Any]:
    totals = Totals()
    for jsonl_path in iter_jsonl_files(path):
        totals.files.append(str(jsonl_path))
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {jsonl_path}:{line_number}") from exc
                totals.add_record(record)
    if not totals.files:
        raise FileNotFoundError(f"no *trees.jsonl files found under {path}")
    return totals.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=Path, required=True, help="Random-mode rollout path")
    parser.add_argument("--new", type=Path, help="New-mode rollout path")
    parser.add_argument(
        "--new-label",
        default="uncertainty_balanced",
        help="JSON label for --new (default: uncertainty_balanced)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    result: dict[str, Any] = {"random": summarize(args.random)}
    if args.new:
        if args.new_label == "random":
            raise ValueError("--new-label must not overwrite the random baseline")
        result[args.new_label] = summarize(args.new)
        old = result["random"]
        new = result[args.new_label]

        def absolute_change(metric: str) -> float | None:
            old_value = old[metric]
            new_value = new[metric]
            if old_value is None or new_value is None:
                return None
            return new_value - old_value

        effective_change = absolute_change("effective_parent_rate")
        nonzero_edge_change = absolute_change("nonzero_sibling_edge_rate")
        zero_variance_change = absolute_change("zero_variance_parent_rate")
        result["comparison"] = {
            "effective_parent_rate_absolute_change": effective_change,
            "nonzero_sibling_edge_rate_absolute_change": nonzero_edge_change,
            "zero_variance_parent_rate_absolute_change": zero_variance_change,
            "effective_parent_rate_improved": (
                effective_change > 0 if effective_change is not None else None
            ),
            "nonzero_sibling_edge_rate_improved": (
                nonzero_edge_change > 0 if nonzero_edge_change is not None else None
            ),
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
