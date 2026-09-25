#!/usr/bin/env python3
"""Audit search trajectories against dataset-provided supporting evidence.

This is deliberately an evaluation-only tool.  It joins a rollout uid such as
``hotpotqa:37324`` back to the raw JSONL row, then measures title recall for
retrieved documents.  MuSiQue additionally exposes decomposition questions,
so the script reports a lightweight lexical query-similarity diagnostic.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL | re.IGNORECASE)
PAIR_RE = re.compile(
    r"<search>\s*(.*?)\s*</search>.*?<information>(.*?)</information>",
    re.DOTALL | re.IGNORECASE,
)
TITLE_RE = re.compile(r'Doc\s+\d+\s*\(Title:\s*["“]?(.*?)["”]?\)\s*', re.IGNORECASE)
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


DEFAULT_RAW = {
    "hotpotqa": "data/flashrag_multihop_raw/hotpotqa/train.jsonl",
    "2wikimultihopqa": "data/flashrag_multihop_raw/2wikimultihopqa/train.jsonl",
    "musique": "data/flashrag_multihop_raw/musique/train.jsonl",
    "bamboogle": "data/flashrag_multihop_raw/bamboogle/test.jsonl",
}


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def jaccard(left: str, right: str) -> float:
    a, b = set(TOKEN_RE.findall(left.casefold())), set(TOKEN_RE.findall(right.casefold()))
    return len(a & b) / len(a | b) if a and b else 0.0


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_fields(row: dict[str, Any], source: str) -> tuple[list[str], list[str]]:
    metadata = row.get("metadata") or {}
    if source in {"hotpotqa", "2wikimultihopqa"}:
        supporting_facts = metadata.get("supporting_facts", {})
        if isinstance(supporting_facts, dict):
            titles = supporting_facts.get("title", [])
        else:
            titles = [fact[0] for fact in supporting_facts if fact]
        return list(dict.fromkeys(titles)), []
    if source == "musique":
        decomposition = metadata.get("question_decomposition", [])
        titles = [
            item.get("support_paragraph", {}).get("title", "")
            for item in decomposition
        ]
        questions = [item.get("question", "") for item in decomposition]
        return [title for title in dict.fromkeys(titles) if title], [q for q in questions if q]
    return [], []


def rollout_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(Path(item) for item in glob.glob(pattern))
    return sorted(paths)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", nargs="+", help="Selected-rollout JSONL paths or globs")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--raw",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Override a raw JSONL source (repeatable)",
    )
    args = parser.parse_args()

    raw_paths = dict(DEFAULT_RAW)
    for spec in args.raw:
        source, separator, path = spec.partition("=")
        if not separator:
            parser.error(f"invalid --raw value: {spec!r}")
        raw_paths[source] = path

    paths = rollout_paths(args.rollouts)
    if not paths:
        parser.error("no rollout files matched")

    raw_cache: dict[str, list[dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    title_recalls: list[float] = []
    next_hop_hits: list[float] = []
    musique_query_scores: list[float] = []
    seen: set[tuple[str, str]] = set()

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = (str(record.get("tree_uid")), str(record.get("node_uid")))
                if key in seen:
                    continue
                seen.add(key)
                counters["trajectories"] += 1
                source = str(record.get("data_source", ""))
                uid = str(record.get("uid", ""))
                try:
                    index = int(uid.rsplit(":", 1)[1])
                except (IndexError, ValueError):
                    counters["invalid_uid"] += 1
                    continue
                if source not in raw_paths:
                    counters["unsupported_source"] += 1
                    continue
                if source not in raw_cache:
                    raw_cache[source] = load_rows(Path(raw_paths[source]))
                if index >= len(raw_cache[source]):
                    counters["index_out_of_range"] += 1
                    continue

                gold_titles, gold_queries = gold_fields(raw_cache[source][index], source)
                if not gold_titles:
                    counters["without_gold_evidence"] += 1
                    per_source[source]["without_gold_evidence"] += 1
                    continue

                counters["with_gold_evidence"] += 1
                per_source[source]["with_gold_evidence"] += 1
                normalized_gold = {normalize(title) for title in gold_titles}
                response = str(record.get("response", ""))
                pairs = PAIR_RE.findall(response)
                retrieved_so_far: set[str] = set()
                trajectory_retrieved: set[str] = set()
                for query, information in pairs:
                    retrieved = {
                        normalize(title.strip().strip('"“”'))
                        for title in TITLE_RE.findall(information)
                    }
                    remaining_gold = normalized_gold - retrieved_so_far
                    if remaining_gold:
                        next_hop_hits.append(float(bool(retrieved & remaining_gold)))
                        counters["scored_query_events"] += 1
                    retrieved_so_far.update(retrieved)
                    trajectory_retrieved.update(retrieved)
                    if gold_queries:
                        musique_query_scores.append(max(jaccard(query, target) for target in gold_queries))

                recall = len(trajectory_retrieved & normalized_gold) / len(normalized_gold)
                title_recalls.append(recall)
                per_source[source]["any_gold_title_hit"] += int(recall > 0)
                per_source[source]["all_gold_titles_hit"] += int(recall == 1)

    summary = {
        "files": [str(path) for path in paths],
        "counts": dict(counters),
        "metrics": {
            "trajectory_gold_title_recall": mean(title_recalls),
            "trajectory_any_gold_title_rate": mean([float(value > 0) for value in title_recalls]),
            "query_event_new_gold_title_hit_rate": mean(next_hop_hits),
            "musique_query_decomposition_lexical_jaccard": mean(musique_query_scores),
        },
        "per_source_counts": {key: dict(value) for key, value in sorted(per_source.items())},
        "notes": {
            "title_metrics": "Exact normalized title matching against dataset supporting evidence.",
            "next_hop": "Previously retrieved gold titles are removed before scoring each later query.",
            "query_similarity": "Lexical Jaccard is auxiliary; retrieval title recall is the primary diagnostic.",
            "training_use": "Evaluation only; no golden evidence is injected into OPD training.",
        },
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
