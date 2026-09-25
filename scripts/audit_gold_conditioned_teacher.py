#!/usr/bin/env python3
"""Four-way offline audit of a gold-conditioned teacher for Self-OPD.

For every sibling-group event that the in-training OPD builder would have
distilled, the target query is scored under five contexts that differ only in
a small "cheat sheet" inserted in native ``<information>`` format right
before the query:

* ``student``   no cheat sheet (pure policy baseline);
* ``gold``      dataset-provided supporting titles (+ MuSiQue decomposition);
* ``shuffle``   supporting titles of a different question (content control);
* ``format``    same template, no information (format/length control);
* ``answer``    the golden final answer (copy-leakage diagnostic).

Gate logic: the gold-conditioned teacher carries real information only if
``gold`` beats both ``shuffle`` and ``format`` (paired event-level bootstrap
CI of the mean lift difference excludes zero).  Everything runs on dumped
rollout trees; no training or retrieval service is touched.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers

SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
INFO_AFTER_SEARCH_RE = re.compile(
    r"<search>\s*.*?\s*</search>\s*\n?<information>(.*?)</information>",
    re.DOTALL,
)
TITLE_RE = re.compile(r'Doc\s+\d+\s*\(Title:\s*["“]?(.*?)["”]?\)\s*', re.IGNORECASE)
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

DEFAULT_RAW = {
    "hotpotqa": "data/flashrag_multihop_raw/hotpotqa/train.jsonl",
    "2wikimultihopqa": "data/flashrag_multihop_raw/2wikimultihopqa/train.jsonl",
    "musique": "data/flashrag_multihop_raw/musique/train.jsonl",
    "bamboogle": "data/flashrag_multihop_raw/bamboogle/test.jsonl",
}
ARMS = ("student", "gold", "shuffle", "format", "answer")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        return [t for t in dict.fromkeys(titles) if t], [q for q in questions if q]
    return [], []


def render_cheat(
    kind: str,
    gold_titles: list[str],
    decomposition: list[str],
    answer: str,
    shuffle_titles: list[str],
) -> str:
    if kind == "gold":
        parts = []
        if gold_titles:
            parts.append(
                "This question is supported by the documents: "
                + "; ".join(f'"{t}"' for t in gold_titles)
            )
        if decomposition:
            parts.append(
                "Sub-questions to answer: "
                + " ".join(f"({i + 1}) {q}" for i, q in enumerate(decomposition))
            )
        content = " ".join(parts)
    elif kind == "shuffle":
        content = (
            "This question is supported by the documents: "
            + "; ".join(f'"{t}"' for t in shuffle_titles)
        )
    elif kind == "answer":
        content = f'The correct final answer is "{answer}"'
    elif kind == "format":
        content = "No reference information is available"
    else:
        raise ValueError(kind)
    return "\n<information>(reference) " + content + "\n</information>\n"


def extract_edge_query(edge_text: str) -> str | None:
    match = SEARCH_RE.search(edge_text)
    return match.group(1).strip() if match else None


def extract_edge_evidence_titles(edge_text: str) -> set[str]:
    match = re.search(r"<information>(.*?)</information>", edge_text, re.DOTALL)
    if not match:
        return set()
    return {
        normalize(title.strip().strip('"“”'))
        for title in TITLE_RE.findall(match.group(1))
    }


def bootstrap_ci(values: list[float], samples: int = 10000, seed: int = 0) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * samples)]
    hi = means[min(samples - 1, int(0.975 * samples))]
    return [lo, hi]


class EventStore:
    """Sibling-group events rebuilt from dumped trees, mirroring self_opd.py."""

    def __init__(self, min_value_gap: float, min_raw_advantage: float,
                 max_query_tokens: int):
        self.min_value_gap = min_value_gap
        self.min_raw_advantage = min_raw_advantage
        self.max_query_tokens = max_query_tokens
        self.events: list[dict[str, Any]] = []
        self.counters: dict[str, int] = defaultdict(int)

    def consider_tree(self, record: dict[str, Any], tokenizer) -> None:
        nodes = {node["node_uid"]: node for node in record["nodes"]}
        parents_seen: set[str] = set()
        for node in record["nodes"]:
            parent_uid = node.get("parent_node_uid")
            if parent_uid is None or parent_uid in parents_seen:
                continue
            parents_seen.add(parent_uid)
            parent = nodes[parent_uid]
            siblings = []
            for child_uid in parent.get("child_node_uids", []):
                child = nodes.get(child_uid)
                if child is None:
                    continue
                parent_text = parent.get("response") or ""
                child_text = child.get("response") or ""
                if child_text.startswith(parent_text):
                    edge = child_text[len(parent_text):]
                else:
                    self.counters["edge_prefix_mismatch"] += 1
                    continue
                query = extract_edge_query(edge)
                if query is None:
                    continue
                if len(tokenizer.encode(query, add_special_tokens=False)) > self.max_query_tokens:
                    continue
                siblings.append((child, query, edge))
            if len(siblings) < 2:
                self.counters["groups_below_two_queries"] += 1
                continue
            values = [float(child.get("node_value", 0.0)) for child, _, _ in siblings]
            value_gap = max(values) - min(values)
            if not math.isfinite(value_gap) or value_gap < self.min_value_gap:
                self.counters["groups_below_gap"] += 1
                continue
            top_index = max(range(len(siblings)), key=lambda i: values[i])
            target_child, target_query, target_edge = siblings[top_index]
            total_other = sum(
                max(1, int(siblings[i][0].get("descendant_leaf_count", 1)))
                for i in range(len(siblings)) if i != top_index
            )
            if total_other <= 0:
                continue
            baseline = sum(
                max(1, int(siblings[i][0].get("descendant_leaf_count", 1))) * values[i]
                for i in range(len(siblings)) if i != top_index
            ) / total_other
            raw_advantage = values[top_index] - baseline
            if not math.isfinite(raw_advantage) or raw_advantage <= self.min_raw_advantage:
                self.counters["groups_below_advantage"] += 1
                continue
            self.events.append({
                "tree_uid": record["tree_uid"],
                "parent_uid": parent_uid,
                "target_node_uid": target_child["node_uid"],
                "prefix_text": parent.get("response") or "",
                "target_query": target_query,
                "target_evidence_titles": extract_edge_evidence_titles(target_edge),
                "sibling_queries": [
                    q for i, (_, q, _) in enumerate(siblings) if i != top_index
                ],
                "target_value": values[top_index],
                "value_gap": value_gap,
                "raw_advantage": raw_advantage,
                "sibling_count": len(siblings),
                "depth": target_child.get("depth", -1),
            })


def build_contexts(event: dict[str, Any], prompt_text: str, cheats: dict[str, str]) -> dict[str, str]:
    prefix = event["prefix_text"]
    contexts = {}
    for arm in ARMS:
        insert = cheats.get(arm, "")
        contexts[arm] = prompt_text + prefix + insert + "<search>" + event["target_query"]
    return contexts


@torch.no_grad()
def score_arm(
    contexts: list[str],
    query_texts: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_context: int,
) -> tuple[list[float], list[list[float]], list[bool]]:
    """Return (mean logprob per query token, per-token logprobs, truncated)."""
    items = []
    truncated = []
    for index, (context, query) in enumerate(zip(contexts, query_texts)):
        # Tokenize prefix and query separately and concatenate.  The naive
        # character cut fails whenever the BPE merges ">" of "<search>" with
        # the query's first letter (tokens like ">T"), which dropped half the
        # events; separate tokenization keeps the query ids identical across
        # arms, so paired comparisons stay fair.
        prefix_ids = tokenizer.encode(
            context[: len(context) - len(query)], add_special_tokens=False
        )
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        if not query_ids:
            items.append(None)
            truncated.append(False)
            continue
        full_ids = prefix_ids + query_ids
        is_truncated = len(full_ids) > max_context
        truncated.append(is_truncated)
        items.append((full_ids, len(prefix_ids)))

    order = sorted(
        (i for i, item in enumerate(items) if item is not None),
        key=lambda i: len(items[i][0]),
    )
    means: list[float | None] = [None] * len(contexts)
    token_logprobs: list[list[float] | None] = [None] * len(contexts)
    for start in range(0, len(order), batch_size):
        batch_index = order[start:start + batch_size]
        batch = [items[i] for i in batch_index]
        max_len = max(len(ids) for ids, _ in batch)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, max_len - len(ids):] = torch.tensor(ids)
            attention[row, max_len - len(ids):] = 1
        position_ids = (attention.cumsum(-1) - 1).clamp_min(0)
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
            position_ids=position_ids.to(device),
            use_cache=False,
        ).logits
        for row, index in enumerate(batch_index):
            ids, prefix_len = items[index]
            query_len = len(ids) - prefix_len
            positions = torch.arange(max_len - query_len - 1, max_len - 1)
            targets = input_ids[row][positions + 1].to(device).unsqueeze(-1)
            token_lp = (
                logits[row][positions.to(device)]
                .float()
                .log_softmax(-1)
                .gather(-1, targets)
                .squeeze(-1)
                .cpu()
                .tolist()
            )
            token_logprobs[index] = token_lp
            means[index] = sum(token_lp) / max(1, len(token_lp))
    return means, token_logprobs, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True,
                        help="trees.jsonl paths or globs")
    parser.add_argument("--selected", nargs="+", default=[],
                        help="selected.jsonl paths or globs (tree_uid -> uid link)")
    parser.add_argument("--model", required=True, help="HF-format model dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--min-value-gap", type=float, default=0.25)
    parser.add_argument("--min-raw-advantage", type=float, default=0.0)
    parser.add_argument("--max-query-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tree_paths = sorted(p for pattern in args.trees for p in glob.glob(pattern))
    if not tree_paths:
        parser.error("no trees matched")
    selected_paths = sorted(p for pattern in args.selected for p in glob.glob(pattern))

    tree_to_uid: dict[str, tuple[str, str]] = {}
    for path in selected_paths:
        for record in load_jsonl(Path(path)):
            if record.get("tree_uid") and record.get("uid"):
                tree_to_uid[record["tree_uid"]] = (
                    str(record.get("data_source", "")),
                    str(record["uid"]),
                )

    raw_paths = dict(DEFAULT_RAW)
    for spec in args.raw:
        source, separator, path = spec.partition("=")
        if not separator:
            parser.error(f"invalid --raw value: {spec!r}")
        raw_paths[source] = path
    raw_cache: dict[str, list[dict[str, Any]]] = {}

    import pandas as pd
    parquet = pd.read_parquet("data/multihopqa_search_mixed_402020_20260830/train.parquet")
    index_to_row: dict[str, Any] = {}
    for _, row in parquet.iterrows():
        info = row.get("extra_info") or {}
        key = info.get("index")
        if key:
            index_to_row[str(key)] = row

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    store = EventStore(
        min_value_gap=args.min_value_gap,
        min_raw_advantage=args.min_raw_advantage,
        max_query_tokens=args.max_query_tokens,
    )
    for path in tree_paths:
        for record in load_jsonl(Path(path)):
            store.consider_tree(record, tokenizer)

    rng = random.Random(args.seed)
    audit_events = []
    counters = dict(store.counters)
    counters["trees_scanned"] = len(tree_paths)
    counters["sibling_events"] = len(store.events)

    for event in store.events:
        source, uid = tree_to_uid.get(event["tree_uid"], ("", ""))
        if not source or ":" not in uid:
            counters["missing_uid_link"] += 1
            continue
        try:
            row_index = int(uid.rsplit(":", 1)[1])
        except ValueError:
            counters["missing_uid_link"] += 1
            continue
        if source not in raw_cache:
            raw_cache[source] = load_jsonl(Path(raw_paths[source]))
        raw_rows = raw_cache[source]
        if row_index >= len(raw_rows):
            counters["index_out_of_range"] += 1
            continue
        raw_row = raw_rows[row_index]
        gold_titles, decomposition = gold_fields(raw_row, source)
        answer = str((raw_row.get("golden_answers") or [""])[0])
        parquet_row = index_to_row.get(f"{source}:{row_index}")
        if parquet_row is None:
            counters["missing_parquet_row"] += 1
            continue
        prompt_text = tokenizer.apply_chat_template(
            list(parquet_row["prompt"]), tokenize=False, add_generation_prompt=True
        )
        if not gold_titles and source != "bamboogle":
            counters["without_gold_titles"] += 1
            continue
        shuffle_pool = [
            other for other in raw_rows
            if gold_fields(other, source)[0] and other is not raw_row
        ]
        if gold_titles and not shuffle_pool:
            counters["shuffle_pool_empty"] += 1
            continue
        shuffle_row = rng.choice(shuffle_pool) if shuffle_pool else None
        shuffle_titles = gold_fields(shuffle_row, source)[0] if shuffle_row else []
        cheats = {
            "student": "",
            "gold": render_cheat("gold", gold_titles, decomposition, answer, []),
            "shuffle": render_cheat("shuffle", gold_titles, decomposition, answer, shuffle_titles),
            "format": render_cheat("format", [], [], answer, []),
            "answer": render_cheat("answer", [], [], answer, []),
        }
        audit_events.append({
            **event,
            "source": source,
            "uid": uid,
            "gold_titles": gold_titles,
            "decomposition": decomposition,
            "target_hit_gold": bool(event["target_evidence_titles"] & {
                normalize(t) for t in gold_titles
            }),
            "contexts": build_contexts(event, prompt_text, cheats),
            "cheat_texts": {arm: cheats[arm] for arm in ARMS if cheats[arm]},
        })

    if not audit_events:
        parser.error("no auditable events; check counters")

    print(f"[audit] scoring {len(audit_events)} events x {len(ARMS)} arms ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()

    results: list[dict[str, Any]] = []
    for arm in ARMS:
        contexts = [event["contexts"][arm] for event in audit_events]
        queries = [event["target_query"] for event in audit_events]
        means, token_logprobs, truncated = score_arm(
            contexts, queries, model, tokenizer, args.device,
            args.batch_size, args.max_context,
        )
        for index, event in enumerate(audit_events):
            results.append({
                "arm": arm,
                "event_index": index,
                "mean_logprob": means[index],
                "token_logprobs": token_logprobs[index],
                "truncated": truncated[index],
            })
        print(f"[audit] arm={arm} scored={sum(m is not None for m in means)}")

    per_event: dict[int, dict[str, Any]] = {i: {} for i in range(len(audit_events))}
    for row in results:
        if row["mean_logprob"] is not None:
            per_event[row["event_index"]][row["arm"]] = row["mean_logprob"]

    def paired(arm_a: str, arm_b: str) -> dict[str, Any]:
        deltas, token_deltas = [], []
        for index in range(len(audit_events)):
            arms = per_event[index]
            if arm_a in arms and arm_b in arms:
                deltas.append(arms[arm_a] - arms[arm_b])
        return {
            "n": len(deltas),
            "mean_lift": sum(deltas) / len(deltas) if deltas else float("nan"),
            "ci95": bootstrap_ci(deltas),
        }

    def positive_fraction(arm: str) -> float | None:
        flags = []
        for index in range(len(audit_events)):
            arms = per_event[index]
            if arm in arms and "student" in arms:
                flags.append(float(arms[arm] > arms["student"]))
        return sum(flags) / len(flags) if flags else None

    def strata(field: str, arm: str = "gold") -> dict[str, Any]:
        groups: dict[str, list[float]] = defaultdict(list)
        for index, event in enumerate(audit_events):
            arms = per_event[index]
            if arm in arms and "student" in arms:
                key = str(event.get(field))
                groups[key].append(arms[arm] - arms["student"])
        return {
            key: {"n": len(v), "mean_lift": sum(v) / len(v)}
            for key, v in sorted(groups.items())
        }

    summary = {
        "protocol": "offline four-way gold-conditioned teacher audit",
        "model": args.model,
        "trees": [str(p) for p in tree_paths],
        "config": {
            "min_value_gap": args.min_value_gap,
            "min_raw_advantage": args.min_raw_advantage,
            "max_query_tokens": args.max_query_tokens,
            "max_context": args.max_context,
            "seed": args.seed,
        },
        "counters": counters,
        "events": len(audit_events),
        "arms": {arm: {"mean_logprob": None} for arm in ARMS},
        "lift_vs_student": {
            arm: {
                "mean_lift": paired(arm, "student")["mean_lift"],
                "ci95": paired(arm, "student")["ci95"],
                "n": paired(arm, "student")["n"],
                "positive_event_fraction": positive_fraction(arm),
            }
            for arm in ARMS if arm != "student"
        },
        "controls": {
            "gold_minus_shuffle": paired("gold", "shuffle"),
            "gold_minus_format": paired("gold", "format"),
            "answer_minus_gold": paired("answer", "gold"),
        },
        "strata": {
            "by_source": strata("source"),
            "by_target_hit_gold": strata("target_hit_gold"),
            "by_sibling_count": strata("sibling_count"),
        },
        "per_event": [
            {
                "event_index": index,
                "source": event["source"],
                "uid": event["uid"],
                "target_query": event["target_query"],
                "sibling_queries": event["sibling_queries"],
                "target_value": event["target_value"],
                "raw_advantage": event["raw_advantage"],
                "target_hit_gold": event["target_hit_gold"],
                "mean_logprob_by_arm": per_event[index],
            }
            for index, event in enumerate(audit_events)
        ],
    }
    for arm in ARMS:
        values = [arms[arm] for arms in per_event.values() if arm in arms]
        summary["arms"][arm]["mean_logprob"] = (
            sum(values) / len(values) if values else None
        )

    gold_shuffle = summary["controls"]["gold_minus_shuffle"]
    gold_format = summary["controls"]["gold_minus_format"]
    summary["verdict"] = {
        "gold_beats_shuffle": gold_shuffle["ci95"][0] > 0,
        "gold_beats_format": gold_format["ci95"][0] > 0,
        "gate_pass": gold_shuffle["ci95"][0] > 0 and gold_format["ci95"][0] > 0,
        "note": "gate requires the gold cheat sheet to beat both the shuffle and the format-only control",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("events", "arms", "lift_vs_student", "controls", "verdict")}, ensure_ascii=False, indent=2))
    print(f"[audit] written: {args.output}")


if __name__ == "__main__":
    main()
