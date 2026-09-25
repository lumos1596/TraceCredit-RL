#!/usr/bin/env python3
"""Qualification audit for outcome-privileged think+search teachers.

Each event is a sibling group with a clear best and worst branch.  The same
context is used to score both actions, so the measured good-minus-bad margin
cannot be explained by target-dependent prompting.  Three arms are compared:
visible state only, real leave-one-out hindsight, and another event's shuffled
hindsight.  Think and search tokens are reported separately.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
import transformers

from audit_gold_conditioned_teacher import DEFAULT_RAW, bootstrap_ci, load_jsonl

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
INFO_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)
ARMS = ("student", "real", "shuffle")
SPANS = ("think", "search", "all")


def edge_text(parent: dict[str, Any], child: dict[str, Any]) -> str | None:
    prefix, response = parent.get("response") or "", child.get("response") or ""
    return response[len(prefix):] if response.startswith(prefix) else None


def action_parts(edge: str) -> tuple[str, str] | None:
    think, search = THINK_RE.search(edge), SEARCH_RE.search(edge)
    if not think or not search or not think.group(1).strip() or not search.group(1).strip():
        return None
    return think.group(1).strip(), search.group(1).strip()


def evidence(edge: str, max_chars: int) -> str:
    match = INFO_RE.search(edge)
    return (match.group(1).strip()[:max_chars] if match else "")


def render_hindsight(event: dict[str, Any]) -> str:
    alternatives = "\n".join(
        f"- anonymized alternative branch {index + 1}: value {value:.4f}"
        for index, (_, value) in enumerate(event["alternatives"])
    ) or "- (no textual alternative available)"
    return (
        "\n<hindsight>\nA better branch from this exact search state later retrieved:\n"
        f"{event['evidence'] or '(no retrieval evidence was returned)'}\n"
        f"Its branch value was {event['good_value']:.4f}; its local advantage "
        f"over the alternatives was {event['advantage']:.4f}.\n"
        "Outcomes of other attempted searches from the same state "
        "(query text withheld to prevent copying):\n"
        f"{alternatives}\n</hindsight>\n"
        "Use this hindsight only to correct the direction of the agent's private "
        "reasoning and next search. Never state or imply a final answer, and never "
        "claim access to future information. Continue as an agent that only knows "
        "the visible current state.\n"
    )


def redact_answer(text: str, answer: str) -> str:
    words = str(answer or "").split()
    if not words:
        return text
    return re.sub(r"\s+".join(re.escape(word) for word in words), "[REDACTED]", text,
                  flags=re.IGNORECASE)


def load_tree_metadata(selected_patterns: list[str], parquet_path: str, tokenizer):
    import pandas as pd
    uid_by_tree = {}
    for pattern in selected_patterns:
        for path in glob.glob(pattern):
            for row in load_jsonl(Path(path)):
                if row.get("tree_uid") and row.get("uid"):
                    uid_by_tree[str(row["tree_uid"])] = (str(row.get("data_source", "")), str(row["uid"]))
    prompts = {}
    for _, row in pd.read_parquet(parquet_path).iterrows():
        info = row.get("extra_info") or {}
        if info.get("index"):
            prompts[str(info["index"])] = tokenizer.apply_chat_template(
                list(row["prompt"]), tokenize=False, add_generation_prompt=True)
    raw_cache = {source: load_jsonl(Path(path)) for source, path in DEFAULT_RAW.items() if Path(path).exists()}
    metadata = {}
    for tree_uid, (source, uid) in uid_by_tree.items():
        try: index = int(uid.rsplit(":", 1)[1])
        except (ValueError, IndexError): continue
        rows = raw_cache.get(source, [])
        answer = str((rows[index].get("golden_answers") or [""])[0]) if index < len(rows) else ""
        metadata[tree_uid] = {"prompt": prompts.get(f"{source}:{index}", ""), "answer": answer}
    return metadata


def collect_events(records: list[dict[str, Any]], min_gap: float,
                   max_evidence_chars: int) -> list[dict[str, Any]]:
    events = []
    for record in records:
        nodes = {node["node_uid"]: node for node in record["nodes"]}
        for parent in record["nodes"]:
            candidates = []
            for uid in parent.get("child_node_uids", []):
                child = nodes.get(uid)
                edge = edge_text(parent, child) if child else None
                parts = action_parts(edge) if edge is not None else None
                if parts:
                    candidates.append((float(child.get("node_value", 0.0)), child, edge, parts))
            if len(candidates) < 2:
                continue
            candidates.sort(key=lambda item: item[0])
            bad, good = candidates[0], candidates[-1]
            if not math.isfinite(good[0] - bad[0]) or good[0] - bad[0] < min_gap:
                continue
            others = [(item[3][1], item[0]) for item in candidates if item[1] is not good[1]]
            events.append({
                "tree_uid": record["tree_uid"], "parent_uid": parent["node_uid"],
                "prefix": parent.get("response") or "", "good": good[3], "bad": bad[3],
                "good_value": good[0], "bad_value": bad[0], "advantage": good[0] - bad[0],
                "evidence": evidence(good[2], max_evidence_chars), "alternatives": others,
            })
    return events


def target_tokens(tokenizer, parts: tuple[str, str]) -> tuple[list[int], dict[str, list[bool]]]:
    segments = [("", "<think>"), ("think", parts[0]), ("", "</think>\n<search>"),
                ("search", parts[1]), ("", "</search>")]
    ids, labels = [], []
    for label, text in segments:
        piece = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(piece); labels.extend([label] * len(piece))
    masks = {span: [label == span for label in labels] for span in ("think", "search")}
    masks["all"] = [bool(label) for label in labels]
    return ids, masks


@torch.no_grad()
def score(items: list[tuple[str, tuple[str, str]]], model, tokenizer, device: str,
          batch_size: int, max_context: int) -> list[dict[str, float] | None]:
    prepared = []
    for prefix, parts in items:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        target_ids, masks = target_tokens(tokenizer, parts)
        keep = max_context - len(target_ids)
        if keep <= 0:
            prepared.append(None); continue
        prepared.append((prefix_ids[-keep:] + target_ids, len(prefix_ids[-keep:]), masks))
    output: list[dict[str, float] | None] = [None] * len(items)
    order = sorted((i for i, x in enumerate(prepared) if x), key=lambda i: len(prepared[i][0]))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]; batch = [prepared[i] for i in indices]
        width = max(len(x[0]) for x in batch); pad = tokenizer.pad_token_id
        input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, (ids, _, _) in enumerate(batch):
            input_ids[row, -len(ids):] = torch.tensor(ids); attention[row, -len(ids):] = 1
        position_ids = (attention.cumsum(-1) - 1).clamp_min(0)
        logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                       position_ids=position_ids.to(device), use_cache=False).logits.float().log_softmax(-1)
        for row, index in enumerate(indices):
            ids, prefix_len, masks = prepared[index]; target_len = len(ids) - prefix_len
            positions = torch.arange(width - target_len - 1, width - 1, device=device)
            targets = input_ids[row, -target_len:].to(device)
            lp = logits[row, positions].gather(-1, targets[:, None]).squeeze(-1).cpu().tolist()
            output[index] = {span: sum(v for v, use in zip(lp, masks[span]) if use) /
                             max(1, sum(masks[span])) for span in SPANS}
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        result["arms"][arm] = {}
        for span in SPANS:
            margins = [row["good"][span] - row["bad"][span] for row in arm_rows]
            result["arms"][arm][span] = {"n": len(margins), "mean_margin": sum(margins) / len(margins),
                                                   "ci95": bootstrap_ci(margins, seed=31)}
    result["controls"] = {}
    for left, right in (("real", "student"), ("real", "shuffle")):
        for span in SPANS:
            by_event = {(r["event"], r["arm"]): r["good"][span] - r["bad"][span] for r in rows}
            deltas = [by_event[(e, left)] - by_event[(e, right)]
                      for e in sorted({r["event"] for r in rows})]
            result["controls"][f"{left}_minus_{right}_{span}"] = {
                "n": len(deltas), "mean_margin_lift": sum(deltas) / len(deltas),
                "ci95": bootstrap_ci(deltas, seed=37)}
    query_gates = [result["controls"][f"real_minus_{base}_search"]["ci95"][0] > 0
                   for base in ("student", "shuffle")]
    think_nonharm = result["controls"]["real_minus_student_think"]["ci95"][1] >= 0
    result["verdict"] = {"gate_pass": all(query_gates) and think_nonharm,
                         "search_beats_student_and_shuffle": all(query_gates),
                         "think_not_significantly_harmed": think_nonharm}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--selected", nargs="+", required=True)
    parser.add_argument("--parquet", default="data/multihopqa_search_mixed_402020_20260830/train.parquet")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--min-value-gap", type=float, default=.25); parser.add_argument("--max-evidence-chars", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only-student", action="store_true", help="score visible-state baseline only")
    args = parser.parse_args()
    paths = sorted(p for pattern in args.trees for p in glob.glob(pattern))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model); tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    records = [row for path in paths for row in load_jsonl(Path(path))]
    metadata = load_tree_metadata(args.selected, args.parquet, tokenizer)
    events = collect_events(records, args.min_value_gap, args.max_evidence_chars)
    events = [event for event in events if event["tree_uid"] in metadata and metadata[event["tree_uid"]]["prompt"]]
    for event in events:
        meta = metadata[event["tree_uid"]]
        event["prefix"] = meta["prompt"] + event["prefix"]
        event["evidence"] = redact_answer(event["evidence"], meta["answer"])
        event["alternatives"] = [(redact_answer(query, meta["answer"]), value)
                                 for query, value in event["alternatives"]]
    random.Random(args.seed).shuffle(events); events = events[:args.max_events]
    if len(events) < 2: parser.error(f"only {len(events)} auditable events")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                               attn_implementation="sdpa").to(args.device).eval()
    sheets = [render_hindsight(event) for event in events]
    rows = []
    active_arms = ("student",) if args.only_student else ARMS
    for arm in active_arms:
        contexts = []
        for i, event in enumerate(events):
            sheet = "" if arm == "student" else sheets[i if arm == "real" else (i + 1) % len(events)]
            contexts.append(event["prefix"] + sheet)
        for quality in ("good", "bad"):
            scores = score([(contexts[i], event[quality]) for i, event in enumerate(events)], model,
                           tokenizer, args.device, args.batch_size, args.max_context)
            for i, value in enumerate(scores):
                if value is None: continue
                existing = next((r for r in rows if r["event"] == i and r["arm"] == arm), None)
                if existing is None: existing = {"event": i, "arm": arm}; rows.append(existing)
                existing[quality] = value
    rows = [r for r in rows if "good" in r and "bad" in r]
    if args.only_student:
        arm_summary = {}
        for span in SPANS:
            margins = [row["good"][span] - row["bad"][span] for row in rows]
            arm_summary[span] = {"n": len(margins), "mean_margin": sum(margins) / len(margins),
                                 "ci95": bootstrap_ci(margins, seed=31)}
        stats = {"arms": {"student": arm_summary}, "controls": {}, "verdict": {"baseline_only": True}}
    else:
        stats = summarize(rows)
    summary = {"protocol": "paired think+search teacher qualification", "model": args.model,
               "trees": paths, "events": len(events), **stats, "per_event": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("events", "arms", "controls", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
