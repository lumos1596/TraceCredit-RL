#!/usr/bin/env python3
"""Fixed-action likelihood audit for answer-free Analyzer skills.

The action text is taken verbatim from existing tree rollouts.  No model
generation or retrieval is performed: the only question is whether adding a
same-question Analyzer skill increases the 3B model's probability of the
already observed high-value action, and whether it increases the good-minus-
bad branch margin rather than all actions indiscriminately.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any

import torch
import transformers

from audit_analyzer_skill_correction_retrieval import load_audited_skills, select_tree_skills, shuffled_skills, skill_sheet
from audit_gold_conditioned_teacher import load_jsonl
from audit_teacher_correction_retrieval import anchored_prefix, bootstrap_ci
from audit_think_search_teacher import collect_events, load_tree_metadata, score


ARMS = ("plain", "real", "shuffle")
SPANS = ("think", "search", "all")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}, "controls": {}}
    for arm in ARMS:
        arm_rows = [row for row in rows if arm in row["scores"]]
        result["arms"][arm] = {}
        for span in SPANS:
            good = [row["scores"][arm]["good"][span] for row in arm_rows]
            bad = [row["scores"][arm]["bad"][span] for row in arm_rows]
            margins = [x - y for x, y in zip(good, bad)]
            result["arms"][arm][span] = {
                "n": len(good),
                "good_mean_logprob": sum(good) / len(good),
                "bad_mean_logprob": sum(bad) / len(bad),
                "good_minus_bad_mean": sum(margins) / len(margins),
                "good_minus_bad_ci95": bootstrap_ci(margins, seed=101),
            }
    for arm in ("real", "shuffle"):
        for span in SPANS:
            good_lift = [row["scores"][arm]["good"][span] - row["scores"]["plain"]["good"][span] for row in rows]
            margin_lift = [
                (row["scores"][arm]["good"][span] - row["scores"][arm]["bad"][span])
                - (row["scores"]["plain"]["good"][span] - row["scores"]["plain"]["bad"][span])
                for row in rows
            ]
            result["controls"][f"{arm}_minus_plain_{span}"] = {
                "n": len(rows),
                "good_logprob_lift_mean": sum(good_lift) / len(good_lift),
                "good_logprob_lift_ci95": bootstrap_ci(good_lift, seed=103),
                "margin_lift_mean": sum(margin_lift) / len(margin_lift),
                "margin_lift_ci95": bootstrap_ci(margin_lift, seed=107),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True)
    parser.add_argument("--selected", nargs="+", required=True)
    parser.add_argument("--skills", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parquet", default="data/multihopqa_search_mixed_402020_20260830/train.parquet")
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--min-value-gap", type=float, default=.25)
    parser.add_argument("--max-evidence-chars", type=int, default=1600)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tree_skills = select_tree_skills(load_audited_skills(args.skills))
    tree_paths = sorted(path for pattern in args.trees for path in glob.glob(pattern))
    selected_paths = sorted(path for pattern in args.selected for path in glob.glob(pattern))
    records = [row for path in tree_paths for row in load_jsonl(Path(path))]
    metadata = load_tree_metadata(selected_paths, args.parquet, tokenizer)
    events = []
    for event in collect_events(records, args.min_value_gap, args.max_evidence_chars):
        meta = metadata.get(event["tree_uid"])
        skill = tree_skills.get(event["tree_uid"])
        if not meta or not meta.get("prompt") or skill is None:
            continue
        # The tree metadata loader only needs the prompt; the accepted skill
        # record is the authoritative question UID for the same tree.
        event["uid"] = skill["uid"]
        event["prefix"] = anchored_prefix(
            tokenizer, meta["prompt"], event["prefix"],
            max(256, args.max_context - 420),
        )
        event["real_skill"] = skill["skill"]
        event["real_skill_uid"] = skill["uid"]
        events.append(event)
    random.Random(args.seed).shuffle(events)
    events = events[:args.max_events]
    if len(events) < 2:
        parser.error(f"only {len(events)} auditable events")
    shuffled = shuffled_skills(events, tree_skills, args.seed + 1)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    rows = [{
        "event": i, "tree_uid": event["tree_uid"], "uid": event["uid"],
        "real_skill_uid": event["real_skill_uid"], "shuffle_skill_uid": shuffled[i]["uid"],
        "scores": {},
    } for i, event in enumerate(events)]
    for arm in ARMS:
        contexts = []
        for i, event in enumerate(events):
            if arm == "plain":
                suffix = ""
            elif arm == "real":
                suffix = skill_sheet(event["real_skill"])
            else:
                suffix = skill_sheet(shuffled[i]["skill"])
            contexts.append(event["prefix"] + suffix)
        good_scores = score([(contexts[i], events[i]["good"]) for i in range(len(events))], model, tokenizer,
                            args.device, args.batch_size, args.max_context)
        bad_scores = score([(contexts[i], events[i]["bad"]) for i in range(len(events))], model, tokenizer,
                           args.device, args.batch_size, args.max_context)
        for i, (good, bad) in enumerate(zip(good_scores, bad_scores)):
            if good is not None and bad is not None:
                rows[i]["scores"][arm] = {"good": good, "bad": bad}
    rows = [row for row in rows if len(row["scores"]) == len(ARMS)]
    if len(rows) < 2:
        parser.error("fewer than two complete scored events")
    summary = {
        "protocol": "fixed on-policy action likelihood: plain vs same-tree Analyzer skill vs shuffled skill",
        "model": args.model, "events": len(rows), "skill_tree_count": len(tree_skills),
        **summarize(rows), "per_event": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"events": summary["events"], "controls": summary["controls"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
