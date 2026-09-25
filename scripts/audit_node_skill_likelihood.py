#!/usr/bin/env python3
"""Score fixed good/bad node actions under plain, real-skill, and shuffled-skill contexts."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
import transformers

from audit_gold_conditioned_teacher import bootstrap_ci
from audit_think_search_teacher import score


ARMS = ("plain", "real", "shuffle")
SPANS = ("think", "search", "all")
INSTRUCTION = ("Use this answer-free procedural skill to choose the next action from the current visible state. "
               "Return exactly <think>...</think><search>...</search>; do not state a final answer.\n")


def read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_action(text: str) -> tuple[str, str]:
    match = re.fullmatch(r"<think>(.+?)</think>\n<search>([^\n<>]+)</search>", text, re.DOTALL)
    if not match:
        raise ValueError("invalid action target")
    return match.group(1).strip(), match.group(2).strip()


def skill_block(skill: dict[str, Any]) -> str:
    raw = json.dumps(skill, ensure_ascii=False, separators=(",", ":"))
    return f"\n<analyzer_skill>\n{raw}\n</analyzer_skill>\n{INSTRUCTION}"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": len(rows), "arms": {}, "controls": {}}
    if not rows:
        return output
    for arm in ARMS:
        output["arms"][arm] = {}
        for span in SPANS:
            good = [r["scores"][arm]["good"][span] for r in rows]
            bad = [r["scores"][arm]["bad"][span] for r in rows]
            margin = [g - b for g, b in zip(good, bad)]
            output["arms"][arm][span] = {
                "good_mean_logprob": sum(good) / len(good),
                "bad_mean_logprob": sum(bad) / len(bad),
                "good_minus_bad": sum(margin) / len(margin),
            }
    for arm in ("real", "shuffle"):
        for span in SPANS:
            good_lift = [r["scores"][arm]["good"][span] - r["scores"]["plain"]["good"][span] for r in rows]
            margin_lift = [
                (r["scores"][arm]["good"][span] - r["scores"][arm]["bad"][span])
                - (r["scores"]["plain"]["good"][span] - r["scores"]["plain"]["bad"][span])
                for r in rows
            ]
            output["controls"][f"{arm}_minus_plain_{span}"] = {
                "good_lift": sum(good_lift) / len(good_lift), "good_lift_ci95": bootstrap_ci(good_lift, seed=211),
                "margin_lift": sum(margin_lift) / len(margin_lift), "margin_lift_ci95": bootstrap_ci(margin_lift, seed=223),
            }
    for span in SPANS:
        real_over_shuffle = [
            (r["scores"]["real"]["good"][span] - r["scores"]["real"]["bad"][span])
            - (r["scores"]["shuffle"]["good"][span] - r["scores"]["shuffle"]["bad"][span])
            for r in rows
        ]
        output["controls"][f"real_minus_shuffle_margin_{span}"] = {
            "mean": sum(real_over_shuffle) / len(real_over_shuffle),
            "ci95": bootstrap_ci(real_over_shuffle, seed=227),
        }
    return output


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True); p.add_argument("--consumer", required=True)
    p.add_argument("--prepared", required=True); p.add_argument("--analyzer-train")
    p.add_argument("--train-split"); p.add_argument("--validation-split")
    p.add_argument("--output", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--min-value-gap", type=float, default=.25)
    p.add_argument("--eval-split", choices=("all", "train", "heldout"), default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260923); args = p.parse_args()
    prepared = {r["event_id"]: r for r in read(args.prepared)}
    if args.train_split:
        train_uids = {r["uid"] for r in read(args.train_split)}
        validation_uids = {r["uid"] for r in read(args.validation_split)} if args.validation_split else set()
        if train_uids & validation_uids:
            raise ValueError("train/validation UID leakage")
    elif args.analyzer_train:
        train_uids = {r["uid"] for r in read(args.analyzer_train)}
        validation_uids = set()
    else:
        raise ValueError("provide --train-split or --analyzer-train")
    consumers = [r for r in read(args.consumer) if float(r.get("value_gap", 0)) >= args.min_value_gap
                 and "[redacted]" not in r["response"].casefold()]
    pool = sorted(consumers, key=lambda r: r["event_id"]); rng = random.Random(args.seed)
    events = []
    for row in consumers:
        if validation_uids and row["uid"] not in train_uids | validation_uids:
            continue
        source = prepared[row["event_id"]]
        candidates = [other for other in pool if other["uid"] != row["uid"]]
        shuffled = candidates[rng.randrange(len(candidates))]
        events.append({"event_id": row["event_id"], "uid": row["uid"],
                       "split": "train" if row["uid"] in train_uids else "heldout",
                       "state": source["state"], "good": parse_action(row["response"]),
                       "bad": (source["bad_think"], source["bad_query"]),
                       "real_skill": row["node_skill"], "shuffle_skill": shuffled["node_skill"],
                       "shuffle_uid": shuffled["uid"]})
    if args.eval_split != "all":
        events = [event for event in events if event["split"] == args.eval_split]
    events.sort(key=lambda event: event["event_id"])
    if args.limit:
        events = events[:args.limit]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    rows = [{"event_id": e["event_id"], "uid": e["uid"], "split": e["split"],
             "shuffle_uid": e["shuffle_uid"], "scores": {}} for e in events]
    for arm in ARMS:
        if arm == "plain": contexts = [e["state"] + "\n" + INSTRUCTION for e in events]
        elif arm == "real": contexts = [e["state"] + skill_block(e["real_skill"]) for e in events]
        else: contexts = [e["state"] + skill_block(e["shuffle_skill"]) for e in events]
        for quality in ("good", "bad"):
            values = score([(contexts[i], e[quality]) for i, e in enumerate(events)], model, tokenizer,
                           args.device, args.batch_size, args.max_context)
            for i, value in enumerate(values): rows[i]["scores"].setdefault(arm, {})[quality] = value
    rows = [r for r in rows if all(r["scores"][a].get("good") and r["scores"][a].get("bad") for a in ARMS)]
    report = {"model": args.model, "events": len(rows), "all": summarize(rows),
              "train": summarize([r for r in rows if r["split"] == "train"]),
              "heldout": summarize([r for r in rows if r["split"] == "heldout"]), "per_event": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("events", "all", "train", "heldout")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
