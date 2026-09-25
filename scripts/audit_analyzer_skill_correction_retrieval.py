#!/usr/bin/env python3
"""Qualify whether a 3B policy can use audited Analyzer skills at a fixed state.

This is deliberately a *generation* preflight, separate from the eventual
same-token likelihood gate used by sampled-NLL OPD.  A 3B model receives a
real, answer-free Analyzer skill derived from a completed trajectory in the
same tree, no skill, or a deterministic skill from a different question.  It
then proposes one next think/search action.  The generated query is executed
against the normal retriever and paired retrieval metrics decide whether the
skill is useful beyond generic prompt text.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import transformers

from audit_gold_conditioned_teacher import DEFAULT_RAW, gold_fields, load_jsonl, normalize
from audit_teacher_correction_retrieval import (
    TITLE_RE, anchored_prefix, bootstrap_ci, document_title, gate, generate,
    leaks_answer, parse_correction, retrieve_batch, score_retrieval,
)
from audit_think_search_teacher import collect_events, load_tree_metadata, redact_answer


ARMS = ("real", "plain", "shuffle")


def load_audited_skills(paths: Sequence[str]) -> list[dict[str, str]]:
    """Load only accepted, answer-free skill records; never use raw generation text."""
    skills = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            for row in load_jsonl(Path(path)):
                if not row.get("accepted") or row.get("rejection_reasons"):
                    continue
                skill = str(row.get("episode_skill", "")).strip()
                tree_uid, uid = str(row.get("tree_uid", "")), str(row.get("uid", ""))
                if skill and tree_uid and uid:
                    skills.append({"tree_uid": tree_uid, "uid": uid, "skill": skill,
                                   "sample_id": str(row.get("sample_id", "")),
                                   "score": float(row.get("source_score", 0.0))})
    if not skills:
        raise ValueError("no accepted Analyzer skills found")
    return skills


def select_tree_skills(skills: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Choose one reproducible skill per tree, preferring the strongest trajectory."""
    selected: dict[str, dict[str, str]] = {}
    for item in skills:
        prior = selected.get(item["tree_uid"])
        if prior is None or (item["score"], item["sample_id"]) > (prior["score"], prior["sample_id"]):
            selected[item["tree_uid"]] = item
    return selected


def shuffled_skills(events: Sequence[dict[str, Any]], tree_skills: dict[str, dict[str, str]], seed: int) -> list[dict[str, str]]:
    """Deterministically assign a skill from another question to every event."""
    pool = sorted(tree_skills.values(), key=lambda x: (x["uid"], x["tree_uid"], x["sample_id"]))
    if len({item["uid"] for item in pool}) < 2:
        raise ValueError("need skills from at least two distinct questions for shuffled control")
    rng = random.Random(seed)
    output = []
    for event in events:
        candidates = [item for item in pool if item["uid"] != event["uid"]]
        if not candidates:
            raise ValueError(f"cannot find cross-question shuffle for {event['tree_uid']}")
        output.append(candidates[rng.randrange(len(candidates))])
    return output


def skill_sheet(skill: str) -> str:
    return (
        "\n<analyzer_skill>\n" + skill + "\n</analyzer_skill>\n"
        "The Analyzer skill is a general, answer-free diagnosis from a completed episode. "
        "Use it only as a procedural hint; do not claim to know future evidence or an answer.\n"
    )


def correction_instruction(event: dict[str, Any], sheet: str) -> str:
    bad_think, bad_query = (redact_answer(value, event.get("answer", "")) for value in event["bad"])
    return (
        event["prefix"] + "\n<student_attempt>\n"
        + f"<think>{bad_think}</think>\n<search>{bad_query}</search>\n</student_attempt>\n"
        + sheet
        + "The student's attempted reasoning/search may be wrong. Produce a better next action using only "
          "the visible state and any answer-free Analyzer skill above. Do not state or imply a final answer. "
          "Return exactly one action as <think>...</think><search>...</search>.\n"
    )


def paired_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in ("student",) + ARMS:
        values = [row["scores"][arm][metric] for row in rows]
        summary[arm] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
    for baseline in ("student", "plain", "shuffle"):
        deltas = [row["scores"]["real"][metric] - row["scores"][baseline][metric] for row in rows]
        summary[f"real_minus_{baseline}"] = {"n": len(deltas), "mean": sum(deltas) / len(deltas),
                                               "ci95": bootstrap_ci(deltas, seed=73)}
    return summary


def skill_gate(rows: list[dict[str, Any]], valid_rates: dict[str, float], leak_rates: dict[str, float],
               min_hit_lift: float, min_valid_rate: float, max_leak_rate: float) -> dict[str, Any]:
    metrics = {metric: paired_summary(rows, metric) for metric in ("hit", "mrr", "recall")}
    hit = metrics["hit"]
    # The generated-action test is stringent: real must beat the original action,
    # plain correction prompt, and a wrong skill.  CIs protect against a noisy lift.
    hit_checks = [hit[f"real_minus_{base}"]["mean"] >= min_hit_lift for base in ("student", "plain", "shuffle")]
    ci_checks = [hit[f"real_minus_{base}"]["ci95"][0] >= 0 for base in ("plain", "shuffle")]
    mrr_checks = [metrics["mrr"][f"real_minus_{base}"]["mean"] > 0 for base in ("plain", "shuffle")]
    valid_check = valid_rates["real"] >= min_valid_rate
    leak_check = leak_rates["real"] <= max_leak_rate
    return {"gate_pass": all(hit_checks + ci_checks + mrr_checks + [valid_check, leak_check]),
            "hit_checks": hit_checks, "ci95_nonnegative_vs_plain_shuffle": ci_checks,
            "mrr_checks": mrr_checks, "valid_rate_check": valid_check, "answer_leak_check": leak_check,
            "min_hit_lift": min_hit_lift, "min_valid_rate": min_valid_rate,
            "max_answer_leak_rate": max_leak_rate, "metrics": metrics,
            "note": "Generation-level feasibility gate; sampled-NLL still requires its separate likelihood audit."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True); parser.add_argument("--selected", nargs="+", required=True)
    parser.add_argument("--skills", nargs="+", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parquet", default="data/multihopqa_search_mixed_402020_20260830/train.parquet")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retrieval-topk", type=int, default=3); parser.add_argument("--retrieval-timeout", type=float, default=90)
    parser.add_argument("--max-events", type=int, default=48); parser.add_argument("--min-value-gap", type=float, default=.25)
    parser.add_argument("--max-evidence-chars", type=int, default=1600); parser.add_argument("--max-context", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=160); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-hit-lift", type=float, default=.05); parser.add_argument("--min-valid-rate", type=float, default=.8)
    parser.add_argument("--max-answer-leak-rate", type=float, default=0.0); parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    skills = select_tree_skills(load_audited_skills(args.skills))
    tree_paths = sorted(path for pattern in args.trees for path in glob.glob(pattern))
    records = [row for path in tree_paths for row in load_jsonl(Path(path))]
    metadata = load_tree_metadata(args.selected, args.parquet, tokenizer)
    raw_cache = {source: load_jsonl(Path(path)) for source, path in DEFAULT_RAW.items() if Path(path).exists()}
    for pattern in args.selected:
        for path in glob.glob(pattern):
            for row in load_jsonl(Path(path)):
                tree_uid, source, uid = str(row.get("tree_uid", "")), str(row.get("data_source", "")), str(row.get("uid", ""))
                if tree_uid not in metadata: continue
                try: index = int(uid.rsplit(":", 1)[1])
                except (ValueError, IndexError): continue
                raw = raw_cache.get(source, [])
                if index < len(raw):
                    titles, _ = gold_fields(raw[index], source)
                    metadata[tree_uid].update(source=source, index=index, uid=uid, gold_titles=titles)

    events = []
    for event in collect_events(records, args.min_value_gap, args.max_evidence_chars):
        meta, real_skill = metadata.get(event["tree_uid"], {}), skills.get(event["tree_uid"])
        gold = {normalize(title) for title in meta.get("gold_titles", []) if title}
        visible = {normalize(title.strip().strip('\"“”')) for title in TITLE_RE.findall(event["prefix"])}
        if not meta.get("prompt") or not real_skill or not (remaining := gold - visible):
            continue
        event.update(uid=meta["uid"], answer=meta.get("answer", ""), remaining_gold=remaining,
                     real_skill=real_skill["skill"])
        event["prefix"] = anchored_prefix(tokenizer, meta["prompt"], event["prefix"], max(256, args.max_context - args.max_new_tokens - 420))
        events.append(event)
    random.Random(args.seed).shuffle(events); events = events[:args.max_events]
    if len(events) < 2: parser.error(f"only {len(events)} usable events with Analyzer skills")
    shuffled = shuffled_skills(events, skills, args.seed + 1)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    generations: dict[str, list[str]] = {}
    for arm in ARMS:
        sheets = [skill_sheet(event["real_skill"]) if arm == "real" else ("" if arm == "plain" else skill_sheet(shuffled[i]["skill"])) for i, event in enumerate(events)]
        generations[arm] = generate(model, tokenizer, [correction_instruction(event, sheet) for event, sheet in zip(events, sheets)], args.device, args.batch_size, args.max_context, args.max_new_tokens)
    del model; torch.cuda.empty_cache()
    jobs, rows = [], []
    for i, event in enumerate(events):
        row = {"event": i, "tree_uid": event["tree_uid"], "uid": event["uid"], "scores": {}, "generations": {},
               "real_skill_sample": skills[event["tree_uid"]]["sample_id"],
               "real_skill_uid": skills[event["tree_uid"]]["uid"],
               "shuffle_skill_sample": shuffled[i]["sample_id"], "shuffle_skill_uid": shuffled[i]["uid"]}
        jobs.append((i, "student", event["bad"][1])); rows.append(row)
        for arm in ARMS:
            raw = generations[arm][i]; parsed = parse_correction(raw); answer_leak = leaks_answer(raw, event["answer"], event["prefix"])
            valid = parsed is not None and not answer_leak
            row["generations"][arm] = {"valid": valid, "answer_leak": answer_leak, "think": parsed[0] if valid else None, "query": parsed[1] if valid else None}
            if valid: jobs.append((i, arm, parsed[1]))
    results = retrieve_batch(args.retrieval_url, [item[2] for item in jobs], args.retrieval_topk, args.retrieval_timeout)
    for (index, arm, query), result in zip(jobs, results):
        rows[index]["scores"][arm] = {**score_retrieval(result, events[index]["remaining_gold"]), "query": query}
    for row in rows:
        for arm in ARMS:
            row["scores"].setdefault(arm, {"hit": 0.0, "mrr": 0.0, "recall": 0.0, "query": None, "rejected": True})
    valid = {arm: sum(row["generations"][arm]["valid"] for row in rows) / len(rows) for arm in ARMS}
    leaks = {arm: sum(row["generations"][arm]["answer_leak"] for row in rows) / len(rows) for arm in ARMS}
    verdict = skill_gate(rows, valid, leaks, args.min_hit_lift, args.min_valid_rate, args.max_answer_leak_rate)
    summary = {"protocol": "3B Analyzer-skill generated correction + executed retrieval", "model": args.model,
               "events_complete": len(rows), "valid_generation_rate": valid, "answer_leak_rate": leaks,
               "verdict": verdict, "per_event": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("events_complete", "valid_generation_rate", "answer_leak_rate", "verdict")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
