#!/usr/bin/env python3
"""Build audited node-level skill -> action SFT pairs from sibling trees."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from audit_gold_conditioned_teacher import load_jsonl
from audit_think_search_teacher import collect_events
from data_process.build_tree_seed_analyzer_sft import (
    answer_aliases, deepseek_completion, extract_json_object, extract_titles,
    normalize, phrase_present, redact_answers, safe_error,
)


FIELDS = {"failure_type", "missing_relation", "next_operation", "stop_condition"}


def stable_id(event: dict[str, Any]) -> str:
    raw = f"{event['tree_uid']}:{event['parent_uid']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_metadata(patterns: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for row in load_jsonl(Path(path)):
                result.setdefault(str(row.get("tree_uid", "")), row)
    return result


def prepare_events(args: argparse.Namespace) -> list[dict[str, Any]]:
    tree_paths = sorted(path for pattern in args.trees for path in glob.glob(pattern))
    records = [row for path in tree_paths for row in load_jsonl(Path(path))]
    metadata = load_metadata(args.selected)
    events = []
    for event in collect_events(records, args.min_value_gap, args.max_evidence_chars):
        meta = metadata.get(event["tree_uid"])
        if not meta:
            continue
        aliases = answer_aliases(meta.get("ground_truth"))
        state = redact_answers(str(meta.get("prompt", "")) + str(event.get("prefix", "")), aliases)
        good_think, good_query = (redact_answers(x, aliases) for x in event["good"])
        bad_think, bad_query = (redact_answers(x, aliases) for x in event["bad"])
        evidence = redact_answers(str(event.get("evidence", "")), aliases)
        if not good_query or not bad_query or good_query == bad_query:
            continue
        events.append({
            "event_id": stable_id(event), "tree_uid": event["tree_uid"],
            "parent_uid": event["parent_uid"], "uid": str(meta.get("uid", "")),
            "data_source": str(meta.get("data_source", "")), "state": state,
            "good_think": good_think, "good_query": good_query,
            "bad_think": bad_think, "bad_query": bad_query,
            "better_evidence": evidence, "good_value": event["good_value"],
            "bad_value": event["bad_value"], "value_gap": event["advantage"],
            "aliases": aliases, "better_titles": extract_titles(evidence),
        })
    # The same training question/node can occur in multiple resumed rollout
    # directories.  Keep one version per stable node and prefer the clearest
    # value separation instead of silently training on duplicated targets.
    deduplicated: dict[str, dict[str, Any]] = {}
    for event in events:
        previous = deduplicated.get(event["event_id"])
        if previous is None or float(event["value_gap"]) > float(previous["value_gap"]):
            deduplicated[event["event_id"]] = event
    events = list(deduplicated.values())
    random.Random(args.seed).shuffle(events)
    return events[:args.limit] if args.limit else events


def teacher_prompt(event: dict[str, Any]) -> str:
    return f"""Create one node-level procedural skill for a search agent. Return ONLY JSON.

Schema:
{{"failure_type":"short category","missing_relation":"abstract unresolved relation or evidence gap","next_operation":"one executable verification/search operation","stop_condition":"what must be verified before advancing"}}

You may compare the two historical actions below only to diagnose what operation was needed. The final skill will be shown before the next action.

Strict constraints:
- Do not reveal, infer, or claim the final answer.
- Do not copy or closely paraphrase either search query, any document title, retrieved sentence, date, number, newly revealed entity, or answer-bearing fact.
- Do not mention rewards, scores, branches, gold evidence, hindsight, or that one action was better.
- Use only entity types, relation types, ambiguity types, evidence gaps, and verification operations.
- Make the skill specific to this decision state but usable without knowing the target action.

Current visible state:
{event['state'][-9000:]}

Historical unsuccessful action:
<think>{event['bad_think']}</think>
<search>{event['bad_query']}</search>

Historical more useful action (private diagnostic input; do not copy it):
<think>{event['good_think']}</think>
<search>{event['good_query']}</search>
Retrieved evidence from that action (private diagnostic input; do not copy it):
{event['better_evidence'][:1800]}
"""


def validate_skill(parsed: dict[str, Any], event: dict[str, Any]) -> list[str]:
    reasons = []
    if set(parsed) != FIELDS:
        reasons.append("schema")
    combined = "\n".join(str(parsed.get(key, "")).strip() for key in sorted(FIELDS))
    if any(not str(parsed.get(key, "")).strip() for key in FIELDS):
        reasons.append("empty")
    words = normalize(combined).split()
    if len(words) < 14 or len(words) > 120:
        reasons.append("length")
    if any(phrase_present(combined, alias) for alias in event["aliases"]):
        reasons.append("answer_leak")
    if len(normalize(event["good_query"]).split()) >= 3 and phrase_present(combined, event["good_query"]):
        reasons.append("target_query_copy")
    if len(normalize(event["bad_query"]).split()) >= 3 and phrase_present(combined, event["bad_query"]):
        reasons.append("bad_query_copy")
    if any(len(normalize(title).split()) >= 2 and phrase_present(combined, title)
           for title in event["better_titles"]):
        reasons.append("title_copy")
    if re.search(r"\b(?:reward|score|gold|branch|hindsight|better action|correct answer)\b", combined, re.I):
        reasons.append("privileged_meta")
    if "[redacted]" in combined.casefold():
        reasons.append("redaction_artifact")
    return sorted(set(reasons))


def render_skill(parsed: dict[str, Any]) -> str:
    return json.dumps({key: str(parsed[key]).strip() for key in
                       ("failure_type", "missing_relation", "next_operation", "stop_condition")},
                      ensure_ascii=False, separators=(",", ":"))


def to_sft(event: dict[str, Any], parsed: dict[str, Any], model: str) -> dict[str, Any]:
    skill = render_skill(parsed)
    prompt = (
        event["state"]
        + "\n<analyzer_skill>\n" + skill + "\n</analyzer_skill>\n"
        + "Use this answer-free procedural skill to choose the next action from the current visible state. "
          "Return exactly <think>...</think><search>...</search>; do not state a final answer.\n"
    )
    response = f"<think>{event['good_think']}</think>\n<search>{event['good_query']}</search>"
    return {"prompt": prompt, "response": response, "task_type": "skill_consumer",
            "event_id": event["event_id"], "uid": event["uid"], "tree_uid": event["tree_uid"],
            "parent_uid": event["parent_uid"], "data_source": event["data_source"],
            "value_gap": event["value_gap"], "node_skill": parsed, "model": model}


def process(event: dict[str, Any], args: argparse.Namespace, key: str) -> dict[str, Any]:
    try:
        raw = deepseek_completion(api_key=key, base_url=args.base_url, model=args.model,
                                  prompt=teacher_prompt(event), timeout=args.timeout, retries=args.retries)
        parsed = extract_json_object(raw)
        reasons = validate_skill(parsed, event)
        return {"accepted": not reasons, "rejection_reasons": reasons,
                "sft": to_sft(event, parsed, args.model) if not reasons else None,
                "event_id": event["event_id"], "uid": event["uid"], "raw_response": raw}
    except Exception as exc:
        return {"accepted": False, "rejection_reasons": ["api_or_parse_error"], "sft": None,
                "event_id": event["event_id"], "uid": event["uid"], "error": safe_error(exc, key)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True); parser.add_argument("--selected", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-value-gap", type=float, default=.1); parser.add_argument("--max-evidence-chars", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260923); parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default="deepseek-chat"); parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--timeout", type=float, default=90); parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    events = prepare_events(args); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    prepared = [{key: value for key, value in event.items() if key not in {"aliases"}} for event in events]
    write_jsonl(output / "prepared.jsonl", prepared)
    if args.dry_run:
        print(json.dumps({"prepared": len(events), "uids": len({e['uid'] for e in events})})); return 0
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, event, args, key) for event in events]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["event_id"])
    accepted = [row["sft"] for row in results if row["accepted"]]
    rejected = [row for row in results if not row["accepted"]]
    write_jsonl(output / "accepted.jsonl", accepted); write_jsonl(output / "rejected.jsonl", rejected)
    reasons = Counter(reason for row in rejected for reason in row["rejection_reasons"])
    summary = {"prepared": len(events), "accepted": len(accepted), "rejected": len(rejected),
               "acceptance_rate": len(accepted) / max(1, len(events)), "unique_uids": len({r['uid'] for r in accepted}),
               "rejection_reasons": dict(reasons), "quality_gate_pass": len(accepted) >= max(10, .7 * len(events))}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
