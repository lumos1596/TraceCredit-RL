#!/usr/bin/env python3
"""Generate teacher corrections, execute their searches, and gate OPD training.

This is a preflight audit, not an offline training-data builder.  A teacher is
shown a student's low-value sibling action and asked for a leak-free corrected
``<think>``/``<search>`` action.  Real hindsight, no hindsight, and shuffled
hindsight are generated independently.  Their queries plus the original
student query are sent to the project's retriever and scored against the gold
supporting titles not already visible at the parent state.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers

from audit_gold_conditioned_teacher import DEFAULT_RAW, bootstrap_ci, gold_fields, load_jsonl, normalize
from audit_think_search_teacher import collect_events, load_tree_metadata, redact_answer, render_hindsight

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
TITLE_RE = re.compile(r'Doc\s+\d+\s*\(Title:\s*["“]?(.*?)["”]?\)\s*', re.IGNORECASE)
ARMS = ("real", "no_cheat", "shuffle")


def parse_correction(text: str) -> tuple[str, str] | None:
    think, search = THINK_RE.search(text), SEARCH_RE.search(text)
    if not think or not search:
        return None
    reasoning, query = think.group(1).strip(), search.group(1).strip()
    if not reasoning or not query or "<" in query or "\n" in query:
        return None
    return reasoning, query


def _contains_phrase(text: str, phrase: str) -> bool:
    haystack, needle = normalize(text), normalize(phrase)
    return bool(needle and f" {needle} " in f" {haystack} ")


def leaks_answer(text: str, answer: str, visible_context: str = "") -> bool:
    """Reject newly revealed answers and explicit answer claims.

    Entity answers already written in the question may legitimately be used in
    a search query.  They are not privileged leakage, but asserting that such
    an entity is the correct/final answer still is.
    """
    if not _contains_phrase(text, answer):
        return False
    if not _contains_phrase(visible_context, answer):
        return True
    normalized = normalize(text)
    answer_words = normalize(answer)
    claim_patterns = (
        rf"(?:correct|final) (?:answer|title|person|film|place) (?:is|was) {re.escape(answer_words)}",
        rf"answer (?:is|was) {re.escape(answer_words)}",
        rf"therefore {re.escape(answer_words)} (?:is|was)",
    )
    return any(re.search(pattern, normalized) for pattern in claim_patterns)


def document_title(item: Any) -> str:
    if isinstance(item, Mapping):
        document = item.get("document", item)
        if isinstance(document, Mapping):
            content = document.get("contents", document.get("content", ""))
            title = document.get("title")
            if title:
                return str(title).strip()
        else:
            content = document
    else:
        content = item
    return str(content or "").splitlines()[0].strip()


def score_retrieval(result: Any, remaining_gold: set[str]) -> dict[str, float]:
    titles = [normalize(document_title(item)) for item in (result or [])]
    ranks = [rank for rank, title in enumerate(titles, 1) if title in remaining_gold]
    hits = {title for title in titles if title in remaining_gold}
    return {"hit": float(bool(ranks)), "mrr": 1.0 / min(ranks) if ranks else 0.0,
            "recall": len(hits) / max(1, len(remaining_gold))}


def retrieve_batch(url: str, queries: Sequence[str], topk: int, timeout: float) -> list[Any]:
    import requests
    response = requests.post(url, json={"queries": list(queries), "topk": topk,
                                        "return_scores": True}, timeout=timeout)
    response.raise_for_status(); payload = response.json()
    results = payload.get("result") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or len(results) != len(queries):
        raise RuntimeError(f"retriever returned invalid result shape for {len(queries)} queries")
    return results


def correction_instruction(event: dict[str, Any], sheet: str) -> str:
    bad_think, bad_query = event["bad"]
    bad_think = redact_answer(bad_think, event.get("answer", ""))
    bad_query = redact_answer(bad_query, event.get("answer", ""))
    return (
        event["prefix"]
        + "\n<student_attempt>\n"
        + f"<think>{bad_think}</think>\n<search>{bad_query}</search>\n"
        + "</student_attempt>\n"
        + sheet
        + "The student's attempted reasoning/search may be wrong. Correct its direction using only "
          "the visible state and any permitted hindsight above. Do not state or imply the final answer. "
          "Return exactly one corrected action as <think>...</think><search>...</search>.\n"
    )


def anchored_prefix(tokenizer, prompt: str, state: str, budget: int) -> str:
    """Retain the question-bearing prompt tail and the most recent search state."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    state_ids = tokenizer.encode(state, add_special_tokens=False)
    if len(prompt_ids) >= budget:
        return tokenizer.decode(prompt_ids[-budget:], skip_special_tokens=False)
    keep_state = budget - len(prompt_ids)
    ids = prompt_ids + (state_ids[-keep_state:] if keep_state else [])
    return tokenizer.decode(ids, skip_special_tokens=False)


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], device: str, batch_size: int,
             max_context: int, max_new_tokens: int) -> list[str]:
    outputs = []
    tokenizer.padding_side = "left"; tokenizer.truncation_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_context - max_new_tokens).to(device)
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
        prompt_width = encoded.input_ids.shape[1]
        outputs.extend(tokenizer.decode(row[prompt_width:], skip_special_tokens=False) for row in generated)
    return outputs


def paired_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    summary = {}
    for arm in ("student",) + ARMS:
        values = [row["scores"][arm][metric] for row in rows if arm in row["scores"]]
        summary[arm] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
    for baseline in ("student", "no_cheat", "shuffle"):
        deltas = [row["scores"]["real"][metric] - row["scores"][baseline][metric] for row in rows]
        summary[f"real_minus_{baseline}"] = {
            "n": len(deltas), "mean": sum(deltas) / len(deltas) if deltas else None,
            "ci95": bootstrap_ci(deltas, seed=47),
        }
    return summary


def gate(rows: list[dict[str, Any]], min_hit_lift: float,
         valid_rates: dict[str, float] | None = None,
         leak_rates: dict[str, float] | None = None,
         min_valid_rate: float = .8, max_leak_rate: float = .05) -> dict[str, Any]:
    hit = paired_summary(rows, "hit"); mrr = paired_summary(rows, "mrr")
    hit_checks = [hit[f"real_minus_{base}"]["mean"] >= min_hit_lift
                  for base in ("student", "no_cheat", "shuffle")]
    mrr_checks = [mrr[f"real_minus_{base}"]["mean"] > 0
                  for base in ("student", "no_cheat", "shuffle")]
    format_check = valid_rates is None or valid_rates.get("real", 0.0) >= min_valid_rate
    leak_check = leak_rates is None or leak_rates.get("real", 1.0) <= max_leak_rate
    return {"gate_pass": all(hit_checks) and all(mrr_checks) and format_check and leak_check,
            "min_hit_lift": min_hit_lift, "min_valid_rate": min_valid_rate,
            "max_answer_leak_rate": max_leak_rate, "valid_rate_check": format_check,
            "answer_leak_check": leak_check,
            "hit_checks": hit_checks, "mrr_checks": mrr_checks,
            "note": "real hindsight must improve hit rate by the configured amount and MRR over all controls"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trees", nargs="+", required=True); p.add_argument("--selected", nargs="+", required=True)
    p.add_argument("--model", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parquet", default="data/multihopqa_search_mixed_402020_20260830/train.parquet")
    p.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--retrieval-topk", type=int, default=3); p.add_argument("--retrieval-timeout", type=float, default=90)
    p.add_argument("--max-events", type=int, default=64); p.add_argument("--min-value-gap", type=float, default=.25)
    p.add_argument("--max-evidence-chars", type=int, default=1600); p.add_argument("--max-context", type=int, default=1536)
    p.add_argument("--max-new-tokens", type=int, default=160); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--min-hit-lift", type=float, default=.05); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--min-valid-rate", type=float, default=.8)
    p.add_argument("--max-answer-leak-rate", type=float, default=.05)
    p.add_argument("--device", default="cuda:0"); args = p.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tree_paths = sorted(path for pattern in args.trees for path in glob.glob(pattern))
    records = [row for path in tree_paths for row in load_jsonl(Path(path))]
    metadata = load_tree_metadata(args.selected, args.parquet, tokenizer)

    # Add source/index/gold fields to the common prompt/answer metadata.
    raw_cache = {source: load_jsonl(Path(path)) for source, path in DEFAULT_RAW.items() if Path(path).exists()}
    for pattern in args.selected:
        for path in glob.glob(pattern):
            for row in load_jsonl(Path(path)):
                tree_uid, source, uid = str(row.get("tree_uid", "")), str(row.get("data_source", "")), str(row.get("uid", ""))
                if tree_uid not in metadata: continue
                try: index = int(uid.rsplit(":", 1)[1])
                except (ValueError, IndexError): continue
                raw = raw_cache.get(source, [])
                if index >= len(raw): continue
                titles, _ = gold_fields(raw[index], source)
                metadata[tree_uid].update(source=source, index=index, gold_titles=titles)

    events = collect_events(records, args.min_value_gap, args.max_evidence_chars)
    usable = []
    for event in events:
        meta = metadata.get(event["tree_uid"], {})
        gold = {normalize(title) for title in meta.get("gold_titles", []) if title}
        visible = {normalize(title.strip().strip('"“”')) for title in TITLE_RE.findall(event["prefix"])}
        remaining = gold - visible
        if not meta.get("prompt") or not remaining: continue
        event["prefix"] = anchored_prefix(tokenizer, meta["prompt"], event["prefix"],
                                           max(256, args.max_context - args.max_new_tokens - 420))
        event["answer"] = meta.get("answer", ""); event["remaining_gold"] = remaining
        event["evidence"] = redact_answer(event["evidence"], event["answer"])
        event["alternatives"] = [(redact_answer(query, event["answer"]), value)
                                 for query, value in event["alternatives"]]
        usable.append(event)
    random.Random(args.seed).shuffle(usable); events = usable[:args.max_events]
    if len(events) < 2: p.error(f"only {len(events)} usable correction events")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    sheets = [render_hindsight(event) for event in events]
    generations: dict[str, list[str]] = {}
    for arm in ARMS:
        prompts = [correction_instruction(event, "" if arm == "no_cheat" else
                   (sheets[i] if arm == "real" else
                    redact_answer(sheets[(i + 1) % len(events)], event["answer"])))
                   for i, event in enumerate(events)]
        generations[arm] = generate(model, tokenizer, prompts, args.device, args.batch_size,
                                    args.max_context, args.max_new_tokens)
    del model; torch.cuda.empty_cache()

    query_jobs: list[tuple[int, str, str]] = []
    rows = [{"event": i, "tree_uid": event["tree_uid"], "parent_uid": event["parent_uid"],
             "remaining_gold_titles": sorted(event["remaining_gold"]), "generations": {}, "scores": {}}
            for i, event in enumerate(events)]
    for i, event in enumerate(events):
        query_jobs.append((i, "student", event["bad"][1]))
        for arm in ARMS:
            raw = generations[arm][i]; parsed = parse_correction(raw)
            answer_leak = leaks_answer(raw, event["answer"], event["prefix"])
            valid = parsed is not None and not answer_leak
            rows[i]["generations"][arm] = {"raw": raw, "valid": valid,
                                                   "think": parsed[0] if valid else None,
                                                   "query": parsed[1] if valid else None,
                                                   "answer_leak": answer_leak}
            if valid: query_jobs.append((i, arm, parsed[1]))
    results = retrieve_batch(args.retrieval_url, [job[2] for job in query_jobs],
                             args.retrieval_topk, args.retrieval_timeout)
    for (event_index, arm, query), result in zip(query_jobs, results):
        rows[event_index]["scores"][arm] = score_retrieval(result, events[event_index]["remaining_gold"])
        rows[event_index]["scores"][arm]["query"] = query
    # Invalid or answer-leaking teacher actions are operational failures, not
    # missing observations. Scoring them as zero prevents survivor bias.
    for row in rows:
        for arm in ARMS:
            if arm not in row["scores"]:
                row["scores"][arm] = {"hit": 0.0, "mrr": 0.0, "recall": 0.0,
                                      "query": None, "rejected": True}
    complete = rows
    valid_rates = {arm: sum(row["generations"][arm]["valid"] for row in rows) / len(rows)
                   for arm in ARMS}
    leak_rates = {arm: sum(row["generations"][arm]["answer_leak"] for row in rows) / len(rows)
                  for arm in ARMS}
    summary = {"protocol": "generated correction + executed retrieval qualification",
               "model": args.model, "events_requested": len(events), "events_complete": len(complete),
               "valid_generation_rate": valid_rates,
               "answer_leak_rate": leak_rates,
               "metrics": {metric: paired_summary(complete, metric) for metric in ("hit", "mrr", "recall")},
               "verdict": gate(complete, args.min_hit_lift, valid_rates, leak_rates,
                               args.min_valid_rate, args.max_answer_leak_rate)
                          if complete else {"gate_pass": False, "note": "no complete events"},
               "per_event": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("events_requested", "events_complete", "valid_generation_rate",
                                                    "answer_leak_rate", "metrics", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
