#!/usr/bin/env python3
"""Generate node skills with a local SFT model on held-out branches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_node_skill_consumption_sft import teacher_prompt
from data_process.build_tree_seed_analyzer_sft import extract_json_object, extract_titles


FIELDS = {"failure_type", "missing_relation", "next_operation", "stop_condition"}


def read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(parsed: dict[str, Any], event: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if set(parsed) != FIELDS:
        reasons.append("schema")
    combined = "\n".join(str(parsed.get(k, "")).strip() for k in sorted(FIELDS))
    if any(not str(parsed.get(k, "")).strip() for k in FIELDS):
        reasons.append("empty")
    if not 14 <= len(combined.split()) <= 120:
        reasons.append("length")
    if "[redacted]" in combined.casefold():
        reasons.append("redaction_artifact")
    # Node skills must remain answer-agnostic: do not copy dates, years,
    # counts, or other numeric identifiers from the privileged branch state.
    if re.search(r"\b(?:19|20)\d{2}\b|\b\d{1,4}\b", combined):
        reasons.append("numeric_fact_copy")
    if re.search(r"\b(?:reward|score|gold|branch|hindsight|better action|correct answer)\b", combined, re.I):
        reasons.append("privileged_meta")
    for field in ("good_query", "bad_query"):
        query = str(event.get(field, ""))
        if len(query.split()) >= 3 and query.casefold() in combined.casefold():
            reasons.append(f"{field}_copy")
    for title in extract_titles(str(event.get("better_evidence", ""))):
        if len(title.split()) >= 2 and title.casefold() in combined.casefold():
            reasons.append("title_copy")
    return sorted(set(reasons))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True); p.add_argument("--consumer", required=True)
    p.add_argument("--prepared", required=True); p.add_argument("--train-split", required=True)
    p.add_argument("--validation-split", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--max-input", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=256); args = p.parse_args()
    prepared = {r["event_id"]: r for r in read(args.prepared)}
    val_uids = {r["uid"] for r in read(args.validation_split)}
    train_uids = {r["uid"] for r in read(args.train_split)}
    rows = []
    for row in read(args.consumer):
        if row.get("uid") not in val_uids or float(row.get("value_gap", 0)) < .1:
            continue
        if "[redacted]" in str(row.get("response", "")).casefold():
            continue
        source = prepared.get(row["event_id"])
        if source is None:
            continue
        event = dict(source)
        event["better_titles"] = extract_titles(str(event.get("better_evidence", "")))
        event["aliases"] = []
        rows.append((row, event))
    rows.sort(key=lambda pair: pair[0]["event_id"])
    if train_uids & val_uids:
        raise ValueError("train/validation UID leakage")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    output_rows = []
    for row, event in rows:
        prompt = teacher_prompt(event)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_input).to(args.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False,
                                       max_new_tokens=args.max_new_tokens,
                                       eos_token_id=tokenizer.eos_token_id,
                                       pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        parsed = None
        reasons = ["parse"]
        try:
            parsed = extract_json_object(text)
            reasons = validate(parsed, event)
        except Exception:
            pass
        out = dict(row)
        out.update({"sft_raw_response": text, "sft_node_skill": parsed if not reasons else None,
                    "sft_rejection_reasons": reasons, "split": "heldout"})
        output_rows.append(out)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in output_rows) + "\n", encoding="utf-8")
    accepted = sum(r["sft_node_skill"] is not None for r in output_rows)
    summary = {"model": args.model, "heldout_events": len(output_rows), "accepted": accepted,
               "acceptance_rate": accepted / max(1, len(output_rows))}
    (out_path.parent / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
