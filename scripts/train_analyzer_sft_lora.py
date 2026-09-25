#!/usr/bin/env python3
"""Train a small LoRA Analyzer on audited prompt -> skill JSON pairs.

The prompt tokens are masked from the loss.  Only the audited Analyzer
response is supervised; no answer-bearing source fields are loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_row(row: dict[str, Any]) -> None:
    if set(row) & {"ground_truth", "answer", "answers", "raw_response"}:
        raise ValueError(f"sensitive field in SFT export: {row.get('sample_id')}")
    task_type = row.get("task_type", "analyzer_generation")
    if task_type == "analyzer_generation":
        parsed = json.loads(row["response"])
        if set(parsed) != {"episode_summary", "episode_skill"}:
            raise ValueError(f"invalid Analyzer response: {row.get('sample_id')}")
    elif task_type == "node_skill_generation":
        parsed = json.loads(row["response"])
        if set(parsed) != {"failure_type", "missing_relation", "next_operation", "stop_condition"}:
            raise ValueError(f"invalid node skill response: {row.get('event_id')}")
        if any(not str(parsed[key]).strip() for key in parsed):
            raise ValueError(f"empty node skill field: {row.get('event_id')}")
        if "[redacted]" in row["response"].casefold():
            raise ValueError(f"redacted token in node skill target: {row.get('event_id')}")
    elif task_type in {"skill_consumer", "action_replay"}:
        if not re.fullmatch(r"<think>.+?</think>\n<search>[^\n<>]+</search>", row["response"], re.DOTALL):
            raise ValueError(f"invalid action response: {row.get('event_id')}")
        if "[redacted]" in row["response"].casefold():
            raise ValueError(f"redacted token in action target: {row.get('event_id')}")
    else:
        raise ValueError(f"unknown task_type={task_type!r}")


def encode_row(row: dict[str, Any], tokenizer, max_length: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
    response_ids = tokenizer.encode("\n" + row["response"] + tokenizer.eos_token, add_special_tokens=False)
    keep_prompt = max_length - len(response_ids)
    if keep_prompt < 256:
        raise ValueError(f"response too long for max_length={max_length}: {row.get('sample_id')}")
    if len(prompt_ids) > keep_prompt:
        # Keep the instruction/task prefix and the most recent trajectory tail.
        head = min(768, keep_prompt // 2)
        prompt_ids = prompt_ids[:head] + prompt_ids[-(keep_prompt - head):]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            n = len(row["input_ids"])
            input_ids[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
            attention[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


@torch.no_grad()
def eval_loss(model, loader, device, amp_dtype) -> float:
    model.eval(); total = 0.0; count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = model(**batch).loss
        total += float(loss.detach()) * batch["input_ids"].shape[0]
        count += batch["input_ids"].shape[0]
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Analyzer-SFT run")
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    train_rows, val_rows = read_jsonl(Path(args.train)), read_jsonl(Path(args.validation))
    for row in train_rows + val_rows:
        validate_row(row)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_data = [encode_row(row, tokenizer, args.max_length) for row in train_rows]
    val_data = [encode_row(row, tokenizer, args.max_length) for row in val_rows]
    collator = Collator(tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=amp_dtype, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, round(total_updates * args.warmup_ratio)), num_training_steps=total_updates
    )
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.jsonl"
    initial_val = eval_loss(model, val_loader, device, amp_dtype)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event": "initial", "validation_loss": initial_val}) + "\n")
        model.train(); optimizer.zero_grad(set_to_none=True); update = 0
        for epoch in range(args.epochs):
            running = 0.0
            for step, batch in enumerate(train_loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = model(**batch).loss / args.grad_accum
                loss.backward(); running += float(loss.detach())
                if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); update += 1
                    if update % 10 == 0 or update == total_updates:
                        val = eval_loss(model, val_loader, device, amp_dtype)
                        model.train()
                        record = {"event": "update", "epoch": epoch, "update": update,
                                  "train_loss_accum": running, "validation_loss": val,
                                  "learning_rate": scheduler.get_last_lr()[0]}
                        log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record), flush=True); running = 0.0
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    # Save a standalone merged checkpoint for the existing likelihood audit.
    merged_dir = output / "merged"
    merged = model.merge_and_unload(); merged.save_pretrained(merged_dir, safe_serialization=True); tokenizer.save_pretrained(merged_dir)
    manifest = {"base_model": args.model, "train_examples": len(train_rows), "validation_examples": len(val_rows),
                "epochs": args.epochs, "max_length": args.max_length, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum, "learning_rate": args.learning_rate,
                "initial_validation_loss": initial_val, "adapter_dir": str(adapter_dir), "merged_dir": str(merged_dir)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
