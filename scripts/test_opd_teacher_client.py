#!/usr/bin/env python3
"""Protocol self-check for the OPD teacher server.

Builds two realistic teacher contexts (prompt + prefix + gold-answer cheat +
<search> + query) with the student tokenizer, then verifies that the server:

1. returns normalized log-probs (logsumexp == 0) for every row;
2. is invariant to batch composition and padding (a row sent alone must match
   the same row sent alongside a different-length row) -- this exercises the
   left-padding, position_ids, and tail-window gather logic;
3. reproduces a directly-computed local reference forward on GPU 1 for the
   same row (teacher on GPU 0), within float tolerance;
4. applies temperature (T=2 result differs from T=1).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request

import numpy as np
import torch
import transformers

sys.path.insert(0, "/home/luwa/Documents/Tree-GRPO")
from search_r1.llm_agent.self_opd import _render_answer_cheat  # noqa: E402


def call_server(url, temperature, rows):
    header = {
        "temperature": float(temperature),
        "rows": [
            {"len": len(ids), "query_positions": [int(p) for p in positions]}
            for ids, positions in rows
        ],
    }
    header_bytes = json.dumps(header).encode("utf-8")
    ids_flat = np.concatenate([np.asarray(ids, dtype=np.int64) for ids, _ in rows])
    body = struct.pack("<I", len(header_bytes)) + header_bytes + ids_flat.tobytes()
    request = urllib.request.Request(
        url.rstrip("/") + "/score", data=body,
        headers={"Content-Type": "application/octet-stream"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=600.0) as response:
        raw = response.read()
    (header_len,) = struct.unpack("<I", raw[:4])
    response_header = json.loads(raw[4:4 + header_len])
    vocab = int(response_header["vocab"])
    payload = np.frombuffer(raw, dtype=np.float32, offset=4 + header_len)
    results, cursor = [], 0
    for row_length in response_header["row_lengths"]:
        count = int(row_length) * vocab
        results.append(payload[cursor:cursor + count].reshape(int(row_length), vocab).copy())
        cursor += count
    assert cursor == payload.size, "response length mismatch"
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8126")
    parser.add_argument("--teacher-model", default="/home/luwa/Documents/Tree-GRPO/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--student-model",
                        default="/home/luwa/Documents/Tree-GRPO/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350")
    parser.add_argument("--reference-device", default="cuda:1")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.student_model)
    student_config = transformers.AutoConfig.from_pretrained(args.student_model)
    assert int(student_config.vocab_size) == 151936, student_config.vocab_size

    think_open = "<" + "think" + ">"
    think_close = "</" + "think" + ">"
    prefix_text = (
        "Answer the given question. You must conduct reasoning inside " + think_open + " and " + think_close + " "
        "first every time you get new information. After reasoning, if you find you lack some "
        "knowledge, you can call a search engine by <search> query </search> and it will return "
        "the top searched results between <information> and </information>. You can search as "
        "many times as your want. If you find no further external knowledge needed, you can "
        "directly provide the answer inside <answer> and </answer>, without detailed "
        "illustrations. For example, <answer> Beijing </answer>. Question: What country of "
        "origin does Susan Dalian and Storm have in common?"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prefix_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    row_a_text = (
        prompt_text
        + "\n\n" + think_open + "\nTo answer, I need to identify both people and their countries.\n" + think_close + "\n"
        + _render_answer_cheat("United States")
    )
    row_b_text = (
        prompt_text
        + "\n\n" + think_open + "\nI need the nationality of the director of Titanic.\n" + think_close + "\n"
        + _render_answer_cheat("James Cameron")
    )
    query_a = "Susan Dalian nationality"
    query_b = "director of the film Titanic country of birth"

    def build_row(context_text, query_text):
        # Separate tokenization at the <search>|query boundary, exactly like
        # the audited score_arm: BPE merges such as ">T" otherwise drop events.
        prefix_ids = tokenizer.encode(context_text, add_special_tokens=False)
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)
        ids = prefix_ids + query_ids
        positions = list(range(len(prefix_ids) - 1, len(ids) - 1))
        return ids, positions

    ids_a, pos_a = build_row(row_a_text, query_a)
    ids_b, pos_b = build_row(row_b_text, query_b)
    print(f"row A: len={len(ids_a)} query_tokens={len(pos_a)}")
    print(f"row B: len={len(ids_b)} query_tokens={len(pos_b)}")

    # 1 & 2: alone vs batched together; plus normalization.
    solo_a = call_server(args.teacher_url, 1.0, [(ids_a, pos_a)])[0]
    solo_b = call_server(args.teacher_url, 1.0, [(ids_b, pos_b)])[0]
    batched = call_server(args.teacher_url, 1.0, [(ids_a, pos_a), (ids_b, pos_b)])
    for name, result in (("A", solo_a), ("B", solo_b)):
        logsumexp = np.logaddexp.reduce(result, axis=1)
        assert np.allclose(logsumexp, 0.0, atol=1e-3), f"row {name} not normalized"
    pad_diff_a = np.abs(batched[0] - solo_a)
    pad_diff_b = np.abs(batched[1] - solo_b)
    print(
        f"padding invariance: A max={pad_diff_a.max():.3e} mean={pad_diff_a.mean():.3e}; "
        f"B max={pad_diff_b.max():.3e} mean={pad_diff_b.mean():.3e}"
    )
    # bf16 forwards across different batch shapes legitimately differ by one
    # logit quantum (~0.06 at |logit|~15); only gross misalignment (a wrong
    # position or padding leak) would exceed these bounds.
    assert pad_diff_a.mean() < 1e-1 and pad_diff_b.mean() < 1e-1, "batch composition changed results"
    assert pad_diff_a.max() < 1.5 and pad_diff_b.max() < 1.5, "batch composition changed results"
    argmax_agree = float(
        (batched[0].argmax(-1) == solo_a.argmax(-1)).mean()
    )
    print(f"padding invariance: argmax agreement A={argmax_agree:.3f}")
    assert argmax_agree >= 0.75, "batch composition changed the predicted tokens"

    # 4: temperature changes the distribution.
    temp2 = call_server(args.teacher_url, 2.0, [(ids_a, pos_a)])[0]
    temp_diff = np.abs(temp2 - solo_a).max()
    print(f"temperature sensitivity: max|diff|={temp_diff:.3e}")
    assert temp_diff > 1e-4, "temperature had no effect"

    # 3: local reference forward with the same weights on another device.
    print(f"loading reference teacher on {args.reference_device} ...")
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.reference_device).eval()
    input_ids = torch.tensor([ids_a], dtype=torch.long, device=args.reference_device)
    attention = torch.ones_like(input_ids)
    position_ids = (attention.cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        logits = reference(
            input_ids=input_ids, attention_mask=attention,
            position_ids=position_ids, use_cache=False,
        ).logits[0]
    positions = torch.tensor(pos_a, dtype=torch.long, device=args.reference_device)
    targets = torch.tensor(ids_a, dtype=torch.long, device=args.reference_device)[positions + 1]
    reference_log_probs = torch.log_softmax(logits[positions].float(), dim=-1)
    reference_target = reference_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    server_target = solo_a[
        np.arange(len(pos_a)), targets.detach().cpu().numpy()
    ]
    ref_np = reference_target.detach().cpu().numpy()
    max_err = np.abs(server_target - ref_np).max()
    mean_lp = float(server_target.mean())
    print(f"reference agreement: max|diff|={max_err:.3e} mean_query_logprob={mean_lp:.4f}")
    # bf16 cross-device tolerance; the effect we are chasing is ~0.3/token.
    assert max_err < 1.5e-1, "server disagrees with local reference forward"
    assert (solo_a.argmax(-1) == reference_log_probs.argmax(-1).detach().cpu().numpy()).all(), \
        "server argmax tokens differ from reference"

    print("TEACHER_SERVER_SELF_CHECK_OK")


if __name__ == "__main__":
    main()
