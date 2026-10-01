#!/usr/bin/env python3
"""Hybrid rescue: simple 2-hop teacher prefix (no gold answer) + forced handoff.

Reuses the cached teacher prefixes from the new-scheme run and appends the old
scheme's forced-closing instruction ("Do not search again ... answer now"),
testing whether high coverage + forced synthesis beats either parent scheme.
Same 33 real all-wrong groups, same step-12 student, 6 temperature-1.0
suffixes with identical seeds to the new-scheme prefix arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import evaluate_rescue_rate_allwrong as rrr  # noqa: E402

# Same forced-closing wording as teacher_rescue.py (minus the teacher plan's
# free-form handoff, which the simple scout does not produce).
HANDOFF = ("<think>The retrieved evidence fully resolves every required relation. "
            "Do not search again. Synthesize the exact entity or value requested "
            "and answer now.</think>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path,
                        default=ROOT / "rollouts/tracecredit-opid-flash-semantic-rescue-sft350-step0to20-20260929")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "verl_checkpoints/tracecredit-opid-flash-semantic-rescue-sft350-step0to20-20260929/actor/global_step_12/hf_merged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path,
                        default=ROOT / "eval_outputs/rescue_rate_allwrong_33.teacher_cache.jsonl")
    parser.add_argument("--new-scheme-output", type=Path,
                        default=ROOT / "eval_outputs/rescue_rate_allwrong_33.jsonl")
    parser.add_argument("--old-scheme-output", type=Path,
                        default=ROOT / "eval_outputs/rescue_rate_oldscheme_33.jsonl")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8002/retrieve")
    parser.add_argument("--min-step", type=int, default=9)
    parser.add_argument("--max-step", type=int, default=12)
    parser.add_argument("--max-groups", type=int, default=40)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--teacher-hops", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--groups-per-wave", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import math
    from transformers import AutoTokenizer
    from vllm import LLM

    groups = rrr.load_all_wrong_groups(args.rollout_dir, args.min_step, args.max_step)
    by_source = defaultdict(list)
    for group in groups:
        by_source[group["data_source"]].append(group)
    cap = max(1, math.ceil(args.max_groups / max(1, len(by_source))))
    selected = []
    for source, items in sorted(by_source.items()):
        selected.extend(items[:min(cap, len(items))])
    selected = selected[:args.max_groups]

    prefixes = {}
    for line in args.teacher_cache.open():
        cached = json.loads(line)
        if cached["result"].get("ok"):
            prefixes[cached["uid"]] = cached["result"]
    missing = [g["uid"] for g in selected if g["uid"] not in prefixes]
    print(f"selected={len(selected)} cached_prefixes={len(selected) - len(missing)} "
          f"missing={len(missing)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    llm = LLM(model=str(args.model), dtype="half", tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=True, disable_log_stats=True)

    import os
    from verl.utils.reward_score.qa_em_format import em_check
    student_budget = args.max_turns - args.teacher_hops
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sink = args.output.open("a")
    for wave_start in range(0, len(selected), args.groups_per_wave):
        wave = selected[wave_start:wave_start + args.groups_per_wave]
        states = []
        for gi, group in enumerate(wave):
            prefix = prefixes.get(group["uid"])
            if prefix is None:
                continue
            for si in range(args.samples):
                # identical seed convention to the new-scheme prefix arm
                states.append({"uid": group["uid"], "arm": "hybrid",
                               "seed": 500 + (wave_start + gi) * 100 + si,
                               "text": group["prompt"] + prefix["blocks"] + HANDOFF,
                               "searches_left": student_budget,
                               "done": False, "steps": []})
        rrr.run_arm_states(states, args, llm, tokenizer)
        by = defaultdict(list)
        for s in states:
            by[s["uid"]].append(s)
        for group in wave:
            if group["uid"] not in prefixes:
                continue
            trajs = []
            for s in by.get(group["uid"], []):
                answer = s.get("final_answer")
                trajs.append({"answer": answer,
                              "em": bool(answer is not None and em_check(answer, group["answers"])),
                              "searches": args.teacher_hops + student_budget - s["searches_left"]})
            record = {"uid": group["uid"], "data_source": group["data_source"],
                      "question": group["question"], "answers": group["answers"],
                      "hybrid": trajs}
            results.append(record)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
        print(f"wave at {wave_start} done total={len(results)}", flush=True)
    sink.close()

    all_rows = [json.loads(l) for l in args.output.open()]

    def rate(rows, key):
        k = sum(any(t["em"] for t in r[key]) for r in rows)
        return k, len(rows)

    print("\n=== HYBRID (2-hop no-answer prefix + forced handoff) ===", flush=True)
    for source in ["all"] + sorted({r["data_source"] for r in all_rows}):
        rows = all_rows if source == "all" else [r for r in all_rows if r["data_source"] == source]
        k, n = rate(rows, "hybrid")
        lo, hi = rrr.wilson_ci(k, n)
        traj = [t for r in rows for t in r["hybrid"]]
        print(f"[{source}] n={n} hybrid RescueRate@6={k}/{n}={k/max(n,1):.3f} "
              f"(CI {lo:.3f}-{hi:.3f}) | per-traj EM={sum(t['em'] for t in traj)}/{len(traj)}"
              f"={sum(t['em'] for t in traj)/max(len(traj),1):.3f}", flush=True)

    new = {r["uid"]: r for r in (json.loads(l) for l in args.new_scheme_output.open())}
    old = {r["uid"]: r for r in (json.loads(l) for l in args.old_scheme_output.open())}
    print("\n=== THREE-WAY on identical groups ===", flush=True)
    n = len(all_rows)
    hk = sum(any(t["em"] for t in r["hybrid"]) for r in all_rows)
    nk = sum(new.get(r["uid"], {}).get("teacher_ok")
             and any(t["em"] for t in new[r["uid"]]["prefix"]) for r in all_rows)
    ok6 = sum(old.get(r["uid"], {}).get("old_ok")
              and any(t["em"] for t in old[r["uid"]]["old"]) for r in all_rows)
    print(f"n={n} | HYBRID {hk}/{n}={hk/n:.3f} | NEW {nk}/{n}={nk/n:.3f} | OLD(gated) {ok6}/{n}"
          f"={ok6/n:.3f}", flush=True)
    h_only_vs_new = sum(any(t["em"] for t in r["hybrid"])
                        and not (new.get(r["uid"], {}).get("teacher_ok")
                                 and any(t["em"] for t in new[r["uid"]]["prefix"]))
                        for r in all_rows)
    new_only_vs_h = sum((new.get(r["uid"], {}).get("teacher_ok")
                         and any(t["em"] for t in new[r["uid"]]["prefix"]))
                        and not any(t["em"] for t in r["hybrid"]) for r in all_rows)
    print(f"hybrid_beats_new={h_only_vs_new} new_beats_hybrid={new_only_vs_h}", flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
