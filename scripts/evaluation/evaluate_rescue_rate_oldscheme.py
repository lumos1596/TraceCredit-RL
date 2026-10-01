#!/usr/bin/env python3
"""Head-to-head: OLD rescue scheme (the one running in training) vs new.

Faithfully replays search_r1/llm_agent/teacher_rescue.py:
  seed = student's own think prefix before its first (failed) search
         + 1-2 teacher <search> actions executed against the real retriever
         + forced handoff ("Do not search again ... answer now")
  then the frozen student continues.  The failed query/observation are sent to
  the teacher service for planning only; they are NOT re-inserted into the
  student seed (identical to training).

For each real all-wrong group we sample 6 temperature-1.0 suffixes and report
RescueRate@6 plus the single-suffix success that training actually harvests.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import evaluate_direct_prefix_teacher as dpt  # noqa: E402
import evaluate_rescue_rate_allwrong as rrr  # noqa: E402
from search_r1.llm_agent.teacher_rescue import request_rescue_plan  # noqa: E402

BRIDGE = ("<think>The preceding evidence leaves another required relation unresolved, "
           "so I will retrieve that specific relation.</think>\n")


def first_row_for(rollout_dir: Path, uid: str, step: int, chunk: int) -> dict | None:
    path = rollout_dir / f"step_{step:06d}_chunk_{chunk:02d}_selected.jsonl"
    for line in open(path):
        row = json.loads(line)
        if row["uid"] == uid:
            return row
    return None


def build_old_seed(group: dict, first_row: dict, args, tokenizer) -> dict:
    """Exact seed construction from teacher_rescue.py lines 122-178."""
    response = first_row["response"]
    match = re.search(r"<search>(.*?)</search>", response, flags=re.S)
    if match is None:
        return {"ok": False, "error": "failed trajectory has no search action"}
    prefix = response[:match.start()]
    if len(prefix) > 3000:
        return {"ok": False, "error": "student prefix exceeds safe first-action bound"}
    failed_query = match.group(1).strip()
    obs_match = re.search(r"<information>(.*?)</information>", response[match.end():], flags=re.S)
    failed_observation = obs_match.group(0) if obs_match else ""
    aliases = [str(a) for a in group["answers"] if str(a or "")]
    if not aliases:
        return {"ok": False, "error": "gold answer unavailable"}

    plan = request_rescue_plan(
        args.teacher_rescue_url, uid=group["uid"], question=first_row["prompt"],
        prefix=prefix, failed_query=failed_query,
        failed_observation=failed_observation, answers=aliases)

    blocks = prefix
    for action_index, query in enumerate(plan["queries"]):
        if action_index:
            blocks += BRIDGE
        docs = dpt.retrieve(args.retriever_url, query)
        blocks += f"<search>{query}</search>"
        blocks += dpt.observation(docs, tokenizer)
    handoff = ("<think>The retrieved evidence fully resolves every required relation. "
                "Do not search again. Synthesize the exact entity or value requested "
                "and answer now. " + plan["handoff"] + "</think>\n")
    blocks += handoff
    return {"ok": True, "blocks": first_row["prompt"] + blocks,
            "queries": plan["queries"], "handoff": plan["handoff"],
            "prefix_chars": len(prefix), "failed_query": failed_query}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path,
                        default=ROOT / "rollouts/tracecredit-opid-flash-semantic-rescue-sft350-step0to20-20260929")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "verl_checkpoints/tracecredit-opid-flash-semantic-rescue-sft350-step0to20-20260929/actor/global_step_12/hf_merged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-scheme-output", type=Path,
                        default=ROOT / "eval_outputs/rescue_rate_allwrong_33.jsonl")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8002/retrieve")
    parser.add_argument("--teacher-rescue-url", default="http://127.0.0.1:8130")
    parser.add_argument("--min-step", type=int, default=9)
    parser.add_argument("--max-step", type=int, default=12)
    parser.add_argument("--max-groups", type=int, default=40)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--groups-per-wave", type=int, default=11)
    parser.add_argument("--teacher-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    # identical group selection as the new-scheme run
    groups = rrr.load_all_wrong_groups(args.rollout_dir, args.min_step, args.max_step)
    by_source = defaultdict(list)
    for group in groups:
        by_source[group["data_source"]].append(group)
    import math
    cap = max(1, math.ceil(args.max_groups / max(1, len(by_source))))
    selected = []
    for source, items in sorted(by_source.items()):
        selected.extend(items[:min(cap, len(items))])
    selected = selected[:args.max_groups]
    print(f"selected={len(selected)} groups (identical ordering to new scheme)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)

    # ---- old-scheme teacher plans (real /rescue endpoint + gate) ----
    seeds: dict[str, dict] = {}
    def call(group):
        first_row = first_row_for(args.rollout_dir, group["uid"], group["step"], group["chunk"])
        if first_row is None:
            return {"ok": False, "error": "source row not found"}
        try:
            return build_old_seed(group, first_row, args, tokenizer)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.teacher_workers) as pool:
        futures = {pool.submit(call, g): g["uid"] for g in selected}
        for done, future in enumerate(as_completed(futures), 1):
            seeds[futures[future]] = future.result()
            if done % 8 == 0 or done == len(futures):
                ok = sum(v["ok"] for v in seeds.values())
                print(f"old-rescue {done}/{len(futures)} gate_ok={ok}", flush=True)
    for uid, value in seeds.items():
        if not value["ok"]:
            print(f"[old gate failed] {uid}: {value['error'][:140]}", flush=True)

    llm = LLM(model=str(args.model), dtype="half", tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=True, disable_log_stats=True)

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sink = args.output.open("a")
    for wave_start in range(0, len(selected), args.groups_per_wave):
        wave = selected[wave_start:wave_start + args.groups_per_wave]
        states = []
        for gi, group in enumerate(wave):
            seed = seeds.get(group["uid"])
            if not seed or not seed["ok"]:
                continue
            remaining = args.max_turns - len(seed["queries"])  # depth already used
            for si in range(args.samples):
                states.append({"uid": group["uid"], "arm": "old",
                               "seed": 1000 + (wave_start + gi) * 100 + si,
                               "text": seed["blocks"], "searches_left": remaining,
                               "done": False, "steps": []})
        rrr.run_arm_states(states, args, llm, tokenizer)
        by = defaultdict(list)
        for s in states:
            by[s["uid"]].append(s)
        for group in wave:
            seed = seeds[group["uid"]]
            record = {k: group[k] for k in ("uid", "data_source", "step", "chunk", "question",
                                               "answers", "original_answered")}
            record["old_ok"] = bool(seed["ok"])
            record["old_error"] = None if seed["ok"] else seed.get("error")
            record["teacher_queries"] = seed.get("queries")
            trajs = []
            for s in by.get(group["uid"], []):
                answer = s.get("final_answer")
                from verl.utils.reward_score.qa_em_format import em_check
                trajs.append({"answer": answer,
                              "em": bool(answer is not None and em_check(answer, group["answers"])),
                              "searches": args.max_turns - s["searches_left"],
                              "error": s.get("search_error")})
            record["old"] = trajs
            results.append(record)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
        print(f"wave at {wave_start} done total={len(results)}", flush=True)
    sink.close()

    # ---- aggregate ----
    all_rows = [json.loads(l) for l in args.output.open()]
    ok_rows = [r for r in all_rows if r["old_ok"]]

    def rate(rows, traj_key, first_only=False):
        n = k = 0
        for r in rows:
            trajs = r.get(traj_key) or []
            if not trajs:
                continue
            n += 1
            if first_only:
                k += int(trajs[0]["em"])
            else:
                k += int(any(t["em"] for t in trajs))
        return k, n

    print("\n=== OLD rescue scheme (currently in training) ===", flush=True)
    print(f"gate pass: {len(ok_rows)}/{len(all_rows)} = {len(ok_rows)/len(all_rows):.3f}",
          flush=True)
    for source in ["all"] + sorted({r["data_source"] for r in all_rows}):
        rows = all_rows if source == "all" else [r for r in all_rows if r["data_source"] == source]
        k6, n6 = rate(rows, "old")
        k1, n1 = rate(rows, "old", first_only=True)
        lo, hi = rrr.wilson_ci(k6, n6)
        print(f"[{source}] n={n6} old RescueRate@6={k6}/{n6}={k6/max(n6,1):.3f} "
              f"(CI {lo:.3f}-{hi:.3f}) | single-suffix(=training harvest)={k1}/{n1}"
              f"={k1/max(n1,1):.3f}", flush=True)

    # ---- head-to-head against new scheme ----
    if args.new_scheme_output.exists():
        new_rows = {r["uid"]: r for r in
                     (json.loads(l) for l in args.new_scheme_output.open())}
        common = [r for r in all_rows if r["uid"] in new_rows]
        print("\n=== HEAD-TO-HEAD on identical all-wrong groups ===", flush=True)
        for denom_name, subset in [("all groups", common),
                                    ("both gates pass",
                                     [r for r in common if r["old_ok"]
                                      and new_rows[r["uid"]]["teacher_ok"]])]:
            n = len(subset)
            new_k = sum(any(t["em"] for t in new_rows[r["uid"]]["prefix"]) for r in subset)
            old_k = sum(bool(r["old"]) and any(t["em"] for t in r["old"]) for r in subset)
            new_only = sum(any(t["em"] for t in new_rows[r["uid"]]["prefix"])
                            and not (bool(r["old"]) and any(t["em"] for t in r["old"]))
                            for r in subset)
            old_only = sum(not any(t["em"] for t in new_rows[r["uid"]]["prefix"])
                            and bool(r["old"]) and any(t["em"] for t in r["old"])
                            for r in subset)
            both = sum(any(t["em"] for t in new_rows[r["uid"]]["prefix"])
                        and bool(r["old"]) and any(t["em"] for t in r["old"])
                        for r in subset)
            print(f"({denom_name}) n={n} | NEW {new_k}/{n}={new_k/max(n,1):.3f} "
                  f"OLD {old_k}/{n}={old_k/max(n,1):.3f} || new_only={new_only} old_only={old_only} "
                  f"both={both}", flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
