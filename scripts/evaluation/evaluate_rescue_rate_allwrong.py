#!/usr/bin/env python3
"""RescueRate pilot on real DAPO all-wrong groups.

Protocol (per group):
  original group = 6/6 wrong student trajectories (from recent rollout dumps)
  prefix arm : simple 2-hop teacher think+search prefix -> resample 6 student
               suffixes at training temperature (1.0), 3 searches remaining
  fresh  arm : 6 fresh student rollouts with no prefix (5 searches), as the
               "more sampling alone" null control

Primary metric:
  RescueRate = P(at least 1 of the 6 suffix trajectories is correct
                | the original group was all wrong)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from verl.utils.reward_score.qa_em_format import em_check  # noqa: E402
import evaluate_direct_prefix_teacher as dpt  # noqa: E402

QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*<\|im_end\|>", re.S)


def extract_answer(response: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.S | re.I)
    return matches[-1].strip() if matches else None


def load_all_wrong_groups(rollout_dir: Path, min_step: int, max_step: int):
    """One record per unique uid, preferring the most recent step."""
    files = [f for f in glob.glob(str(rollout_dir / "step_*_chunk_*_selected.jsonl"))
             if min_step <= int(re.search(r"step_(\d+)", f).group(1)) <= max_step]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for path in files:
        step, chunk = (int(v) for v in re.search(r"step_(\d+)_chunk_(\d+)", path).groups())
        for line in open(path):
            row = json.loads(line)
            grouped[(step, chunk, row["uid"])].append(row)
    best: dict[str, dict] = {}
    for (step, chunk, uid), items in grouped.items():
        if len(items) != 6:
            continue
        answers = items[0]["ground_truth"]["target"]
        ems = [bool((a := extract_answer(it["response"])) is not None and em_check(a, answers))
               for it in items]
        if any(ems):
            continue
        if uid not in best or step > best[uid]["step"]:
            first = items[0]
            question = QUESTION_RE.search(first["prompt"])
            if not question:
                continue
            best[uid] = {"uid": uid, "data_source": first["data_source"], "step": step,
                         "chunk": chunk, "prompt": first["prompt"],
                         "question": question.group(1).strip(), "answers": list(answers),
                         "original_answered": sum(extract_answer(it["response"]) is not None
                                                  for it in items)}
    return sorted(best.values(), key=lambda r: (r["data_source"], r["uid"]))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run_arm_states(states: list[dict], args, llm, tokenizer):
    repair = ("\nMy previous action is invalid. If I want to search, I should put the query "
              "between <search> and </search>. If I want to give the final answer, I should put "
              "the answer between <answer> and </answer>. Let me try again.\n")
    keep_context = args.max_model_len - 600
    for _ in range(args.max_turns + 4):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        prompts = [{"prompt_token_ids": tokenizer.encode(s["text"], add_special_tokens=False)[-keep_context:]}
                   for s in active]
        from vllm import SamplingParams
        sampling = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=500,
                                   repetition_penalty=1.05, seed=args.seed + s["seed"])
                    for s in active]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for state, output in zip(active, outputs):
            piece = output.outputs[0].text
            if "</search>" in piece:
                piece = piece.split("</search>", 1)[0] + "</search>"
            elif "</answer>" in piece:
                piece = piece.split("</answer>", 1)[0] + "</answer>"
            state["text"] += piece
            state["steps"].append(piece)
            action = re.search(r"<(search|answer)>(.*?)</\1>", piece, re.S | re.I)
            if action and action.group(1).lower() == "answer":
                state["final_answer"] = action.group(2).strip()
                state["done"] = True
                continue
            if action and action.group(1).lower() == "search":
                if state["searches_left"] <= 0:
                    state["done"] = True
                    continue
                try:
                    state["text"] += dpt.observation(
                        dpt.retrieve(args.retriever_url, action.group(2).strip()), tokenizer)
                    state["searches_left"] -= 1
                except Exception as exc:
                    state["search_error"] = str(exc)
                    state["done"] = True
            else:
                state["text"] += repair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path,
                        default=ROOT / "rollouts/tracecredit-opid-flash-semantic-rescue-sft350-step0to20-20260929")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "verl_checkpoints/tracecredit-opid-flash-semantic-rescue-sft350-step0to20-20260929/actor/global_step_12/hf_merged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8002/retrieve")
    parser.add_argument("--api-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key-from-pid", type=int, default=3940943)
    parser.add_argument("--min-step", type=int, default=9)
    parser.add_argument("--max-step", type=int, default=12)
    parser.add_argument("--max-groups", type=int, default=33)
    parser.add_argument("--offset", type=int, default=0,
                        help="skip the first N selected groups (resume support)")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--teacher-hops", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--groups-per-wave", type=int, default=10)
    parser.add_argument("--teacher-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    groups = load_all_wrong_groups(args.rollout_dir, args.min_step, args.max_step)
    # balanced, deterministic subset
    by_source = defaultdict(list)
    for group in groups:
        by_source[group["data_source"]].append(group)
    selected = []
    per_source = Counter(g["data_source"] for g in groups)
    cap = max(1, math.ceil(args.max_groups / max(1, len(by_source))))
    for source, items in sorted(by_source.items()):
        selected.extend(items[:min(cap, len(items))])
    selected = selected[:args.max_groups]
    selected = selected[args.offset:]
    print(f"all_wrong_pool={len(groups)} per_source={dict(per_source)} selected={len(selected)} "
          f"offset={args.offset}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    api_key = dpt.api_key_from_pid(args.api_key_from_pid)

    # ---- teacher prefixes (with local disk cache for cheap resume) ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
    teacher_args = argparse.Namespace(**{**vars(args), "teacher_hops": args.teacher_hops})
    cache_path = args.output.with_suffix(".teacher_cache.jsonl")
    prefixes: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.open():
            cached = json.loads(line)
            if cached["result"].get("ok"):
                prefixes[cached["uid"]] = cached["result"]
    pending = [g for g in selected if g["uid"] not in prefixes]
    print(f"teacher: cached={len(selected) - len(pending)} pending={len(pending)}", flush=True)
    cache_stream = cache_path.open("a")
    with ThreadPoolExecutor(max_workers=args.teacher_workers) as pool:
        futures = {pool.submit(dpt.build_teacher_prefix, g["question"], tokenizer,
                               teacher_args, api_key): g["uid"] for g in pending}
        for done, future in enumerate(as_completed(futures), 1):
            uid = futures[future]
            try:
                prefixes[uid] = future.result()
            except Exception as exc:
                prefixes[uid] = {"ok": False, "error": f"exception: {type(exc).__name__}: {exc}"}
            cache_stream.write(json.dumps({"uid": uid, "result": prefixes[uid]},
                                          ensure_ascii=False) + "\n")
            cache_stream.flush()
            if done % 8 == 0 or done == len(futures):
                print(f"teacher {done}/{len(futures)} ok={sum(v['ok'] for v in prefixes.values())}",
                      flush=True)
    cache_stream.close()

    llm = LLM(model=str(args.model), dtype="half", tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=True, disable_log_stats=True)

    results = []
    student_budget = args.max_turns - args.teacher_hops
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sink = args.output.open("a")
    for wave_start in range(0, len(selected), args.groups_per_wave):
        wave = selected[wave_start:wave_start + args.groups_per_wave]
        states = []
        for gi, group in enumerate(wave):
            global_index = args.offset + wave_start + gi  # stable across resumed runs
            for si in range(args.samples):
                states.append({"uid": group["uid"], "arm": "fresh", "seed": global_index * 100 + si,
                               "text": group["prompt"], "searches_left": args.max_turns,
                               "done": False, "steps": []})
                prefix = prefixes.get(group["uid"])
                if prefix and prefix["ok"]:
                    states.append({"uid": group["uid"], "arm": "prefix",
                                   "seed": 500 + global_index * 100 + si,
                                   "text": group["prompt"] + prefix["blocks"],
                                   "searches_left": student_budget,
                                   "done": False, "steps": []})
        run_arm_states(states, args, llm, tokenizer)
        by = defaultdict(list)
        for s in states:
            by[(s["uid"], s["arm"])].append(s)
        for group in wave:
            record = {k: group[k] for k in ("uid", "data_source", "step", "chunk", "question",
                                            "answers", "original_answered")}
            prefix = prefixes.get(group["uid"])
            record["teacher_ok"] = bool(prefix and prefix["ok"])
            record["teacher_error"] = None if record["teacher_ok"] else prefix.get("error")
            record["teacher_hops"] = prefix.get("hops") if record["teacher_ok"] else None

            def summarize(arm_states):
                out = []
                for s in arm_states:
                    answer = s.get("final_answer")
                    out.append({"answer": answer,
                                "em": bool(answer is not None and em_check(answer, group["answers"])),
                                "searches": args.max_turns - s["searches_left"] if arm_name == "fresh"
                                            else args.teacher_hops + student_budget - s["searches_left"],
                                "error": s.get("search_error")})
                return out

            for arm_name in ("fresh", "prefix"):
                record[arm_name] = summarize(by.get((group["uid"], arm_name), []))
            results.append(record)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
        print(f"wave at offset {args.offset + wave_start} done groups_total={args.offset + len(results)}",
              flush=True)
    sink.close()

    # ---- RescueRate ----
    def rate(subset, arm):
        n = k = 0
        for r in subset:
            trajs = r[arm]
            if not trajs:
                continue
            n += 1
            k += int(any(t["em"] for t in trajs))
        return k, n

    # aggregate over the whole file so resumed chunks are included
    all_results = [json.loads(line) for line in args.output.open()]
    # de-duplicate by uid (a resumed rerun may append the same group)
    deduped = {r["uid"]: r for r in all_results}
    results_all = list(deduped.values())

    print("\n=== RescueRate (>=1 correct of 6 resamples, on original all-wrong groups) ===",
          flush=True)
    evaluable = [r for r in results_all if r["teacher_ok"]]
    for source in ["all"] + sorted({r["data_source"] for r in evaluable}):
        subset = evaluable if source == "all" else [r for r in evaluable if r["data_source"] == source]
        kp, np_ = rate(subset, "prefix")
        kf, nf = rate(subset, "fresh")
        teacher_only = sum(any(t["em"] for t in r["prefix"]) and not any(t["em"] for t in r["fresh"])
                           for r in subset)
        both = sum(any(t["em"] for t in r["prefix"]) and any(t["em"] for t in r["fresh"])
                   for r in subset)
        fresh_only = sum(not any(t["em"] for t in r["prefix"]) and any(t["em"] for t in r["fresh"])
                         for r in subset)
        lop, hip = wilson_ci(kp, np_)
        lof, hif = wilson_ci(kf, nf)
        print(f"[{source}] n={np_} | prefix RescueRate={kp}/{np_}={kp/np_:.3f} "
              f"(CI {lop:.3f}-{hip:.3f}) | fresh-resample={kf}/{nf}={kf/nf:.3f} "
              f"(CI {lof:.3f}-{hif:.3f}) | teacher_only={teacher_only} both={both} "
              f"fresh_only={fresh_only}", flush=True)
    print(f"teacher_prefix_failed={len(results_all) - len(evaluable)}", flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
