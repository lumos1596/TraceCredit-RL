#!/usr/bin/env python3
"""Compare frozen SFT350 student continuations from saved teacher query candidates."""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from verl.utils.reward_score.qa_em_format import em_check, is_valid_sequence  # noqa: E402


def retrieve(url: str, query: str) -> list[str]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"queries": [query], "topk": 3, "return_scores": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return [item["document"]["contents"] for item in json.load(response)["result"][0]]


def observation(documents: list[str], tokenizer) -> str:
    references = ""
    for index, document in enumerate(documents, 1):
        title, _, body = document.partition("\n")
        references += f"Doc {index}(Title: {title}) {body}\n"
    content = f"\n\n<information>{references.strip()}</information>\n\n"
    tokens = tokenizer.encode(content, add_special_tokens=False)
    closing = tokenizer.encode("</information>", add_special_tokens=False)
    if len(tokens) > 500:
        tokens = tokens[:500 - len(closing)] + closing
    return tokenizer.decode(tokens, skip_special_tokens=False)


def relevance(documents: list[str], question: str) -> int:
    stop = {"what", "which", "where", "when", "with", "from", "that", "this", "were", "their"}
    terms = set(re.findall(r"[a-zA-Z]{4,}", question.casefold())) - stop
    return sum(len(terms & set(re.findall(r"[a-zA-Z]{4,}", doc.casefold()))) for doc in documents)


def load_original(rows: list[dict], rollout_dir: Path) -> dict[str, dict]:
    wanted = {row["uid"]: row["step"] for row in rows}
    original = {}
    for path in rollout_dir.glob("step_*_selected.jsonl"):
        for line in path.open():
            row = json.loads(line)
            uid = row["uid"]
            if uid in wanted and row["global_step"] == wanted[uid] and uid not in original:
                original[uid] = row
    return original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-questions", type=int, default=24)
    parser.add_argument("--samples-per-arm", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--new-handoff-hint", default="",
                        help="Optional non-answer-bearing teacher think after new-arm evidence")
    parser.add_argument("--answer-now-when-sufficient", action="store_true",
                        help="Tell the student to answer immediately when the teacher verified full evidence")
    args = parser.parse_args()

    rows = sorted((json.loads(line) for line in args.input.open()), key=lambda row: row["uid"])
    rows = rows[:args.max_questions]
    original = load_original(rows, args.rollout_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    llm = LLM(model=str(args.model), dtype="half", tensor_parallel_size=1,
              gpu_memory_utilization=0.75, max_model_len=2560,
              enforce_eager=True, disable_log_stats=True)

    states = []
    for row in rows:
        source = original.get(row["uid"])
        if source is None:
            continue
        first_search = re.search(r"<search>(.*?)</search>", source["response"], re.S)
        if first_search is None:
            continue
        for arm in ("baseline", "new"):
            if arm == "baseline":
                chosen = row["baseline"]
                prefix, target = source["response"][:first_search.start()], 0
                teacher_actions = [chosen]
            else:
                candidates = [item for item in row["new"] if item.get("valid") and item.get("docs")]
                if not candidates:
                    continue
                # Candidate selection must use only information available at rollout time.
                chosen = max(candidates, key=lambda item: relevance(item["docs"], row["question"]))
                prefix, target = row["new_prefix"], row["target_search_index"]
                teacher_actions = chosen.get("teacher_actions", [chosen])
            if not chosen.get("valid") or not chosen.get("docs"):
                continue
            teacher_actions = [action for action in teacher_actions
                               if action.get("valid") and action.get("docs")]
            if not teacher_actions:
                continue
            seed_text = source["prompt"] + prefix
            for action_index, teacher_action in enumerate(teacher_actions):
                if action_index:
                    seed_text += ("<think>The preceding evidence leaves another required relation unresolved, "
                                  "so I will retrieve that specific relation.</think>\n")
                seed_text += (f"<search>{teacher_action['query']}</search>"
                              + observation(teacher_action["docs"], tokenizer))
            if arm == "new":
                handoff = chosen.get("handoff", args.new_handoff_hint)
                plan = chosen.get("teacher_plan", {})
                if args.answer_now_when_sufficient and plan.get("evidence_sufficient"):
                    handoff = ("The retrieved evidence fully resolves every required relation. Do not search "
                               "again. Synthesize the exact entity or value requested and answer now. " + handoff)
                if handoff:
                    seed_text += f"<think>{handoff}</think>\n"
            for sample_index in range(args.samples_per_arm):
                states.append({"uid": row["uid"], "arm": arm, "sample_index": sample_index,
                               "answer": row["answers"], "teacher_query": chosen["query"],
                               "teacher_docs": chosen["docs"], "target": target,
                               "text": seed_text, "prompt_chars": len(source["prompt"]),
                               "remaining": 2 - target - (len(teacher_actions) - 1),
                               "done": False, "steps": []})

    print(f"continuations={len(states)}", flush=True)
    for turn in range(3):
        active = [state for state in states if not state["done"] and state["remaining"] >= 0]
        if not active:
            break
        prompts = [{"prompt_token_ids": tokenizer.encode(state["text"], add_special_tokens=False)[-2048:]}
                   for state in active]
        sampling = [SamplingParams(temperature=args.temperature, max_tokens=500,
                                   repetition_penalty=1.05,
                                   seed=args.seed + state["sample_index"])
                    for state in active]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for state, output in zip(active, outputs):
            # Match GenerationManager._postprocess_responses: any model text
            # after the first complete action is discarded before the tool runs.
            piece = output.outputs[0].text
            if "</search>" in piece:
                piece = piece.split("</search>", 1)[0] + "</search>"
            elif "</answer>" in piece:
                piece = piece.split("</answer>", 1)[0] + "</answer>"
            state["text"] += piece
            state["steps"].append(piece)
            action = re.search(r"<(search|answer)>(.*?)</\1>", piece, re.S)
            if action and action.group(1) == "answer":
                state["final_answer"] = action.group(2).strip()
                state["done"] = True
                continue
            if state["remaining"] == 0:
                state["done"] = True
                continue
            if action and action.group(1) == "search":
                try:
                    state["text"] += observation(retrieve(args.retriever_url, action.group(2).strip()), tokenizer)
                except Exception as error:
                    state["search_error"] = str(error)
                    state["done"] = True
            else:
                state["text"] += ("\nMy previous action is invalid. If I want to search, I should "
                                  "put the query between <search> and </search>. If I want to give "
                                  "the final answer, I should put the answer between <answer> and </answer>. "
                                  "Let me try again.\n")
            state["remaining"] -= 1
        print(f"turn={turn + 1} done={sum(state['done'] for state in states)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for state in states:
            state["em"] = bool(state.get("final_answer") is not None
                               and em_check(state["final_answer"], state["answer"]))
            state["format_valid"] = bool(is_valid_sequence(state["text"])[0])
            state["response"] = state["text"][state.pop("prompt_chars"):]
            state.pop("text", None)
            stream.write(json.dumps(state, ensure_ascii=False) + "\n")
    paired = {state["uid"] for state in states if state["arm"] == "baseline"} & {
        state["uid"] for state in states if state["arm"] == "new"}
    for arm in ("baseline", "new"):
        subset = [state for state in states if state["uid"] in paired and state["arm"] == arm]
        passed = sum(any(state["em"] for state in subset if state["uid"] == uid)
                     for uid in paired)
        print(f"{arm}: paired={len(subset)} em={sum(state['em'] for state in subset)} "
              f"answered={sum('final_answer' in state for state in subset)} "
              f"pass@{args.samples_per_arm}={passed}/{len(paired)}", flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
