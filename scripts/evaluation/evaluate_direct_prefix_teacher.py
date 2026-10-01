#!/usr/bin/env python3
"""Pilot: simple think+search teacher prefix vs frozen student baseline.

Instead of the closed-loop plan / evidence_sufficient / handoff machinery, the
teacher (DeepSeek chat API, NOT shown the gold answer) directly writes the
first two hops as plain ``<think>`` + ``<search>`` blocks.  The local
retriever supplies real observations, and the frozen student continues the
remaining turns itself.  Paired comparison on the same questions:

* baseline: student answers alone (up to ``--max-turns`` searches)
* prefix:   teacher hops 1-2 (real retrieval) -> student continues
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from verl.utils.reward_score.qa_em_format import em_check  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")

TEACHER_SYSTEM = (
    "You are the retrieval scout for a weaker question-answering student. "
    "The student must answer a multi-hop question by itself. You perform only "
    "the first {hops} retrieval hops. On every turn output exactly two blocks "
    "and nothing else:\n"
    "<think>one or two sentences stating which entity or relation this hop "
    "must resolve, using only the question and the visible snippets</think>\n"
    "<search>one concise factual web search query (3-30 words)</search>\n"
    "Rules: never state a final answer; do not conclude for the student; each "
    "query must be self-contained; output no text outside the two blocks."
)


def api_key_from_pid(pid: int) -> str:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    prefix = b"DEEPSEEK_API_KEY="
    for item in raw:
        if item.startswith(prefix):
            return item[len(prefix):].decode()
    raise RuntimeError("DEEPSEEK_API_KEY is unavailable")


def retrieve(url: str, query: str) -> list[str]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"queries": [query], "topk": 3, "return_scores": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
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


def teacher_complete(api_url: str, api_key: str, model: str, messages: list[dict]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": False,
    }
    for attempt in range(4):
        response = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}"},
                                 json=payload, timeout=180)
        if response.status_code < 300:
            return response.json()["choices"][0]["message"].get("content") or ""
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {response.text[:300]}")
    raise RuntimeError("DeepSeek request failed after retries")


FALLBACK_THINK = "I will retrieve the next relation needed to answer the question."


def parse_hop(text: str) -> tuple[str, str] | None:
    # Tolerant parsing: blocks are located independently so extra preamble,
    # markdown fences or a second trailing block do not kill the whole hop.
    text = re.sub(r"^```(?:json|html|text)?\s*|\s*```$", "", text.strip(), flags=re.I)
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S | re.I)
    search_match = re.search(r"<search>\s*(.*?)\s*</search>", text, re.S | re.I)
    if not search_match:
        return None
    think = " ".join(think_match.group(1).split()) if think_match else FALLBACK_THINK
    query = " ".join(search_match.group(1).split()).strip('"').strip("'")
    words = query.split()
    if len(words) > 30:  # salvage overly long queries instead of dropping the question
        query = " ".join(words[:30])
    if len(query.split()) < 3 or len(query) > 240 or TAG_RE.search(query):
        return None
    # The scout is forbidden from answering; drop any accidental answer block.
    think = re.sub(r"<answer>.*?</answer>", "", think, flags=re.DOTALL | re.IGNORECASE).strip()
    if not think:
        think = FALLBACK_THINK
    return think, query


def build_teacher_prefix(question: str, tokenizer, args, api_key: str) -> dict:
    """Run ``hops`` teacher think+search rounds against the real retriever."""
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM.format(hops=args.teacher_hops)},
        {"role": "user", "content": f"Question: {question}\n\nProduce hop 1 now."},
    ]
    blocks, hops, errors = "", [], []
    for hop_index in range(1, args.teacher_hops + 1):
        content = teacher_complete(args.api_url, api_key, args.model_name, messages)
        parsed = parse_hop(content)
        if parsed is None:
            # One repair chance, then give up (this arm is excluded from pairing).
            messages.append({"role": "assistant", "content": content[:1000]})
            messages.append({"role": "user", "content":
                             "Your previous output was invalid. Output exactly "
                             "<think>...</think><search>...</search>, query 3-30 words."})
            content = teacher_complete(args.api_url, api_key, args.model_name, messages)
            parsed = parse_hop(content)
            if parsed is None:
                return {"ok": False, "error": f"invalid_hop_{hop_index}", "raw": content[:500]}
        think, query = parsed
        try:
            docs = retrieve(args.retriever_url, query)
        except Exception as exc:  # retriever-side failure
            return {"ok": False, "error": f"retrieval_hop_{hop_index}: {type(exc).__name__}"}
        if not docs:
            errors.append(f"hop_{hop_index}_empty_docs")
        obs = observation(docs, tokenizer)
        block = f"<think>{think}</think>\n<search>{query}</search>{obs}"
        blocks += block
        hops.append({"hop": hop_index, "think": think, "query": query,
                     "docs": docs, "empty_docs": not docs})
        messages.append({"role": "assistant", "content": f"<think>{think}</think>\n<search>{query}</search>"})
        if hop_index < args.teacher_hops:
            messages.append({"role": "user", "content":
                             f"Search results:\n{obs}\n\nProduce hop {hop_index + 1} now."})
    return {"ok": True, "blocks": blocks, "hops": hops, "warnings": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path,
                        default="data/multihopqa_search_mixed_402020_20260830/balanced_test.parquet")
    parser.add_argument("--model", type=Path,
                        default="verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8002/retrieve")
    parser.add_argument("--api-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--api-key-from-pid", type=int, default=3940943)
    parser.add_argument("--per-source", type=int, default=12)
    parser.add_argument("--teacher-hops", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=5,
                        help="total search budget; prefix arm leaves max-turns-teacher-hops for student")
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--teacher-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    api_key = api_key_from_pid(args.api_key_from_pid)
    df = pd.read_parquet(args.parquet)
    picked = pd.concat([
        group.sample(n=min(args.per_source, len(group)), random_state=args.seed)
        for _, group in df.groupby("data_source")
    ]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"questions={len(picked)} per_source={args.per_source}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    questions = []
    for _, row in picked.iterrows():
        prompt = tokenizer.apply_chat_template(list(row["prompt"]), tokenize=False,
                                               add_generation_prompt=True)
        question = list(row["prompt"])[-1]["content"].split("Question:", 1)[-1].strip()
        answers = list(row["reward_model"]["ground_truth"]["target"])
        questions.append({"uid": str(row["extra_info"]["index"]),
                          "data_source": row["data_source"], "question": question,
                          "prompt": prompt, "answers": answers})

    # ---- stage 1: teacher prefixes (threaded API calls) ----
    prefixes: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.teacher_workers) as pool:
        futures = {pool.submit(build_teacher_prefix, item["question"], tokenizer, args, api_key): item["uid"]
                   for item in questions}
        for done, future in enumerate(as_completed(futures), 1):
            uid = futures[future]
            try:
                prefixes[uid] = future.result()
            except Exception as exc:
                prefixes[uid] = {"ok": False, "error": f"exception: {type(exc).__name__}: {exc}"}
            if done % 8 == 0 or done == len(futures):
                ok = sum(v["ok"] for v in prefixes.values())
                print(f"teacher {done}/{len(futures)} ok={ok}", flush=True)
    failures = {uid: v for uid, v in prefixes.items() if not v["ok"]}
    for uid, value in list(failures.items())[:5]:
        print(f"[teacher prefix failed] {uid}: {value['error']}", flush=True)

    # ---- stage 2: student rollouts (baseline + prefix), batched per turn ----
    llm = LLM(model=str(args.model), dtype="half", tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=True, disable_log_stats=True)

    states = []
    student_budget = args.max_turns - args.teacher_hops
    for item in questions:
        states.append({"uid": item["uid"], "data_source": item["data_source"], "arm": "baseline",
                       "answers": item["answers"], "text": item["prompt"],
                       "prompt_chars": len(item["prompt"]), "searches_left": args.max_turns,
                       "done": False, "steps": []})
        prefix = prefixes.get(item["uid"])
        if prefix and prefix["ok"]:
            states.append({"uid": item["uid"], "data_source": item["data_source"], "arm": "prefix",
                           "answers": item["answers"], "text": item["prompt"] + prefix["blocks"],
                           "prompt_chars": len(item["prompt"]), "searches_left": student_budget,
                           "done": False, "steps": []})

    repair = ("\nMy previous action is invalid. If I want to search, I should put the query "
              "between <search> and </search>. If I want to give the final answer, I should put "
              "the answer between <answer> and </answer>. Let me try again.\n")
    keep_context = args.max_model_len - 600
    for iteration in range(args.max_turns + 4):  # searches plus a few invalid-action retries
        active = [state for state in states if not state["done"]]
        if not active:
            break
        prompts = [{"prompt_token_ids": tokenizer.encode(state["text"], add_special_tokens=False)[-keep_context:]}
                   for state in active]
        sampling = [SamplingParams(temperature=0.0, max_tokens=500, repetition_penalty=1.05,
                                   seed=args.seed)] * len(active)
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for state, output in zip(active, outputs):
            piece = output.outputs[0].text
            if "</search>" in piece:
                piece = piece.split("</search>", 1)[0] + "</search>"
            elif "</answer>" in piece:
                piece = piece.split("</answer>", 1)[0] + "</answer>"
            state["text"] += piece
            state["steps"].append(piece)
            action = re.search(r"<(search|answer)>(.*?)</\1>", piece, re.S | re.IGNORECASE)
            if action and action.group(1).lower() == "answer":
                state["final_answer"] = action.group(2).strip()
                state["done"] = True
                continue
            if action and action.group(1).lower() == "search":
                if state["searches_left"] <= 0:
                    state["done"] = True  # budget exhausted without an answer
                    continue
                try:
                    state["text"] += observation(retrieve(args.retriever_url,
                                                          action.group(2).strip()), tokenizer)
                    state["searches_left"] -= 1
                except Exception as exc:
                    state["search_error"] = str(exc)
                    state["done"] = True
            else:
                state["text"] += repair

    # ---- score + report ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    by_key = {(state["uid"], state["arm"]): state for state in states}
    results = []
    paired_uids = {item["uid"] for item in questions
                   if (item["uid"], "baseline") in by_key and (item["uid"], "prefix") in by_key}
    with args.output.open("w") as stream:
        for item in questions:
            base = by_key.get((item["uid"], "baseline"))
            pref = by_key.get((item["uid"], "prefix"))
            for state in (base, pref):
                if state is None:
                    continue
                state["em"] = bool(state.get("final_answer") is not None
                                   and em_check(state["final_answer"], item["answers"]))
                state["response"] = state["text"][state.pop("prompt_chars"):]
                state.pop("text", None)
            record = {"uid": item["uid"], "data_source": item["data_source"],
                      "question": item["question"], "answers": item["answers"],
                      "baseline": base, "prefix": pref,
                      "teacher": None if prefixes[item["uid"]]["ok"] else
                                 {"error": prefixes[item["uid"]]["error"]},
                      "teacher_hops": prefixes[item["uid"]].get("hops")}
            results.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def em(state):
        return bool(state and state.get("em"))

    def answered(state):
        return bool(state and state.get("final_answer") is not None)

    print(f"\nteacher_prefix_ok={len(paired_uids)}/{len(questions)} "
          f"failed={len(questions) - len(paired_uids)}", flush=True)
    for source in ["all"] + sorted({item["data_source"] for item in questions}):
        uids = paired_uids if source == "all" else {item["uid"] for item in questions
                                                    if item["data_source"] == source} & paired_uids
        if not uids:
            continue
        base_em = sum(em(by_key[(uid, "baseline")]) for uid in uids)
        pref_em = sum(em(by_key[(uid, "prefix")]) for uid in uids)
        rescued = sum(not em(by_key[(uid, "baseline")]) and em(by_key[(uid, "prefix")]) for uid in uids)
        harmed = sum(em(by_key[(uid, "baseline")]) and not em(by_key[(uid, "prefix")]) for uid in uids)
        base_ans = sum(answered(by_key[(uid, "baseline")]) for uid in uids)
        pref_ans = sum(answered(by_key[(uid, "prefix")]) for uid in uids)
        print(f"[{source}] n={len(uids)} baseline_em={base_em} ({base_em/len(uids):.3f}) "
              f"prefix_em={pref_em} ({pref_em/len(uids):.3f}) | rescued={rescued} harmed={harmed} "
              f"| answered base={base_ans} prefix={pref_ans}", flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
