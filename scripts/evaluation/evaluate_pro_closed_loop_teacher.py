#!/usr/bin/env python3
"""Run a DeepSeek Pro plan-search-verify rescue loop on saved all-wrong cases."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Retrieve documents for one concise factual search query.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 3, "maxLength": 240}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def api_key_from_process(pid: int) -> str:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    prefix = b"DEEPSEEK_API_KEY="
    for item in raw:
        if item.startswith(prefix):
            return item[len(prefix):].decode()
    raise RuntimeError("DEEPSEEK_API_KEY is unavailable")


def retrieve(url: str, query: str) -> list[str]:
    response = requests.post(url, json={"queries": [query], "topk": 3, "return_scores": True}, timeout=120)
    response.raise_for_status()
    return [item["document"]["contents"] for item in response.json()["result"][0]]


def query_valid(query: str, question: str, answers: list[str]) -> tuple[bool, str]:
    query = " ".join(query.split())
    if not 3 <= len(query.split()) <= 30 or len(query) > 240:
        return False, "length"
    for answer in answers:
        if len(answer) > 3 and answer.casefold() in query.casefold() and answer.casefold() not in question.casefold():
            return False, "privileged_answer"
    return True, ""


def safe_handoff(text: str, question: str, answers: list[str]) -> tuple[str, str]:
    text = " ".join(text.split())[:500]
    for answer in answers:
        if len(answer) > 3 and answer.casefold() in text.casefold() and answer.casefold() not in question.casefold():
            return "", "privileged_answer"
    return text, "" if text else "empty"


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("final response is not an object")
    return value


def complete(api_url: str, api_key: str, model: str, messages: list[dict], tool_choice,
             *, thinking: bool = True) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "tools": [SEARCH_TOOL],
        "tool_choice": tool_choice,
        "thinking": {"type": "enabled" if thinking else "disabled"},
        "reasoning_effort": "high" if thinking else "none",
        "max_tokens": 4096 if thinking else 1024,
        "stream": False,
    }
    for attempt in range(3):
        response = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}"},
                                 json=payload, timeout=180)
        if response.status_code < 300:
            return response.json()["choices"][0]["message"]
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {response.text[:500]}")
    raise RuntimeError(f"DeepSeek request failed: HTTP {response.status_code}")


def run_case(row: dict, args, api_key: str) -> dict:
    question, answers = row["question"], row["answers"]
    system = (
        "You are a retrieval teacher rescuing a failed student trajectory. First decompose the exact "
        "entity/relation chain. Use the search tool to retrieve evidence and inspect its result. You may "
        "revise a failed query or search one additional missing relation, with at most two searches total. "
        "Do not put the privileged answer in a query or handoff unless it already appears in the question. "
        "Do not answer the question for the student. Stop only when the visible evidence supports every "
        "relation needed for the student to answer, or when two searches are exhausted."
    )
    user = (
        f"Question:\n{question}\n\nStudent reasoning before its failed search:\n{row['new_prefix'][-3000:]}\n\n"
        f"Failed student query:\n{row['old_searches'][0]}\n\nIts returned evidence:\n{row['old_first_obs'][:3500]}\n\n"
        f"Privileged correct answer for planning and verification only:\n{answers[0]}\n\n"
        "After tool use, return JSON with keys relation_chain (array of strings), resolved_relations "
        "(array), unresolved_relations (array), evidence_sufficient (boolean), and handoff (string). "
        "The handoff must tell the student what relation to synthesize from the evidence without stating "
        "the final answer."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    actions, reasoning = [], []
    for _round in range(args.max_searches + 2):
        force_final = len(actions) >= args.max_searches
        if force_final:
            messages.append({"role": "user", "content": "Search budget is exhausted. Return the required final JSON now."})
        message = complete(args.api_url, api_key, args.model, messages,
                           "none" if force_final else "auto", thinking=not force_final)
        if message.get("reasoning_content"):
            reasoning.append(str(message["reasoning_content"])[:6000])
        calls = message.get("tool_calls") or []
        if calls and not force_final:
            assistant = {"role": "assistant", "content": message.get("content"),
                         "reasoning_content": message.get("reasoning_content"), "tool_calls": calls}
            messages.append(assistant)
            for call in calls:
                if len(actions) >= args.max_searches:
                    result = {"valid": False, "reject_reason": "search_budget_exhausted", "documents": []}
                else:
                    try:
                        query = json.loads(call["function"]["arguments"])["query"]
                    except Exception as error:
                        query = ""
                        valid, reason, docs = False, f"invalid_arguments:{type(error).__name__}", []
                    else:
                        valid, reason = query_valid(query, question, answers)
                        docs = retrieve(args.retriever_url, query) if valid else []
                    action = {"query": query, "valid": valid, "reject_reason": reason, "docs": docs}
                    actions.append(action)
                    result = {"valid": valid, "reject_reason": reason,
                              "documents": [doc[:2200] for doc in docs]}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
            continue
        content = str(message.get("content") or "")
        if not content:
            raise ValueError(f"teacher returned empty final content; fields={sorted(message)}")
        plan = parse_json(content)
        break
    else:
        raise RuntimeError("teacher did not terminate")
    if not actions:
        raise ValueError("teacher performed no rescue search")
    handoff, reject = safe_handoff(str(plan.get("handoff", "")), question, answers)
    first = dict(actions[0])
    first["teacher_actions"] = actions
    first["handoff"] = handoff
    first["handoff_reject_reason"] = reject
    first["teacher_plan"] = plan
    first["teacher_reasoning"] = reasoning
    result = dict(row)
    result["new"] = [first]
    result["target_search_index"] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--api-url", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-from-pid", type=int, required=True)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=24)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    api_key = api_key_from_process(args.api_key_from_pid)
    source = args.output if args.retry_errors and args.output.exists() else args.input
    rows = sorted((json.loads(line) for line in source.open()), key=lambda row: row["uid"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = args.output.with_suffix(args.output.suffix + ".tmp") if source == args.output else args.output
    with destination.open("w") as stream:
        for row in rows[:args.max_questions]:
            if args.retry_errors and "pro_teacher_error" not in row:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            try:
                result = run_case(row, args, api_key)
                result.pop("pro_teacher_error", None)
                print(row["uid"], "searches", len(result["new"][0]["teacher_actions"]),
                      "sufficient", result["new"][0]["teacher_plan"].get("evidence_sufficient"), flush=True)
            except Exception as error:
                result = dict(row)
                result["new"] = []
                result["pro_teacher_error"] = f"{type(error).__name__}: {error}"[:500]
                print(row["uid"], result["pro_teacher_error"], flush=True)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    if destination != args.output:
        destination.replace(args.output)


if __name__ == "__main__":
    main()
