#!/usr/bin/env python3
"""Evaluate adaptive no-gold teacher searches after a failed first retrieval."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def documents(url: str, query: str) -> list[str]:
    payload = post_json(url, {"queries": [query], "topk": 3, "return_scores": True})
    return [item["document"]["contents"] for item in payload["result"][0]]


def valid_query(query: str, question: str, aliases: list[str]) -> tuple[bool, str]:
    if not 3 <= len(query.split()) <= 30 or len(query) > 240:
        return False, "length"
    if re.search(r"<[^>]+>|\b(?:answer|search for|look up|verify that)\b", query, re.I):
        return False, "instruction_or_tag"
    for alias in aliases:
        if len(alias) > 3 and alias.casefold() in query.casefold() and alias.casefold() not in question.casefold():
            return False, "privileged_answer"
    return True, ""


def answer_hit(docs: list[str], aliases: list[str]) -> bool:
    return any(len(alias) > 2 and alias.casefold() in doc.casefold()
               for alias in aliases for doc in docs)


def evaluate(row: dict, teacher_url: str, retriever_url: str, candidates: int) -> dict:
    question = row["question"]
    aliases = row["answers"]
    first_prefix = row["new_prefix"].split("<search>", 1)[0]
    result = {key: value for key, value in row.items() if key not in ("baseline", "new", "new_prefix")}
    result["new_prefix"] = first_prefix
    result["target_search_index"] = 0
    baseline = {key: value for key, value in row["baseline"].items() if key != "docs"}
    if baseline.get("query"):
        baseline["valid"], baseline["reject_reason"] = valid_query(
            baseline["query"], question, aliases)
    if baseline.get("valid") and baseline.get("query"):
        baseline["docs"] = documents(retriever_url, baseline["query"])
        baseline["hit"] = answer_hit(baseline["docs"], aliases)
    result["baseline"] = baseline

    attempts = []
    for index in range(candidates):
        feedback = "\n".join(
            f"Query: {item.get('query', '')}\nReturned titles: "
            + "; ".join(doc.partition("\n")[0] for doc in item.get("docs", []))
            for item in attempts
        )
        prompt = (
            'Return ONLY JSON with exactly these nonempty string fields: '
            '{"failure_type":"...","missing_relation":"...","next_operation":"...",'
            '"stop_condition":"..."}. The next_operation MUST be one executable web search '
            'query of 3 to 30 words, not an instruction or a final answer. Search for a '
            'specific entity and missing relation. Do not put an answer candidate in the '
            'query. Avoid earlier searches and revise based on irrelevant returned titles.\n'
            f'Question:\n{question[:2000]}\n'
            f'Student reasoning before the failed search:\n{first_prefix[-2500:]}\n'
            f'Failed search:\n{row["old_searches"][0][:400]}\n'
            f'Its returned documents:\n{row["old_first_obs"][:2600]}\n'
            f'Previous corrective attempts:\n{feedback[-1600:]}\n'
            f'Candidate number: {index + 1}.'
        )
        item = {}
        try:
            response = post_json(teacher_url.rstrip("/") + "/generate", {"prompt": prompt})
            query = " ".join(str(response["skill"]["next_operation"]).split())
            item["query"] = query
            item["valid"], item["reject_reason"] = valid_query(query, question, aliases)
            if item["valid"]:
                item["docs"] = documents(retriever_url, query)
                item["hit"] = answer_hit(item["docs"], aliases)
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"[:250]
        attempts.append(item)
    result["new"] = attempts
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--max-questions", type=int, default=24)
    parser.add_argument("--candidates", type=int, default=3)
    args = parser.parse_args()
    rows = sorted((json.loads(line) for line in args.input.open()), key=lambda row: row["uid"])
    rows = rows[:args.max_questions]
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(evaluate, row, args.teacher_url, args.retriever_url,
                            args.candidates): row["uid"] for row in rows}
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            print(result["uid"], "baseline", result["baseline"].get("hit"),
                  "adaptive", [item.get("hit") for item in result["new"]], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in sorted(results, key=lambda row: row["uid"]):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for arm in ("baseline", "new"):
        items = [row["baseline"] for row in results] if arm == "baseline" else [
            item for row in results for item in row["new"]]
        print(arm, "valid", sum(bool(item.get("valid")) for item in items),
              "hits", sum(bool(item.get("hit")) for item in items), flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
