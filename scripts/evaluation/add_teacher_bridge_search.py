#!/usr/bin/env python3
"""Add a second no-gold teacher bridge search after the first rescue evidence."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def valid(query: str, question: str, answers: list[str]) -> bool:
    if not 3 <= len(query.split()) <= 30 or len(query) > 240:
        return False
    if re.search(r"<[^>]+>|\b(?:answer|search for|look up|verify that)\b", query, re.I):
        return False
    return not any(len(answer) > 3 and answer.casefold() in query.casefold()
                   and answer.casefold() not in question.casefold() for answer in answers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--retriever-url", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for line in args.input.open():
            row = json.loads(line)
            first = row["new"][0]
            docs = "\n\n".join(f"Doc {i + 1}: {doc[:1600]}" for i, doc in enumerate(first.get("docs", [])))
            prompt = (
                'Return ONLY JSON with exactly these nonempty string fields: '
                '{"failure_type":"...","missing_relation":"...","next_operation":"...","stop_condition":"..."}. '
                "Using the question and first retrieval evidence, next_operation must be one short bridge search "
                "for the unresolved relation. It must retrieve a missing supporting fact, never an answer candidate "
                "or final answer. Do not use XML tags or instructions.\n"
                f"Question:\n{row['question'][:1800]}\nFirst search:\n{first.get('query', '')}\nEvidence:\n{docs}")
            try:
                skill = post(args.teacher_url.rstrip("/") + "/generate", {"prompt": prompt})["skill"]
                query = " ".join(str(skill["next_operation"]).split())
                # The analyzer occasionally prefixes an otherwise usable query with
                # "search" despite the requested schema.  Strip only that imperative;
                # this does not add any privileged content.
                query = re.sub(r"^(?:search(?:\s+for)?|look\s+up)\s*:?\s*", "", query,
                               flags=re.I)
                second = {"query": query, "valid": valid(query, row["question"], row["answers"])}
                if second["valid"]:
                    result = post(args.retriever_url, {"queries": [query], "topk": 3, "return_scores": True})
                    second["docs"] = [item["document"]["contents"] for item in result["result"][0]]
                first["teacher_actions"] = [dict(first), second]
                # Avoid recursive copies in output and keep first action explicit.
                first["teacher_actions"][0].pop("teacher_actions", None)
                row["new"] = [first]
            except Exception as error:
                first["bridge_error"] = f"{type(error).__name__}: {error}"[:250]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row["uid"], flush=True)


if __name__ == "__main__":
    main()
