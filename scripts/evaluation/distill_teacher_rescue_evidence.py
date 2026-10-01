#!/usr/bin/env python3
"""Ask the no-gold teacher to turn retrieved evidence into a short student handoff."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


def relevance(documents: list[str], question: str) -> int:
    stop = {"what", "which", "where", "when", "with", "from", "that", "this", "were", "their"}
    terms = set(re.findall(r"[a-zA-Z]{4,}", question.casefold())) - stop
    return sum(len(terms & set(re.findall(r"[a-zA-Z]{4,}", doc.casefold()))) for doc in documents)


def request(url: str, prompt: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/generate",
        data=json.dumps({"prompt": prompt}, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)["skill"]


def safe_handoff(text: str, question: str, answers: list[str]) -> str:
    text = " ".join(text.split())[:360]
    if not text:
        return ""
    for answer in answers:
        if len(answer) > 3 and answer.casefold() in text.casefold() and answer.casefold() not in question.casefold():
            return ""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-url", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for line in args.input.open():
            row = json.loads(line)
            candidates = [x for x in row["new"] if x.get("valid") and x.get("docs")]
            if candidates:
                chosen = max(candidates, key=lambda x: relevance(x["docs"], row["question"]))
                docs = "\n\n".join(f"Doc {i + 1}: {doc[:1600]}" for i, doc in enumerate(chosen["docs"]))
                prompt = (
                    "Return ONLY JSON with exactly these fields: "
                    '{"episode_summary":"...","episode_skill":"...","step_skills":{"0":"..."}}. '
                    "Read the evidence for the question. episode_skill must be one short factual "
                    "reasoning instruction for a student, grounded in these documents: say which relation "
                    "or comparison to verify and which document entity to use. Do not give a final answer, "
                    "do not state an answer candidate, and do not use XML tags.\n"
                    f"Question:\n{row['question'][:1800]}\nCorrective search:\n{chosen['query']}\nEvidence:\n{docs}")
                try:
                    skill = request(args.teacher_url, prompt)
                    handoff = safe_handoff(str(skill.get("episode_skill", "")), row["question"], row["answers"])
                    if handoff:
                        chosen = dict(chosen)
                        chosen["handoff"] = handoff
                        row["new"] = [chosen]
                    else:
                        row["new"] = [chosen]
                        row["new"][0]["handoff_rejected"] = True
                except Exception as error:
                    row["new"] = [chosen]
                    row["new"][0]["handoff_error"] = f"{type(error).__name__}: {error}"[:250]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row["uid"], flush=True)


if __name__ == "__main__":
    main()
