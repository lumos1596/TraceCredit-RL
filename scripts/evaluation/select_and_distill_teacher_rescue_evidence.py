#!/usr/bin/env python3
"""Let a no-gold teacher select one retrieved evidence set and write a short handoff."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


def request(url: str, prompt: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/generate",
        data=json.dumps({"prompt": prompt}, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)["skill"]


def clean_handoff(text: str, question: str, answers: list[str]) -> str:
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
            choices = [x for x in row["new"] if x.get("valid") and x.get("docs")]
            if choices:
                rendered = []
                for index, choice in enumerate(choices, 1):
                    docs = "\n".join(f"Doc {j + 1}: {doc[:1200]}" for j, doc in enumerate(choice["docs"]))
                    rendered.append(f"CANDIDATE {index}\nQuery: {choice['query']}\n{docs}")
                prompt = (
                    "Return ONLY JSON with exactly these fields: "
                    '{"episode_summary":"...","episode_skill":"...","step_skills":{"selected_candidate":"N"}}. '
                    "N must be the number of the candidate whose documents best support the missing relation in "
                    "the question. episode_skill must be one short factual reasoning instruction grounded in the "
                    "selected documents: state what relation or comparison to verify and which entity to inspect. "
                    "Do not give a final answer, state an answer candidate, or use XML tags.\n"
                    f"Question:\n{row['question'][:1800]}\nCandidates:\n" + "\n\n".join(rendered))
                try:
                    skill = request(args.teacher_url, prompt)
                    selected = int(str(skill.get("step_skills", {}).get("selected_candidate", "1"))) - 1
                    if not 0 <= selected < len(choices):
                        selected = 0
                    choice = dict(choices[selected])
                    handoff = clean_handoff(str(skill.get("episode_skill", "")), row["question"], row["answers"])
                    if handoff:
                        choice["handoff"] = handoff
                    else:
                        choice["handoff_rejected"] = True
                    row["new"] = [choice]
                    row["teacher_selected_candidate"] = selected + 1
                except Exception as error:
                    row["new"] = [choices[0]]
                    row["new"][0]["selection_error"] = f"{type(error).__name__}: {error}"[:250]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row["uid"], flush=True)


if __name__ == "__main__":
    main()
