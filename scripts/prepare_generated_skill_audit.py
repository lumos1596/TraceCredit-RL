#!/usr/bin/env python3
"""Convert held-out local skill generations into the likelihood-audit format."""

import argparse
import json
import re
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generated", required=True); p.add_argument("--output", required=True)
    p.add_argument("--strict-answer-agnostic", action="store_true")
    args = p.parse_args()
    rows = []
    for row in read(args.generated):
        skill = row.get("sft_node_skill")
        if not isinstance(skill, dict):
            continue
        if args.strict_answer_agnostic:
            combined = "\n".join(str(skill.get(k, "")) for k in sorted(skill))
            if re.search(r"\b(?:19|20)\d{2}\b|\b\d{1,4}\b", combined):
                continue
        out = dict(row)
        out["node_skill"] = skill
        out.pop("sft_raw_response", None)
        out.pop("sft_node_skill", None)
        out.pop("sft_rejection_reasons", None)
        rows.append(out)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": len(rows), "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
