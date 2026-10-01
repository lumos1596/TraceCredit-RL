#!/usr/bin/env python3
"""Turn a no-gold bridge query into a teacher handoff that the student must execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for line in args.input.open():
            row = json.loads(line)
            first = dict(row["new"][0])
            actions = first.pop("teacher_actions", [])
            bridge = actions[1] if len(actions) > 1 else {}
            if bridge.get("valid") and bridge.get("query"):
                original = first.get("handoff", "Read the first evidence and identify the unresolved relation.")
                first["handoff"] = (f"{original} Before giving any answer, your next action must be exactly one "
                                    f"search that verifies the remaining relation. Use this query: {bridge['query']}.")
            row["new"] = [first]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
