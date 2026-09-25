#!/usr/bin/env python3
"""Filter a node-skill consumer JSONL file to event IDs listed in another JSONL."""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    wanted = set()
    for line in open(args.ids, encoding="utf-8"):
        row = json.loads(line)
        if row.get("sft_node_skill"):
            wanted.add(row["event_id"])

    kept = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for line in open(args.source, encoding="utf-8"):
            row = json.loads(line)
            if row.get("event_id") in wanted:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({"wanted": len(wanted), "kept": kept}, ensure_ascii=False))


if __name__ == "__main__":
    main()
