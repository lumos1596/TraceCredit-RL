#!/usr/bin/env python3
"""Replay saved teacher queries against another Search-R1 retriever endpoint.

This only sends search queries to the endpoint. It does not call the teacher or
load a student model, so query-generation differences cannot confound the
retriever comparison.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def retrieve(url: str, queries: list[str], topk: int) -> list[list[str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"queries": queries, "topk": topk, "return_scores": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    results = payload["result"]
    if len(results) != len(queries):
        raise RuntimeError(f"expected {len(queries)} result lists, got {len(results)}")
    return [[doc["document"]["contents"] for doc in row] for row in results]


def answer_hit(docs: list[str], aliases: list[str]) -> bool:
    return any(
        len(alias) > 2 and alias.casefold() in document.casefold()
        for alias in aliases for document in docs
    )


def visible_observation(documents: list[str], tokenizer) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--tokenizer", type=Path,
                        help="Compute answer hits in the student's actual 500-token observation")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)

    rows = [json.loads(line) for line in args.input.open()]
    queries = sorted({
        item["query"]
        for row in rows
        for item in [row["baseline"], *row["new"]]
        if item.get("valid") and item.get("query")
    })
    lookup = {}
    # Small batches keep request bodies and response sizes bounded.
    for start in range(0, len(queries), 32):
        batch = queries[start:start + 32]
        lookup.update(zip(batch, retrieve(args.retriever_url, batch, args.topk)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in rows:
            result = {key: value for key, value in row.items() if key not in ("baseline", "new")}
            for arm in ("baseline", "new"):
                items = [row["baseline"]] if arm == "baseline" else row["new"]
                rescored = []
                for item in items:
                    clean = {key: value for key, value in item.items() if key != "docs"}
                    if item.get("valid") and item.get("query"):
                        docs = lookup[item["query"]]
                        clean["docs"] = docs
                        clean["hit"] = answer_hit(docs, row["answers"])
                        if tokenizer is not None:
                            clean["visible_hit"] = answer_hit(
                                [visible_observation(docs, tokenizer)], row["answers"])
                    rescored.append(clean)
                result[arm] = rescored[0] if arm == "baseline" else rescored
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"questions={len(rows)} distinct_queries={len(queries)} output={args.output}")
    for arm in ("baseline", "new"):
        items = [row["baseline"] for row in rows] if arm == "baseline" else [
            item for row in rows for item in row["new"]
        ]
        valid = sum(bool(item.get("valid")) for item in items)
        hits = sum(bool(item.get("valid") and answer_hit(lookup[item["query"]], row["answers"]))
                   for row in rows
                   for item in ([row["baseline"]] if arm == "baseline" else row["new"]))
        print(f"{arm}: valid={valid}/{len(items)} answer_hit={hits}/{valid}")
    if tokenizer is not None:
        print("visible hits are saved per query in the output JSONL")


if __name__ == "__main__":
    main()
