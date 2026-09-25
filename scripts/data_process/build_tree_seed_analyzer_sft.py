#!/usr/bin/env python3
"""Build a small, audited SEED-style analyzer SFT pilot from tree rollouts.

The input is the project's ``*_selected.jsonl`` rollout dump.  Each selected
leaf already contains the original question prompt, the complete interaction
trajectory, the terminal score, and the gold answer used only by the local
leakage validator.  Gold answers are redacted before the external analyzer is
called and are never serialized into output artifacts.

The API key is read only from ``DEEPSEEK_API_KEY``.  It is never accepted as a
CLI argument and is scrubbed from exception messages before they are written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_GLOB = (
    "rollouts/multihop-tree-dapo-branch-credit-step20to30-retry-20260906-092827/"
    "step_000021_chunk_*_selected.jsonl"
)
DEFAULT_OUTPUT_DIR = "data/tree_seed_analyzer_sft_pilot_deepseek"


def normalize(text: object) -> str:
    return " ".join(re.findall(r"[\w]+", str(text or "").casefold(), flags=re.UNICODE))


def phrase_present(text: str, phrase: str) -> bool:
    needle = normalize(phrase)
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", normalize(text)) is not None


def answer_aliases(ground_truth: object) -> list[str]:
    if isinstance(ground_truth, dict):
        raw = ground_truth.get("target", [])
    else:
        raw = ground_truth
    if not isinstance(raw, list):
        raw = [raw]
    aliases = []
    for item in raw:
        value = str(item or "").strip()
        if value and value not in aliases:
            aliases.append(value)
    return aliases


def redact_answers(text: str, aliases: Iterable[str]) -> str:
    result = str(text or "")
    # The final answer action itself is never useful to an answer-free analyzer.
    result = re.sub(
        r"<answer>.*?</answer>",
        "<answer>[REDACTED]</answer>",
        result,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for alias in sorted({str(x).strip() for x in aliases if str(x).strip()}, key=len, reverse=True):
        # Short answers such as "no" and "yes" must not corrupt words like
        # "known" or "yesterday".  Unicode word guards also work for longer
        # entity/date answers while retaining punctuation around the phrase.
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        result = re.sub(pattern, "[REDACTED]", result, flags=re.IGNORECASE)
    return result


def extract_question(prompt: str) -> str:
    matches = list(re.finditer(r"Question:\s*(.*?)(?:\n<\|im_end\|>|$)", prompt, re.DOTALL))
    return " ".join(matches[-1].group(1).split()) if matches else ""


def extract_queries(trajectory: str) -> list[str]:
    return [" ".join(x.split()) for x in re.findall(
        r"<search>(.*?)</search>", trajectory, flags=re.IGNORECASE | re.DOTALL
    ) if x.strip()]


def extract_titles(trajectory: str) -> list[str]:
    return [" ".join(x.split()) for x in re.findall(
        r"Title:\s*[\"']([^\"']+)[\"']", trajectory, flags=re.IGNORECASE
    ) if x.strip()]


def iter_jsonl(paths: Iterable[str]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source_path"] = path
                row["_source_line"] = line_number
                yield row


def stable_id(row: dict[str, Any]) -> str:
    payload = f"{row.get('tree_uid')}:{row.get('node_uid')}:{row.get('uid')}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def sample_rows(
    rows: list[dict[str, Any]], limit: int, seed: int, max_per_question: int = 1
) -> list[dict[str, Any]]:
    """Choose a score-balanced subset with a per-question rollout cap."""
    if max_per_question <= 0:
        raise ValueError("max_per_question must be positive")
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    for row in rows:
        score = float(row.get("original_score", 0.0))
        bucket = "low" if score <= 0.25 else "high" if score >= 0.75 else "mid"
        buckets[bucket].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    chosen: list[dict[str, Any]] = []
    uid_counts: Counter[str] = Counter()
    order = ["low", "high", "mid"]
    while len(chosen) < limit and any(buckets.values()):
        progressed = False
        for bucket in order:
            while buckets[bucket]:
                row = buckets[bucket].pop()
                uid = str(row.get("uid") or row.get("tree_uid"))
                if uid_counts[uid] >= max_per_question:
                    continue
                chosen.append(row)
                uid_counts[uid] += 1
                progressed = True
                break
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return chosen


def build_prompt(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    aliases = answer_aliases(row.get("ground_truth"))
    question = redact_answers(extract_question(str(row.get("prompt", ""))), aliases)
    trajectory = redact_answers(str(row.get("response", "")), aliases)
    score = float(row.get("original_score", 0.0))
    outcome = "success" if score >= 0.99 else "partial" if score > 0.0 else "failure"
    prompt = f"""Analyze the following search-agent episode and return ONLY one valid JSON object.

Write two fields:
1. episode_summary: a concise procedural summary of what the agent did well or poorly. Do not summarize factual answers.
2. episode_skill: one short, policy-facing and reusable search skill. For a successful or partially successful trajectory, extract the useful workflow and any remaining verification gap. For a failed trajectory, extract the core mistake and an avoidance rule.

Strict constraints:
- Do not state, guess, imply, or reconstruct the final answer.
- Do not copy any search query, document title, retrieved sentence, date, number, or answer-bearing fact.
- Do not mention gold evidence, reference answers, sibling branches, rewards, scores, or hindsight.
- Express only entity/relation/evidence-gap diagnosis and a general search or verification strategy.
- The skill must be usable before knowing the answer and should start with "Workflow:" or "Avoid:".
- Return exactly these top-level fields and no markdown:
{{"episode_summary":"string","episode_skill":"string"}}

Episode context:
- Task description: {question}
- episode_outcome: {outcome}
- Interaction trajectory:
{trajectory}
"""
    private = {
        "aliases": aliases,
        "queries": extract_queries(trajectory),
        "titles": extract_titles(trajectory),
        "question": question,
        "outcome": outcome,
        "score": score,
    }
    return prompt, private


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced + [raw]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("no JSON object found")


def validate_output(parsed: dict[str, Any], private: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if set(parsed) != {"episode_summary", "episode_skill"}:
        reasons.append("schema")
    summary = str(parsed.get("episode_summary", "")).strip()
    skill = str(parsed.get("episode_skill", "")).strip()
    combined = f"{summary}\n{skill}"
    if not summary or not skill:
        reasons.append("empty_field")
    if not re.match(r"^(Workflow|Avoid):", skill, flags=re.IGNORECASE):
        reasons.append("skill_prefix")
    if len(normalize(skill).split()) < 8 or len(normalize(skill).split()) > 100:
        reasons.append("skill_length")
    if any(phrase_present(combined, alias) for alias in private["aliases"]):
        reasons.append("answer_leak")
    if any(len(normalize(query).split()) >= 3 and phrase_present(combined, query)
           for query in private["queries"]):
        reasons.append("query_copy")
    if any(len(normalize(title).split()) >= 2 and phrase_present(combined, title)
           for title in private["titles"]):
        reasons.append("title_copy")
    if re.search(
        r"\b(?:the|final|correct|reference) answer\s+(?:is|was|would be|should be)\b",
        combined,
        re.I,
    ):
        reasons.append("answer_claim")
    if re.search(r"\bredacted\b|\[redacted\]", combined, re.I):
        reasons.append("redaction_artifact")
    if re.search(r"\b(?:reward|score|gold|sibling|hindsight)\b", combined, re.I):
        reasons.append("privileged_meta")
    return sorted(set(reasons))


def safe_error(exc: BaseException, api_key: str) -> str:
    return str(exc).replace(api_key, "<redacted>")[:1000]


def deepseek_completion(
    *, api_key: str, base_url: str, model: str, prompt: str, timeout: float, retries: int
) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }).encode()
    endpoint = base_url.rstrip("/") + "/chat/completions"
    last_error: BaseException | None = None
    for attempt in range(max(1, retries)):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            return str(payload["choices"][0]["message"]["content"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(8.0, 2.0 ** attempt))
    raise RuntimeError(safe_error(last_error or RuntimeError("unknown API error"), api_key))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_one(row: dict[str, Any], args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    prompt, private = build_prompt(row)
    record: dict[str, Any] = {
        "sample_id": stable_id(row),
        "uid": row.get("uid"),
        "data_source": row.get("data_source"),
        "tree_uid": row.get("tree_uid"),
        "node_uid": row.get("node_uid"),
        "source_path": row.get("_source_path"),
        "source_line": row.get("_source_line"),
        "source_score": private["score"],
        "source_outcome": private["outcome"],
        "prompt": prompt,
        "model": args.model,
    }
    try:
        raw = deepseek_completion(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            timeout=args.timeout,
            retries=args.retries,
        )
        parsed = extract_json_object(raw)
        reasons = validate_output(parsed, private)
        record.update({
            "raw_response": raw,
            "response": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
            "episode_summary": str(parsed.get("episode_summary", "")).strip(),
            "episode_skill": str(parsed.get("episode_skill", "")).strip(),
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "error": None,
        })
    except Exception as exc:
        record.update({
            "raw_response": "",
            "response": "",
            "episode_summary": "",
            "episode_skill": "",
            "accepted": False,
            "rejection_reasons": ["api_or_parse_error"],
            "error": safe_error(exc, api_key),
        })
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-per-question", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"no files matched: {args.input_glob}")
    rows = list(iter_jsonl(paths))
    selected = sample_rows(rows, args.limit, args.seed, args.max_per_question)
    if len(selected) < args.limit:
        raise RuntimeError(f"requested {args.limit} unique questions, found {len(selected)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidates": output_dir / "candidates.jsonl",
        "accepted": output_dir / "accepted.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "sft": output_dir / "sft_pilot.jsonl",
        "summary": output_dir / "summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("output exists; pass --overwrite: " + ", ".join(map(str, existing)))
    for path in existing:
        path.unlink()

    if args.dry_run:
        for row in selected:
            prompt, _ = build_prompt(row)
            append_jsonl(outputs["candidates"], {
                "sample_id": stable_id(row), "uid": row.get("uid"), "prompt": prompt
            })
        print(json.dumps({"dry_run": True, "selected": len(selected), "output": str(output_dir)}))
        return 0

    try:
        api_key = os.environ["DEEPSEEK_API_KEY"]
    except KeyError as exc:
        raise RuntimeError("DEEPSEEK_API_KEY is absent; refusing to call the API") from exc
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is empty; refusing to call the API")

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process_one, row, args, api_key) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda x: x["sample_id"])

    for record in records:
        append_jsonl(outputs["candidates"], record)
        target = outputs["accepted"] if record["accepted"] else outputs["rejected"]
        append_jsonl(target, record)
        if record["accepted"]:
            append_jsonl(outputs["sft"], {
                "prompt": record["prompt"],
                "response": record["response"],
                "sample_id": record["sample_id"],
                "uid": record["uid"],
                "data_source": record["data_source"],
                "source_score": record["source_score"],
                "source_outcome": record["source_outcome"],
            })

    reasons = Counter(reason for record in records for reason in record["rejection_reasons"])
    accepted = sum(bool(record["accepted"]) for record in records)
    summary = {
        "model": args.model,
        "input_glob": args.input_glob,
        "selected": len(records),
        "accepted": accepted,
        "rejected": len(records) - accepted,
        "acceptance_rate": accepted / len(records) if records else 0.0,
        "outcome_counts": dict(Counter(record["source_outcome"] for record in records)),
        "data_source_counts": dict(Counter(record["data_source"] for record in records)),
        "rejection_reasons": dict(reasons),
        "candidate_answer_leak_rate": reasons.get("answer_leak", 0) / len(records) if records else 0.0,
        "accepted_set_leak_free": all(
            "answer_leak" not in record["rejection_reasons"] for record in records if record["accepted"]
        ),
        "quality_gate_pass": accepted >= max(8, int(0.75 * len(records))),
        "note": (
            "Gate applies to the filtered accepted set. Rejected rows, including answer leaks, "
            "are retained only for audit and never exported as SFT targets."
        ),
    }
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["quality_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
