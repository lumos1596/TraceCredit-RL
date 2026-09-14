#!/usr/bin/env python3
"""Build mixed multi-hop train/evaluation parquet files for TraceCredit-RL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets


PROMPT_TEMPLATE = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


def convert_file(
    path: Path,
    data_source: str,
    split: str,
    cache_dir: Path,
) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "json",
        data_files=str(path),
        split="train",
        cache_dir=str(cache_dir),
    )

    def convert(example: dict, index: int) -> dict:
        question = example["question"].strip()
        if not question.endswith("?"):
            question += "?"
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": example["golden_answers"]},
            },
            # UID grouping in the tree-rollout trainer is keyed by this value. Prefixing the
            # per-file index prevents cross-dataset collisions in a mixture.
            "extra_info": {"split": split, "index": f"{data_source}:{index}"},
        }

    return dataset.map(
        convert,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Preparing {data_source}/{split}",
    )


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing FlashRAG input: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-total", type=int, default=37500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-per-source", type=int, default=120)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / ".hf_cache"

    if args.train_total <= 0:
        raise ValueError("--train-total must be positive")

    # 40/40/20 keeps the two large benchmarks equally represented and gives
    # MuSiQue enough weight without sampling any source with replacement.
    train_specs = [
        ("hotpotqa", "hotpotqa/train.jsonl", 0.4),
        ("2wikimultihopqa", "2wikimultihopqa/train.jsonl", 0.4),
        ("musique", "musique/train.jsonl", 0.2),
    ]
    train_parts = []
    train_counts = {}
    assigned = 0
    for idx, (name, relative, fraction) in enumerate(train_specs):
        source = convert_file(require_file(args.raw_dir / relative), name, "train", cache_dir)
        requested = args.train_total - assigned if idx == len(train_specs) - 1 else int(args.train_total * fraction)
        if requested > len(source):
            raise ValueError(f"{name} has {len(source)} rows but the mixture requests {requested}")
        source = source.shuffle(seed=args.seed + idx).select(range(requested))
        train_parts.append(source)
        train_counts[name] = requested
        assigned += requested
    train = datasets.concatenate_datasets(train_parts).shuffle(seed=args.seed)
    validation_specs = [
        ("hotpotqa", "hotpotqa/dev.jsonl"),
        ("2wikimultihopqa", "2wikimultihopqa/dev.jsonl"),
        ("musique", "musique/dev.jsonl"),
        ("bamboogle", "bamboogle/test.jsonl"),
    ]
    validation = datasets.concatenate_datasets(
        [
            convert_file(require_file(args.raw_dir / relative), name, "test", cache_dir)
            for name, relative in validation_specs
        ]
    )

    balanced_parts = []
    for idx, (name, relative) in enumerate(validation_specs):
        source = convert_file(require_file(args.raw_dir / relative), name, "balanced_test", cache_dir)
        count = min(args.balanced_per_source, len(source))
        balanced_parts.append(source.shuffle(seed=args.seed + 100 + idx).select(range(count)))
    balanced_validation = datasets.concatenate_datasets(balanced_parts).shuffle(seed=args.seed)

    train.to_parquet(args.output_dir / "train.parquet")
    validation.to_parquet(args.output_dir / "test.parquet")
    balanced_validation.to_parquet(args.output_dir / "balanced_test.parquet")
    manifest = {
        "seed": args.seed,
        "train_total": len(train),
        "train_counts": train_counts,
        "validation_rows": len(validation),
        "balanced_validation_rows": len(balanced_validation),
        "balanced_per_source_requested": args.balanced_per_source,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"train rows: {len(train)} -> {args.output_dir / 'train.parquet'}")
    print(f"test rows: {len(validation)} -> {args.output_dir / 'test.parquet'}")
    print(f"balanced test rows: {len(balanced_validation)} -> {args.output_dir / 'balanced_test.parquet'}")


if __name__ == "__main__":
    main()
