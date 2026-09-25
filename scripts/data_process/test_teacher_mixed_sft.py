#!/usr/bin/env python3
"""Offline tests for the bounded teacher-data and SFT plumbing.

This file deliberately imports helpers only.  It never loads a model, starts a
retriever, contacts an API, allocates CUDA, or launches a training/evaluation
process.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_teacher_mixed_sft_dataset as builder  # noqa: E402
import generate_multihop_sft_teacher_production_deepseek as multi  # noqa: E402
import generate_singlehop_sft_teacher_production as single  # noqa: E402


class TeacherMixedOfflineTests(unittest.TestCase):
    def test_builtin_self_tests_cover_exact_synthetic_contract(self) -> None:
        self.assertEqual(single._self_test()["status"], "passed")
        self.assertEqual(multi._self_test()["status"], "passed")
        self.assertEqual(builder.self_test()["status"], "passed")

    def test_resume_appender_is_incremental(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-mixed-test-") as directory:
            path = Path(directory) / "candidates.jsonl"
            writer = single.JSONLAppender(path, append=False)
            writer.append({"candidate_id": "q0__sample_0", "sample_index": 0})
            writer.close()
            writer = single.JSONLAppender(path, append=True)
            writer.append({"candidate_id": "q0__sample_1", "sample_index": 1})
            writer.close()
            records = single.load_jsonl(path)
            self.assertEqual([record["sample_index"] for record in records], [0, 1])

    def test_exact_stop_and_one_accepted_per_question(self) -> None:
        item = {
            "question_id": "q0",
            "question": "Who?",
            "question_key": "who?",
            "data_source": "nq",
        }
        candidates = single._candidate_specs(item, samples_per_question=2)
        records = [{**candidate, "eligible": True} for candidate in candidates]
        accepted = single._select_accepted(records, target=1, question_order=[item])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["candidate_id"], "q0__sample_0")

        pool = {
            "hotpotqa": [{"question_id": "m0", "data_source": "hotpotqa"}],
            "2wikimultihopqa": [],
            "musique": [],
        }
        multi_records = [
            {
                "candidate_id": f"m0__sample_{index}",
                "question_id": "m0",
                "data_source": "hotpotqa",
                "sample_index": index,
                "eligible": True,
            }
            for index in (0, 1)
        ]
        selected = multi.select_accepted(
            multi_records,
            pool,
            {"hotpotqa": 1, "2wikimultihopqa": 0, "musique": 0},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["sample_index"], 0)

    def test_paired_question_is_rejected_on_resume(self) -> None:
        pool = {
            "hotpotqa": [{"question_id": "paired", "data_source": "hotpotqa"}],
            "2wikimultihopqa": [],
            "musique": [],
        }
        record = {
            "candidate_id": "paired__sample_0",
            "question_id": "paired",
            "data_source": "hotpotqa",
            "sample_index": 0,
        }
        with self.assertRaises(ValueError):
            multi.validate_existing_candidates([record], pool, {"paired"})

    def test_retryable_api_history_does_not_consume_candidate_slot(self) -> None:
        pool = {
            "hotpotqa": [{"question_id": "q0", "data_source": "hotpotqa"}],
            "2wikimultihopqa": [],
            "musique": [],
        }
        failure = {
            "candidate_id": "q0__sample_0",
            "question_id": "q0",
            "data_source": "hotpotqa",
            "sample_index": 0,
            "status": "failed",
            "raw_generation_turns": [],
            "failure_reasons": ["deepseek_api_failed", "DeepSeek HTTP 402: insufficient balance"],
            "eligible": False,
        }
        multi.validate_existing_candidates([failure], pool, set())
        self.assertTrue(multi.is_retryable_api_failure(failure))
        self.assertEqual(sum(multi.is_effective_candidate(row) for row in [failure]), 0)
        self.assertTrue(
            multi.has_unresolved_retryable_slots(
                [failure], pool, {"hotpotqa": 1, "2wikimultihopqa": 0, "musique": 0}
            )
        )
        self.assertEqual(multi.systemic_api_failure_reason(failure), "http_402")

        generated = {
            **failure,
            "raw_generation_turns": [{"turn_index": 0}],
            "failure_reasons": ["final_answer_em_failed"],
        }
        multi.validate_existing_candidates([failure, generated], pool, set())
        self.assertEqual(sum(multi.is_effective_candidate(row) for row in [failure, generated]), 1)
        self.assertFalse(
            multi.has_unresolved_retryable_slots(
                [failure, generated], pool, {"hotpotqa": 1, "2wikimultihopqa": 0, "musique": 0}
            )
        )

    def test_duplicate_effective_candidate_id_is_rejected(self) -> None:
        pool = {
            "hotpotqa": [{"question_id": "q0", "data_source": "hotpotqa"}],
            "2wikimultihopqa": [],
            "musique": [],
        }
        generated = {
            "candidate_id": "q0__sample_0",
            "question_id": "q0",
            "data_source": "hotpotqa",
            "sample_index": 0,
            "raw_generation_turns": [{"turn_index": 0}],
            "failure_reasons": [],
        }
        with self.assertRaisesRegex(ValueError, "duplicate effective candidate_id"):
            multi.validate_existing_candidates([generated, dict(generated)], pool, set())

    def test_information_user_turn_is_masked_by_existing_sft_dataset(self) -> None:
        trajectory = (
            "<think>Find evidence.</think><search>author</search>"
            "<information>Doc</information><think>Evidence is enough.</think>"
            "<answer>Ada</answer>"
        )
        messages, _, _ = builder.trajectory_to_messages(
            "Who?", trajectory, expected_searches=(1, 1)
        )

        # Exercise the same SFTDataset implementation used by the trainer,
        # without reading parquet or constructing a GPU process.
        from verl.utils.dataset.sft_dataset import SFTDataset

        dataset = SFTDataset.__new__(SFTDataset)
        dataset.tokenizer = builder.FakeTokenizer()
        dataset.max_length = builder.MAX_LENGTH
        dataset.truncation = "error"
        dataset.data = [{"messages": json.dumps(messages, ensure_ascii=False)}]
        sample = dataset[0]
        prefixes = dataset._template_ids(messages)
        previous = 0
        for message, prefix in zip(messages, prefixes):
            segment = sample["loss_mask"][previous : len(prefix)].tolist()
            self.assertTrue(segment)
            if message["role"] == "user":
                self.assertEqual(set(segment), {0})
            else:
                self.assertEqual(set(segment), {1})
            previous = len(prefix)

    def test_information_compression_preserves_all_non_information_content(self) -> None:
        trajectory = (
            "<think>  Keep this reasoning exactly.  </think><search> exact query </search>"
            "<information>BEGIN-"
            + (" useful evidence" * 180)
            + "-END</information>"
            "<think> final reasoning stays exact </think><answer> final answer </answer>"
        )
        messages, _, _ = builder.trajectory_to_messages(
            "  exact question  ", trajectory, expected_searches=(1, 1)
        )
        tokenizer = builder.FakeTokenizer()
        compressed, metadata = builder.compress_messages_to_budget(
            messages, tokenizer, max_length=800
        )
        compressed_again, metadata_again = builder.compress_messages_to_budget(
            messages, tokenizer, max_length=800
        )

        self.assertTrue(metadata["compressed"])
        self.assertLessEqual(metadata["final_token_length"], 800)
        self.assertEqual(builder.canonical(compressed), builder.canonical(compressed_again))
        self.assertEqual(metadata, metadata_again)
        self.assertEqual(compressed[0], messages[0])
        for before, after in zip(messages, compressed):
            if before["role"] == "assistant":
                self.assertEqual(after, before)
        body = builder._information_body(compressed[2]["content"])
        self.assertIsNotNone(body)
        self.assertIn(builder.COMPRESSION_MARKER, body)
        self.assertTrue(body.startswith("BEGIN-"))
        self.assertTrue(body.endswith("-END"))
        self.assertGreaterEqual(metadata["information_characters_removed"], 0)
        self.assertGreaterEqual(metadata["information_tokens_removed"], 0)

    def test_compression_fails_when_fixed_content_exceeds_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed non-information content alone"):
            builder.compress_messages_to_budget(
                [
                    {"role": "user", "content": "x" * 4000},
                    {"role": "assistant", "content": "answer"},
                ],
                builder.FakeTokenizer(),
                max_length=100,
            )

    def test_dedup_and_tokenizer_length_fail_closed(self) -> None:
        row = {
            "question_id": "q0",
            "question_key": "same question",
            "source_candidate_id": "q0__sample_0",
            "trajectory_sha256": "trajectory-0",
        }
        with self.assertRaises(ValueError):
            builder.validate_unique_rows(
                [row, {**row, "question_id": "q1", "source_candidate_id": "q1__sample_0"}]
            )
        with self.assertRaises(ValueError):
            builder.validate_unique_rows(
                [row, {**row, "question_key": "different question", "question_id": "q1"}]
            )

        too_long = {
            "messages": json.dumps(
                [
                    {"role": "user", "content": "x" * 4000},
                    {"role": "assistant", "content": "ok"},
                ]
            ),
            "source_candidate_id": "too-long__sample_0",
        }
        with self.assertRaises(ValueError):
            builder.validate_tokenization([too_long], builder.FakeTokenizer())

    def test_secret_redaction_does_not_persist_sentinel(self) -> None:
        sentinel = "SECRET_SENTINEL_DO_NOT_PERSIST"
        redacted = multi.fixed._json_line(
            {"authorization": f"Bearer {sentinel}", "nested": sentinel}, sentinel
        )
        self.assertNotIn(sentinel, redacted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
