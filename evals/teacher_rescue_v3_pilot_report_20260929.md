# Teacher rescue v3 pilot, 2026-09-29

**Correction:** The student continuation evaluator used for the EM figures below did not truncate model text at the first completed tool action, unlike the production rollout. Those student EM figures are superseded by the corrected full-corpus evaluation in `teacher_rescue_full_hybrid_pilot_report_20260929.md`. The retrieval-only measurements remain valid.

## Decision

Do not launch a new training run yet. The proposed change to the teacher prompt and intervention point did not clear the evidence-quality gate. Evaluate hybrid retrieval on the same questions first.

## Setup

- 24 all-wrong prompts from the existing SFT350-start rescue rollout, selected before looking at teacher results.
- Current arm: one DeepSeek `deepseek-chat` corrective query at the first student search.
- Proposed arm: DeepSeek sees the failed search result and preceding student context; three candidate queries are retrieved with the same E5 endpoint and `topk=3`.
- Additional no-gold prompt variant: two candidates for 16 of these prompts.
- Query validity follows the current rescue implementation's answer-leak rule. Evidence hit is a case-insensitive literal match to any gold answer alias in the returned top-3 documents. This is a strict diagnostic, not a complete semantic-support metric.
- Student suffix check: 10 paired prompts with valid queries in both arms, same SFT350 checkpoint, greedy vLLM decoding, three-action-turn limit and 500-token observations. This is a small diagnostic, not a full validation result.

## Results

| Arm | Generated queries | Valid queries | Literal answer in top-3 evidence |
| --- | ---: | ---: | ---: |
| Current first-search teacher | 24 | 18 | 0 |
| Context-aware teacher, three candidates | 72 | 51 | 0 |
| No-gold teacher, two candidates | 32 | 25 | 0 |

In the 10 paired student suffixes, current-arm EM was 0/10 and proposed-arm EM was 1/10. The one proposed-arm success used a search about the other entity in a comparison question; the generated reasoning attributed a birth year from an unrelated retrieved biography to the target. It is therefore not convincing evidence that the intervention supplied the necessary fact.

The original all-wrong rollout snapshot contains 138 groups (828 trajectories). A literal gold-answer check found the answer in the first leaf's first observation for 4/138 groups, and in any observation for 30/828 trajectories. This is consistent with retrieval being a major bottleneck.

As a separate reachability probe, a direct gold-entity query found its own answer in top-3 E5 results for only 6/21 eligible prompts. For `Sar-i Sang`, `White Wilderness`, and `One57`, an exact-entity query still missed the entity in E5 top-1000. Increasing E5 top-k alone cannot recover a document absent from that pool.

## Interpretation and next gate

The current E5 service has poor recall for exact names and titles on these hard examples. A BM25 channel against the **same wiki-18 corpus** is a plausible remedy, but it has not been measured here. The repository has a `BM25Retriever` implementation requiring a Pyserini/Lucene index; no local BM25 index or service was found. The E5 endpoint at local port 8000 is an SSH tunnel to another host, so the corpus/index cannot be inspected locally through that endpoint.

Before a new training run, compare E5, BM25, and E5+BM25 candidate union/reranking on these 24 questions. Require a material increase in answer-bearing or human-verified supporting evidence and then rerun the paired student suffix test. Log actual teacher queries, returned documents, student suffixes, and whether each rescued row enters the optimizer. Fix the leakage rule for answers already literally present in the question (for example comparison questions), while keeping genuinely privileged answers out of teacher queries.

Data: `teacher_rescue_v3_query_pilot_20260929.jsonl`, `teacher_rescue_v3_nogold_pilot_20260929.jsonl`, and `teacher_rescue_v3_student_pilot_20260929.jsonl` in this directory.
