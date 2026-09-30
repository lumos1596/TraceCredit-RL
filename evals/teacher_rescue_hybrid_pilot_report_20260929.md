# Teacher rescue with the 8001 hybrid retriever, 2026-09-29

**Correction:** The student continuation evaluator used for the EM figures below did not truncate model text at the first completed tool action, unlike the production rollout. Those student EM figures are superseded by the corrected full-corpus evaluation in `teacher_rescue_full_hybrid_pilot_report_20260929.md`. The retrieval-only measurements remain valid.

## Decision on the current 100k-document service

Do not retrain by changing only `RETRIEVER_URL` from port 8000 to 8001. The hard-prompt pilot has not shown a meaningful improvement in retrieval evidence or final answers.

## Matched evaluation

- The same 24 all-wrong prompts and exact teacher queries from the E5 pilot were replayed through the 8001 service. The request used only `queries`, `topk=3`, and `return_scores=true`; the service used its default hybrid mode and 1:4 dense:BM25 weights.
- Current first-search teacher: 18 valid queries, **0/18** with a literal answer alias in the top-3 documents.
- Context-aware teacher: 51 valid queries across three candidates per question, **0/51** with a literal answer alias in top-3.
- With top-30, the counts became 2/18 and 6/51, but inspected matches included unrelated documents and common answer strings. They do not establish supporting evidence, and the student receives only top-3 under the current 500-token observation budget.
- Frozen SFT350 greedy student continuations on the same 17 paired prompts: with E5, the current arm scored **0/17 EM** and the context-aware arm **1/17**; with the 100k-document hybrid service, both arms scored **1/17**. The hybrid service did not improve the proposed arm. This is a diagnostic pilot, not a full training comparison.

In a larger local replay of the first student search from 156 all-wrong groups, literal answer hits in top-3 were **5/156 for E5** and **5/156 for hybrid** (one gained, one lost).

## Corpus coverage finding

The 8001 service's `/health` reports **100,000 documents**. Its BM25 FTS index contains no occurrence of `Sar-i Sang`, `One57`, or `White Wilderness`; direct queries for these names therefore cannot retrieve their answer-bearing pages from this index. The remote machine also has a complete wiki-18 JSONL with 21,015,324 documents. A low-priority full-corpus BM25 index build was started in remote tmux session `full_bm25_build` on 2026-09-29. Its log is `/home/luwa/Documents/Tree-GRPO/verl_log/full_bm25_build_20260929.log` on the service host. This does not change the existing 8000 or 8001 services.

## Next gate

Once the full BM25 index is complete, serve it on a separate port with the current E5 channel, rerun these exact queries and student continuations, and proceed to SFT350 training only if evidence support and paired student EM improve. Also log each rescue's query, selected documents, student suffix, and whether DAPO retains the rescued row.

Artifacts: `teacher_rescue_hybrid_retrieval_pilot_20260929.jsonl`, `teacher_rescue_hybrid_student_pilot_20260929.jsonl`, and `teacher_rescue_e5_student_full24_20260929.jsonl` in this directory. The replay tools are `scripts/evaluation/reevaluate_teacher_rescue_retriever.py` and `scripts/evaluation/evaluate_teacher_rescue_student.py`.
