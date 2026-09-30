# Full-corpus teacher-rescue follow-up, 2026-09-29

## Decision

Do not start a replacement SFT350 training run from these results alone. The best no-gold intervention improves frozen-student greedy EM on a matched hard set from 3/21 to 5/21, but the stochastic pass@4 result is only 4/21 to 5/21. The gain is real on this sample but too small and too uncertain to justify replacing the completed rescue run.

## Matched setup

- 24 preselected all-wrong groups from the SFT350-start rescue rollout; 21 have a valid query in both arms.
- Retrieval is the separate local port 8002 service: full 21,015,324-document BM25 plus the existing E5 channel, 1:4 dense:BM25 fusion. The original 8000 and 8001 services were not changed.
- The frozen student is `global_step_350`. Its generated text is truncated at the first complete action exactly as production does.
- Candidate query selection uses only question/document term overlap. It does not inspect answer aliases.

## Results

| Intervention after an all-wrong first search | Greedy EM, paired | Stochastic pass@4, paired |
| --- | ---: | ---: |
| Current one-query teacher rescue baseline | 3/21 | 4/21 |
| Adaptive no-gold query | 4/21 | 5/21 with a generic handoff |
| Adaptive query + evidence-relation handoff | **5/21** | **5/21** |
| Teacher selects a candidate set + handoff | 4/21 | not run |
| Teacher executes a second bridge search | 2/21 | not run |
| Student is forced to execute the bridge query | 3/21 | not run |

The relation handoff is a short teacher `<think>` after the retrieved observation. It names the relation and document entity to inspect, but is rejected if it contains a gold alias absent from the question. It helped two cases that require distinguishing related entities: the Moonrunners father relation and the Ammobium/Sidalcea classification. It still fails where evidence is missing, truncated, or requires a multi-hop inference the SFT350 student does not perform.

## Why the tested two-hop intervention failed

This result does not establish that every second teacher search is harmful. The tested version reused the one-hop handoff, which tells the student to verify the unresolved relation. After two teacher searches, all 24 greedy students still chose a third search as their first action, rather than synthesizing an answer. It then produced only 7 final answers and 1 format-valid trajectory, compared with 13 final answers and 8 format-valid trajectories for the one-hop evidence-handoff variant. Its EM was 2/21.

The second bridge query also frequently retrieved poor top-3 evidence even when it was syntactically valid: for example, the Patryk Chojnowski and Sidalcea bridge searches returned pages about beam search, while the Kim Warwick query returned unrelated Kim pages. Therefore the extra turn both consumes context/action budget and often fails to add the missing fact. Forcing the bridge query to be student-generated restored the normal action count but still reached only 3/21, because it replaced answer synthesis with another uncertain retrieval.

Any future two-hop test needs a separate post-second-search handoff that explicitly instructs answer synthesis, and must run only when the second retrieval passes an evidence-quality gate.

## Next experiment gate

The only viable direction so far is:

1. full-corpus hybrid retrieval;
2. one adaptive no-gold corrective query after the failed first observation;
3. one short no-gold evidence-relation handoff; and
4. one student suffix rollout.

Before training this variant, evaluate it on a larger held-out all-wrong sample and require a stable improvement in both greedy EM and pass@K. Teacher candidate ranking and a second teacher retrieval should not be added.

## Artifacts

- Query generation: `teacher_rescue_full_hybrid_adaptive_queries_20260929.jsonl`
- Best handoff inputs: `teacher_rescue_full_hybrid_adaptive_distilled_20260929.jsonl`
- Best greedy continuation result: `teacher_rescue_full_hybrid_adaptive_distilled_student_20260929.jsonl`
- Best pass@4 result: `teacher_rescue_full_hybrid_adaptive_distilled_k4_student_20260929.jsonl`
