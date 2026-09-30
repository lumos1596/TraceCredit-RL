# DeepSeek V4 Flash closed-loop teacher rescue pilot

## Setup

- Same 24 preselected all-wrong questions and full-corpus hybrid retriever as the earlier pilot.
- Teacher: `deepseek-v4-flash`, thinking enabled, reasoning effort high.
- At most two search tool calls. After each call, the returned top-3 documents are sent back to the teacher.
- Teacher must decompose the relation chain, mark resolved and unresolved relations, decide whether evidence is sufficient, and provide a no-answer handoff.
- Frozen student: SFT350, production-compatible action truncation.
- If evidence is sufficient, the handoff explicitly tells the student to stop searching and synthesize the answer.

## Results

The Flash teacher completed 24/24 cases and marked 13/24 as evidence-sufficient. On the 18 questions with a valid current-baseline arm:

| Arm | Strict EM | Answered | Format valid |
| --- | ---: | ---: | ---: |
| Current rescue baseline | 2/18 | 5/18 | 5/18 |
| Earlier adaptive query + handoff | 3/18 | 8/18 | 5/18 |
| Flash closed loop + answer-now handoff | **5/18** | **15/18** | **9/18** |

Among the 9 matched cases that Flash marked evidence-sufficient:

- strict EM: 5/9;
- semantically correct under manual inspection: 8/9;
- all 9 produced an answer;
- 6/9 passed the strict trajectory-format check.

The three additional semantic successes rejected by strict EM were `One57` versus a full article-style target, `125` versus `125 state representatives`, and `August 26, 1910` versus `1910`. The one genuine student failure under a complete verified prefix was Ringo Lam versus Roberto Benigni: the evidence contained 1955 and 1952, but the student selected the younger person.

Across all 13 evidence-sufficient cases, strict EM was 7/13. Manual semantic correctness was 11/13; the additional genuine failure was Moonrunners, where the student answered Robert Mitchum instead of James Mitchum.

For the 9 matched cases that Flash marked insufficient, strict EM was 0/9. This makes the teacher's sufficiency gate useful: it cleanly separates productive rescues from trajectories that should not enter DAPO as promoted rescue rows.

## Decision

Flash is adequate for the next implementation; a Pro run is not necessary yet. The closed loop improves the same-subset strict EM from 3/18 to 5/18, and its evidence-sufficient gate gives 5/9 strict and 8/9 semantic student success. The production variant should:

1. allow at most two teacher search/tool turns;
2. feed each retrieval result back to Flash;
3. retain only `evidence_sufficient=true` rescues;
4. insert valid teacher think/search transitions;
5. tell the student to answer immediately after a verified complete chain; and
6. log strict EM and normalized semantic aliases separately.

Artifacts:

- `teacher_rescue_flash_closed_loop_20260929.jsonl`
- `teacher_rescue_flash_closed_loop_answer_now_valid_student_20260929.jsonl`
