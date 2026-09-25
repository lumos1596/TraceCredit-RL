#!/usr/bin/env bash
# One-step 3B smoke for the TraceCredit-adapted OPD V2, followed by the same
# deterministic natural-120 evaluation used by the baseline and V1 smoke.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RUN_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-tracecredit-guided-opd-v2-step41-20260919}
TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_tracecredit_self_opd_step40_to50.sh"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
OPD_CHECKPOINT="$PROJECT_DIR/verl_checkpoints/$RUN_NAME/actor/global_step_41"
RESULT_ROOT="$PROJECT_DIR/evaluation/formal_em/tracecredit_guided_opd_v2_step41_smoke"
PRIOR_ROOT="$PROJECT_DIR/evaluation/formal_em/tracecredit_self_opd_step41_smoke"

validate_checkpoint() {
    local checkpoint=$1 rank kind
    for kind in model optim extra_state; do
        for rank in 0 1 2; do
            [[ -s "$checkpoint/${kind}_world_size_3_rank_${rank}.pt" ]] || {
                echo "missing or empty checkpoint shard: $checkpoint/${kind}_world_size_3_rank_${rank}.pt" >&2
                return 1
            }
        done
    done
}

cd "$PROJECT_DIR"
EXPERIMENT_NAME="$RUN_NAME" \
TOTAL_TRAINING_STEPS=42 \
SAVE_FREQ=1 \
SAVE_AT_END=true \
SELF_OPD_COEF=${SELF_OPD_COEF:-0.0003} \
SELF_OPD_MAX_TEACHER_LENGTH=${SELF_OPD_MAX_TEACHER_LENGTH:-768} \
SELF_OPD_MAX_QUERY_TOKENS=${SELF_OPD_MAX_QUERY_TOKENS:-32} \
SELF_OPD_MAX_EVIDENCE_TOKENS=${SELF_OPD_MAX_EVIDENCE_TOKENS:-96} \
SELF_OPD_MIN_VALUE_GAP=${SELF_OPD_MIN_VALUE_GAP:-0.25} \
SELF_OPD_MIN_RAW_ADVANTAGE=${SELF_OPD_MIN_RAW_ADVANTAGE:-0.0} \
SELF_OPD_MAX_ADVANTAGE_WEIGHT=${SELF_OPD_MAX_ADVANTAGE_WEIGHT:-3.0} \
    "$TRAIN_SCRIPT"

validate_checkpoint "$OPD_CHECKPOINT"

LABEL=tracecredit_guided_opd_v2_step41_smoke \
MODEL_PATH="$SFT_MODEL" \
CHECKPOINT="$OPD_CHECKPOINT" \
CHECKPOINT_STEP=41 \
EVAL_SPLIT=natural \
SAMPLE_COUNT=120 \
RESULT_ROOT="$RESULT_ROOT/v2" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
    "$EVAL_SCRIPT"

"$PROJECT_DIR/.conda/envs/treegrpo/bin/python" - "$PRIOR_ROOT" "$RESULT_ROOT" <<'PY'
import json
import pathlib
import sys

prior = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
baseline = json.loads((prior / "tracecredit/natural_n120/result.json").read_text())
v1 = json.loads((prior / "tracecredit_self_opd/natural_n120/result.json").read_text())
v2 = json.loads((root / "v2/natural_n120/result.json").read_text())
summary = {
    "tracecredit_micro_em": baseline["micro_em"],
    "self_opd_v1_micro_em": v1["micro_em"],
    "tracecredit_guided_opd_v2_micro_em": v2["micro_em"],
    "v2_delta_vs_tracecredit": v2["micro_em"] - baseline["micro_em"],
    "v2_delta_vs_v1": v2["micro_em"] - v1["micro_em"],
    "sample_count": v2["sample_count"],
    "checkpoint_step": 41,
    "scope": "one-update monitored feasibility smoke; not a conclusive efficacy estimate",
}
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "TRACECREDIT_GUIDED_OPD_V2_SMOKE_COMPLETE=$(date -Is)"
