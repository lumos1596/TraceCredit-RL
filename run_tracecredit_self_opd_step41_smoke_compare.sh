#!/usr/bin/env bash
# One matched optimizer step from the shared step-40 state, then deterministic
# natural-120 evaluation against the already-produced grouped TraceCredit step41.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RUN_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-tracecredit-self-opd-smoke-step41-20260918}
TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_tracecredit_self_opd_step40_to50.sh"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
BASELINE_CHECKPOINT="$PROJECT_DIR/verl_checkpoints/multihop-tree-dapo-branch-credit-grouped-step40to50-20260918/actor/global_step_41"
OPD_CHECKPOINT="$PROJECT_DIR/verl_checkpoints/$RUN_NAME/actor/global_step_41"
RESULT_ROOT="$PROJECT_DIR/evaluation/formal_em/tracecredit_self_opd_step41_smoke"

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
validate_checkpoint "$BASELINE_CHECKPOINT"

EXPERIMENT_NAME="$RUN_NAME" \
TOTAL_TRAINING_STEPS=42 \
SAVE_FREQ=1 \
SAVE_AT_END=true \
SELF_OPD_MAX_QUERY_TOKENS=${SELF_OPD_MAX_QUERY_TOKENS:-32} \
    "$TRAIN_SCRIPT"
validate_checkpoint "$OPD_CHECKPOINT"

for variant in tracecredit tracecredit_self_opd; do
    if [[ "$variant" == tracecredit ]]; then
        checkpoint=$BASELINE_CHECKPOINT
    else
        checkpoint=$OPD_CHECKPOINT
    fi
    LABEL="${variant}_grouped_step41_smoke" \
    MODEL_PATH="$SFT_MODEL" \
    CHECKPOINT="$checkpoint" \
    CHECKPOINT_STEP=41 \
    EVAL_SPLIT=natural \
    SAMPLE_COUNT=120 \
    RESULT_ROOT="$RESULT_ROOT/$variant" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
        "$EVAL_SCRIPT"
done

"$PROJECT_DIR/.conda/envs/treegrpo/bin/python" - "$RESULT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
baseline = json.loads((root / "tracecredit/natural_n120/result.json").read_text())
opd = json.loads((root / "tracecredit_self_opd/natural_n120/result.json").read_text())
summary = {
    "tracecredit_micro_em": baseline["micro_em"],
    "tracecredit_self_opd_micro_em": opd["micro_em"],
    "delta_micro_em": opd["micro_em"] - baseline["micro_em"],
    "sample_count": baseline["sample_count"],
    "checkpoint_step": 41,
    "scope": "one-update feasibility smoke; not a conclusive efficacy estimate",
}
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "TRACECREDIT_SELF_OPD_SMOKE_COMPLETE=$(date -Is)"
