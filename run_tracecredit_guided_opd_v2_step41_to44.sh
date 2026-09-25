#!/usr/bin/env bash
# Continue the validated V2 checkpoint for three optimizer steps, then evaluate
# every new checkpoint with the same deterministic natural-120 protocol.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
BASE_TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
PYTHON="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/multihop-tree-dapo-tracecredit-guided-opd-v2-step41-20260919/actor/global_step_41"}
RUN_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-tracecredit-guided-opd-v2-step42to44-20260919}
CHECKPOINT_ROOT="$PROJECT_DIR/verl_checkpoints/$RUN_NAME/actor"
ROLLOUT_ROOT="$PROJECT_DIR/rollouts/$RUN_NAME"
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_DIR/evaluation/formal_em/tracecredit_guided_opd_v2_step42to44"}
STEP41_RESULT="$PROJECT_DIR/evaluation/formal_em/tracecredit_guided_opd_v2_step41_smoke/v2/natural_n120/result.json"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}
FINAL_STEP=${FINAL_STEP:-44}

checkpoint_complete() {
    local checkpoint=$1 rank kind
    for kind in model optim extra_state; do
        for ((rank = 0; rank < N_GPUS_PER_NODE; rank++)); do
            [[ -s "$checkpoint/${kind}_world_size_${N_GPUS_PER_NODE}_rank_${rank}.pt" ]] || return 1
        done
    done
}

checkpoint_complete "$BASE_CHECKPOINT" || {
    echo "incomplete base checkpoint: $BASE_CHECKPOINT" >&2
    exit 2
}
[[ -x "$PYTHON" ]] || { echo "missing python: $PYTHON" >&2; exit 2; }
[[ -x "$BASE_TRAIN_SCRIPT" ]] || { echo "missing train script: $BASE_TRAIN_SCRIPT" >&2; exit 2; }
[[ -x "$EVAL_SCRIPT" ]] || { echo "missing eval script: $EVAL_SCRIPT" >&2; exit 2; }
mkdir -p "$RESULT_ROOT"

latest_step=41
init_checkpoint=$BASE_CHECKPOINT
if [[ -d "$CHECKPOINT_ROOT" ]]; then
    mapfile -t candidates < <(
        find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name 'global_step_*' -printf '%f\n' | sort -Vr
    )
    for candidate in "${candidates[@]}"; do
        if checkpoint_complete "$CHECKPOINT_ROOT/$candidate"; then
            latest_step=${candidate#global_step_}
            init_checkpoint="$CHECKPOINT_ROOT/$candidate"
            break
        fi
        echo "ignoring incomplete checkpoint: $CHECKPOINT_ROOT/$candidate" >&2
    done
fi

cd "$PROJECT_DIR"
if (( latest_step < FINAL_STEP )); then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2} \
    EXPERIMENT_NAME="$RUN_NAME" \
    INIT_CHECKPOINT="$init_checkpoint" \
    RESUME_GLOBAL_STEP="$latest_step" \
    RESUME_ACTOR_STATE=true \
    BRANCH_CREDIT_COEF=1.0 \
    BRANCH_CREDIT_VALUE_MODE=correctness \
    BRANCH_CREDIT_CORRECTNESS_THRESHOLD=0.8 \
    EXPAND_MODE=grouped_random \
    SELF_OPD_ENABLED=true \
    SELF_OPD_COEF=${SELF_OPD_COEF:-0.0003} \
    SELF_OPD_TEMPERATURE=${SELF_OPD_TEMPERATURE:-1.0} \
    SELF_OPD_MAX_EVENTS=${SELF_OPD_MAX_EVENTS:-3} \
    SELF_OPD_MAX_TEACHER_LENGTH=${SELF_OPD_MAX_TEACHER_LENGTH:-768} \
    SELF_OPD_MAX_QUERY_TOKENS=${SELF_OPD_MAX_QUERY_TOKENS:-32} \
    SELF_OPD_MAX_EVIDENCE_TOKENS=${SELF_OPD_MAX_EVIDENCE_TOKENS:-96} \
    SELF_OPD_MIN_VALUE_GAP=${SELF_OPD_MIN_VALUE_GAP:-0.25} \
    SELF_OPD_MIN_RAW_ADVANTAGE=${SELF_OPD_MIN_RAW_ADVANTAGE:-0.0} \
    SELF_OPD_MAX_ADVANTAGE_WEIGHT=${SELF_OPD_MAX_ADVANTAGE_WEIGHT:-3.0} \
    TOTAL_TRAINING_STEPS=$((FINAL_STEP + 1)) \
    SAVE_FREQ=1 \
    SAVE_AT_END=true \
    DAPO_TARGET_EFFECTIVE_PROMPTS=${DAPO_TARGET_EFFECTIVE_PROMPTS:-24} \
    DAPO_MAX_CHUNKS=${DAPO_MAX_CHUNKS:-48} \
    LOGGER="['console']" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
        "$BASE_TRAIN_SCRIPT"
fi

for ((step = 42; step <= FINAL_STEP; step++)); do
    checkpoint="$CHECKPOINT_ROOT/global_step_$step"
    validate_result="$RESULT_ROOT/step${step}/natural_n120/result.json"
    checkpoint_complete "$checkpoint" || {
        echo "missing completed checkpoint for evaluation: $checkpoint" >&2
        exit 3
    }

    mkdir -p "$RESULT_ROOT/step${step}"
    "$PYTHON" "$PROJECT_DIR/scripts/audit_gold_evidence.py" \
        "$ROLLOUT_ROOT/step_$(printf '%06d' "$step")_chunk_*_selected.jsonl" \
        --output "$RESULT_ROOT/step${step}/gold_evidence_rollout_audit.json"

    if [[ ! -s "$validate_result" ]]; then
        LABEL="tracecredit_guided_opd_v2_step${step}" \
        MODEL_PATH="$SFT_MODEL" \
        CHECKPOINT="$checkpoint" \
        CHECKPOINT_STEP="$step" \
        EVAL_SPLIT=natural \
        SAMPLE_COUNT=120 \
        RESULT_ROOT="$RESULT_ROOT/step${step}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
            "$EVAL_SCRIPT"
    fi
done

"$PYTHON" - "$STEP41_RESULT" "$RESULT_ROOT" "$FINAL_STEP" <<'PY'
import json
import pathlib
import sys

step41_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
final_step = int(sys.argv[3])
results = {"41": json.loads(step41_path.read_text())}
for step in range(42, final_step + 1):
    path = root / f"step{step}" / "natural_n120" / "result.json"
    results[str(step)] = json.loads(path.read_text())

micro_em = {step: result["micro_em"] for step, result in results.items()}
summary = {
    "protocol": "deterministic natural-120",
    "micro_em_by_step": micro_em,
    f"delta_step{final_step}_vs_step41": micro_em[str(final_step)] - micro_em["41"],
    "all_error_checks_clean": all(
        not any(result["error_checks"].values()) for result in results.values()
    ),
    "scope": "short continuation trend; Teacher directional metrics are in the training log",
}
(root / "progression.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "TRACECREDIT_GUIDED_OPD_V2_CONTINUATION_COMPLETE=$(date -Is)"
