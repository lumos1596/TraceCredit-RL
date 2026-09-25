#!/usr/bin/env bash
# Continuation of the dense answer-OPD run: branch from the step-40 checkpoint
# of multihop-tree-dapo-opd-answer7b-dense-step20to40-20260920 (natural-120
# micro_em 0.358 vs 0.217 RL-only baseline) and run 20 more steps to step 60
# with the identical configuration (event_selection=all, 7B teacher on GPU 0,
# coef 0.001), evaluating every second checkpoint on the deterministic
# natural-120 protocol to check whether the 2wiki-heavy gain continues.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
BASE_TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
PYTHON="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
TEACHER_MODEL="$PROJECT_DIR/models/Qwen2.5-7B-Instruct"
TEACHER_PORT=${TEACHER_PORT:-8126}
TEACHER_URL="http://127.0.0.1:$TEACHER_PORT"
BASE_CHECKPOINT=${BASE_CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/multihop-tree-dapo-opd-answer7b-dense-step20to40-20260920/actor/global_step_40"}
RUN_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-opd-answer7b-dense-step40to60-20260921}
CHECKPOINT_ROOT="$PROJECT_DIR/verl_checkpoints/$RUN_NAME/actor"
ROLLOUT_ROOT="$PROJECT_DIR/rollouts/$RUN_NAME"
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_DIR/evaluation/formal_em/opd_answer7b_dense_step40to60"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}
FINAL_STEP=${FINAL_STEP:-60}
FIRST_STEP=40
SELF_OPD_COEF=${SELF_OPD_COEF:-0.001}
TEACHER_SERVER_LOG="$PROJECT_DIR/verl_log/opd_teacher_server_${RUN_NAME}.log"
TEACHER_PID=""

checkpoint_complete() {
    local checkpoint=$1 rank kind
    for kind in model optim extra_state; do
        for ((rank = 0; rank < N_GPUS_PER_NODE; rank++)); do
            [[ -s "$checkpoint/${kind}_world_size_${N_GPUS_PER_NODE}_rank_${rank}.pt" ]] || return 1
        done
    done
}

cleanup() {
    if [[ -n "$TEACHER_PID" ]] && kill -0 "$TEACHER_PID" 2>/dev/null; then
        echo "[orchestrate] stopping teacher server pid=$TEACHER_PID"
        kill "$TEACHER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

[[ -x "$PYTHON" ]] || { echo "missing python: $PYTHON" >&2; exit 2; }
[[ -x "$BASE_TRAIN_SCRIPT" ]] || { echo "missing train script: $BASE_TRAIN_SCRIPT" >&2; exit 2; }
[[ -x "$EVAL_SCRIPT" ]] || { echo "missing eval script: $EVAL_SCRIPT" >&2; exit 2; }
[[ -d "$TEACHER_MODEL" ]] || { echo "missing teacher model: $TEACHER_MODEL" >&2; exit 2; }
checkpoint_complete "$BASE_CHECKPOINT" || {
    echo "incomplete base checkpoint: $BASE_CHECKPOINT" >&2
    exit 2
}
curl -s -m 5 http://127.0.0.1:8000/docs -o /dev/null || {
    echo "retriever service at http://127.0.0.1:8000 is not reachable" >&2
    exit 2
}

# 1. Dedicated 7B teacher server on GPU 0 (training keeps GPUs 1,2,3).
mkdir -p "$PROJECT_DIR/verl_log" "$RESULT_ROOT"
if curl -s -m 3 "$TEACHER_URL/health" >/dev/null 2>&1; then
    echo "[orchestrate] teacher server already running at $TEACHER_URL"
else
    echo "[orchestrate] starting 7B teacher server on GPU 0 (log: $TEACHER_SERVER_LOG)"
    CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" "$PROJECT_DIR/scripts/run_opd_teacher_server.py" \
        --model "$TEACHER_MODEL" --port "$TEACHER_PORT" --vocab-size 151936 \
        > "$TEACHER_SERVER_LOG" 2>&1 &
    TEACHER_PID=$!
    for _ in $(seq 1 120); do
        if curl -s -m 3 "$TEACHER_URL/health" >/dev/null 2>&1; then
            break
        fi
        kill -0 "$TEACHER_PID" 2>/dev/null || {
            echo "teacher server died during startup:" >&2
            tail -20 "$TEACHER_SERVER_LOG" >&2
            exit 3
        }
        sleep 5
    done
    curl -s -m 3 "$TEACHER_URL/health" >/dev/null 2>&1 || {
        echo "teacher server failed to become healthy in 600s" >&2
        tail -20 "$TEACHER_SERVER_LOG" >&2
        exit 3
    }
fi
echo "[orchestrate] teacher healthy at $TEACHER_URL"

latest_step=$FIRST_STEP
init_checkpoint=$BASE_CHECKPOINT
resume_state=false
if [[ -d "$CHECKPOINT_ROOT" ]]; then
    mapfile -t candidates < <(
        find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name 'global_step_*' -printf '%f\n' | sort -Vr
    )
    for candidate in "${candidates[@]}"; do
        if checkpoint_complete "$CHECKPOINT_ROOT/$candidate"; then
            latest_step=${candidate#global_step_}
            init_checkpoint="$CHECKPOINT_ROOT/$candidate"
            resume_state=true
            break
        fi
        echo "ignoring incomplete checkpoint: $candidate" >&2
    done
fi

cd "$PROJECT_DIR"
if (( latest_step < FINAL_STEP )); then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3} \
    EXPERIMENT_NAME="$RUN_NAME" \
    INIT_CHECKPOINT="$init_checkpoint" \
    RESUME_GLOBAL_STEP="$latest_step" \
    RESUME_ACTOR_STATE="$resume_state" \
    BRANCH_CREDIT_COEF=1.0 \
    BRANCH_CREDIT_VALUE_MODE=correctness \
    BRANCH_CREDIT_CORRECTNESS_THRESHOLD=0.8 \
    EXPAND_MODE=grouped_random \
    SELF_OPD_ENABLED=true \
    SELF_OPD_COEF="$SELF_OPD_COEF" \
    SELF_OPD_TEMPERATURE=${SELF_OPD_TEMPERATURE:-1.0} \
    SELF_OPD_MAX_EVENTS=${SELF_OPD_MAX_EVENTS:-8} \
    SELF_OPD_MAX_TEACHER_LENGTH=${SELF_OPD_MAX_TEACHER_LENGTH:-768} \
    SELF_OPD_MAX_QUERY_TOKENS=${SELF_OPD_MAX_QUERY_TOKENS:-32} \
    SELF_OPD_MAX_EVIDENCE_TOKENS=${SELF_OPD_MAX_EVIDENCE_TOKENS:-96} \
    SELF_OPD_MIN_VALUE_GAP=${SELF_OPD_MIN_VALUE_GAP:-0.25} \
    SELF_OPD_MIN_RAW_ADVANTAGE=${SELF_OPD_MIN_RAW_ADVANTAGE:-0.0} \
    SELF_OPD_MAX_ADVANTAGE_WEIGHT=${SELF_OPD_MAX_ADVANTAGE_WEIGHT:-3.0} \
    SELF_OPD_TEACHER_CONTEXT=answer \
    SELF_OPD_EVENT_SELECTION=all \
    SELF_OPD_TEACHER_URL="$TEACHER_URL" \
    TOTAL_TRAINING_STEPS=$((FINAL_STEP + 1)) \
    SAVE_FREQ=1 \
    SAVE_AT_END=true \
    DAPO_TARGET_EFFECTIVE_PROMPTS=${DAPO_TARGET_EFFECTIVE_PROMPTS:-24} \
    DAPO_MAX_CHUNKS=${DAPO_MAX_CHUNKS:-48} \
    LOGGER="['console']" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
        "$BASE_TRAIN_SCRIPT"
fi

# 2. Baseline eval of the branch point, then every second new checkpoint.
mkdir -p "$RESULT_ROOT/step${FIRST_STEP}"
if [[ ! -s "$RESULT_ROOT/step${FIRST_STEP}/natural_n120/result.json" ]]; then
    LABEL="opd_answer7b_dense_step${FIRST_STEP}" \
    MODEL_PATH="$SFT_MODEL" \
    CHECKPOINT="$BASE_CHECKPOINT" \
    CHECKPOINT_STEP="$FIRST_STEP" \
    EVAL_SPLIT=natural \
    SAMPLE_COUNT=120 \
    RESULT_ROOT="$RESULT_ROOT/step${FIRST_STEP}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}" \
        "$EVAL_SCRIPT"
fi

for ((step = FIRST_STEP + 2; step <= FINAL_STEP; step += 2)); do
    checkpoint="$CHECKPOINT_ROOT/global_step_$step"
    validate_result="$RESULT_ROOT/step${step}/natural_n120/result.json"
    checkpoint_complete "$checkpoint" || {
        echo "missing completed checkpoint for evaluation: $checkpoint" >&2
        exit 3
    }

    mkdir -p "$RESULT_ROOT/step${step}"
    if [[ ! -s "$validate_result" ]]; then
        LABEL="opd_answer7b_dense_step${step}" \
        MODEL_PATH="$SFT_MODEL" \
        CHECKPOINT="$checkpoint" \
        CHECKPOINT_STEP="$step" \
        EVAL_SPLIT=natural \
        SAMPLE_COUNT=120 \
        RESULT_ROOT="$RESULT_ROOT/step${step}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}" \
            "$EVAL_SCRIPT"
    fi
done

"$PYTHON" - "$RESULT_ROOT" "$FIRST_STEP" "$FINAL_STEP" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
first_step = int(sys.argv[2])
final_step = int(sys.argv[3])
results = {}
for step in range(first_step, final_step + 1, 2):
    path = root / f"step{step}" / "natural_n120" / "result.json"
    results[str(step)] = json.loads(path.read_text())

micro_em = {step: result["micro_em"] for step, result in results.items()}
summary = {
    "protocol": "deterministic natural-120",
    "micro_em_by_step": micro_em,
    f"delta_step{final_step}_vs_step{first_step}": micro_em[str(final_step)] - micro_em[str(first_step)],
    "all_error_checks_clean": all(
        not any(result["error_checks"].values()) for result in results.values()
    ),
    "scope": "dense answer-OPD continuation from the step20to40 branch point "
             "(step-40 natural-120 micro_em 0.358); tests whether the 2wiki-heavy "
             "gain persists for 20 more steps",
}
(root / "progression.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "OPD_DENSE_CONTINUATION_COMPLETE=$(date -Is)"
