#!/usr/bin/env bash
# Outcome-privileged 7B think+search OPD from the SFT checkpoint. Gold answers
# are used only to redact privileged text and are never sent to the teacher.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
BASE_TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
PYTHON="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
TEACHER_MODEL="$PROJECT_DIR/models/Qwen2.5-7B-Instruct"
TEACHER_PORT=${TEACHER_PORT:-8126}
TEACHER_URL="http://127.0.0.1:$TEACHER_PORT"
RUN_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-opd-outcome7b-thinksearch-sftto20-20260923}
CHECKPOINT_ROOT="$PROJECT_DIR/verl_checkpoints/$RUN_NAME/actor"
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_DIR/evaluation/formal_em/opd_outcome7b_thinksearch_sftto20"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}
FINAL_STEP=${FINAL_STEP:-20}
FIRST_STEP=0
SELF_OPD_COEF=${SELF_OPD_COEF:-0.001}
TEACHER_SERVER_LOG="$PROJECT_DIR/verl_log/opd_teacher_server_${RUN_NAME}.log"
PIPELINE_LOG="$PROJECT_DIR/verl_log/${RUN_NAME}_pipeline.log"
TEACHER_PID=""

# A four-GPU monitor passes all selected physical IDs in CUDA_VISIBLE_DEVICES:
# reserve the first for the 7B teacher and the next three for FSDP training.
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#visible_gpus[@]} >= N_GPUS_PER_NODE + 1 )); then
    TEACHER_GPU=${TEACHER_GPU:-${visible_gpus[0]}}
    training_slice=("${visible_gpus[@]:1:N_GPUS_PER_NODE}")
    IFS=','; TRAINING_GPUS=${TRAINING_GPUS:-${training_slice[*]}}; unset IFS
else
    TEACHER_GPU=${TEACHER_GPU:-0}
    TRAINING_GPUS=${TRAINING_GPUS:-${CUDA_VISIBLE_DEVICES:-1,2,3}}
fi

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
[[ -e "$SFT_MODEL/config.json" ]] || { echo "missing SFT model: $SFT_MODEL" >&2; exit 2; }
[[ -d "$TEACHER_MODEL" ]] || { echo "missing teacher model: $TEACHER_MODEL" >&2; exit 2; }
curl -s -m 5 http://127.0.0.1:8000/docs -o /dev/null || {
    echo "retriever service at http://127.0.0.1:8000 is not reachable" >&2
    exit 2
}

mkdir -p "$PROJECT_DIR/verl_log" "$RESULT_ROOT"
if curl -s -m 3 "$TEACHER_URL/health" >/dev/null 2>&1; then
    echo "[orchestrate] teacher server already running at $TEACHER_URL"
else
    echo "[orchestrate] starting 7B teacher server on GPU $TEACHER_GPU (log: $TEACHER_SERVER_LOG)"
    CUDA_VISIBLE_DEVICES="$TEACHER_GPU" nohup "$PYTHON" "$PROJECT_DIR/scripts/run_opd_teacher_server.py" \
        --model "$TEACHER_MODEL" --port "$TEACHER_PORT" --vocab-size 151936 \
        > "$TEACHER_SERVER_LOG" 2>&1 &
    TEACHER_PID=$!
    for _ in $(seq 1 120); do
        if curl -s -m 3 "$TEACHER_URL/health" >/dev/null 2>&1; then break; fi
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

# A fresh run has no sharded RL checkpoint at all. On restart, resume the
# latest complete checkpoint including optimizer/scheduler state.
latest_step=$FIRST_STEP
init_checkpoint=""
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
    CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" \
    EXPERIMENT_NAME="$RUN_NAME" \
    MODEL_DIR="$SFT_MODEL" \
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
    SELF_OPD_MIN_DIRECTIONAL_LIFT=${SELF_OPD_MIN_DIRECTIONAL_LIFT:-0.0} \
    SELF_OPD_THINK_COEF=${SELF_OPD_THINK_COEF:-0.2} \
    SELF_OPD_QUERY_COEF=${SELF_OPD_QUERY_COEF:-1.0} \
    SELF_OPD_MAX_EVENTS=${SELF_OPD_MAX_EVENTS:-3} \
    SELF_OPD_MAX_TEACHER_LENGTH=${SELF_OPD_MAX_TEACHER_LENGTH:-1024} \
    SELF_OPD_MAX_QUERY_TOKENS=${SELF_OPD_MAX_QUERY_TOKENS:-32} \
    SELF_OPD_MAX_ACTION_TOKENS=${SELF_OPD_MAX_ACTION_TOKENS:-128} \
    SELF_OPD_MAX_EVIDENCE_TOKENS=${SELF_OPD_MAX_EVIDENCE_TOKENS:-96} \
    SELF_OPD_MIN_VALUE_GAP=${SELF_OPD_MIN_VALUE_GAP:-0.25} \
    SELF_OPD_MIN_RAW_ADVANTAGE=${SELF_OPD_MIN_RAW_ADVANTAGE:-0.0} \
    SELF_OPD_MAX_ADVANTAGE_WEIGHT=${SELF_OPD_MAX_ADVANTAGE_WEIGHT:-3.0} \
    SELF_OPD_TEACHER_CONTEXT=hindsight \
    SELF_OPD_EVENT_SELECTION=contrast \
    SELF_OPD_TEACHER_URL="$TEACHER_URL" \
    TOTAL_TRAINING_STEPS=$((FINAL_STEP + 1)) \
    SAVE_FREQ=2 \
    SAVE_AT_END=true \
    DAPO_TARGET_EFFECTIVE_PROMPTS=${DAPO_TARGET_EFFECTIVE_PROMPTS:-24} \
    DAPO_MAX_CHUNKS=${DAPO_MAX_CHUNKS:-48} \
    LOGGER="['console']" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
        "$BASE_TRAIN_SCRIPT"
fi

# Evaluate the unmodified SFT model as step 0, then every saved even step.
mkdir -p "$RESULT_ROOT/step0"
if [[ ! -s "$RESULT_ROOT/step0/natural_n120/result.json" ]]; then
    LABEL=opd_answer7b_dense_sft_step0 \
    MODEL_PATH="$SFT_MODEL" CHECKPOINT_STEP=0 EVAL_SPLIT=natural SAMPLE_COUNT=120 \
    RESULT_ROOT="$RESULT_ROOT/step0" CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" \
        "$EVAL_SCRIPT"
fi

for ((step = 2; step <= FINAL_STEP; step += 2)); do
    checkpoint="$CHECKPOINT_ROOT/global_step_$step"
    result="$RESULT_ROOT/step${step}/natural_n120/result.json"
    checkpoint_complete "$checkpoint" || {
        echo "missing completed checkpoint for evaluation: $checkpoint" >&2
        exit 3
    }
    mkdir -p "$RESULT_ROOT/step${step}"
    if [[ ! -s "$result" ]]; then
        LABEL="opd_answer7b_dense_sft_step${step}" MODEL_PATH="$SFT_MODEL" \
        CHECKPOINT="$checkpoint" CHECKPOINT_STEP="$step" EVAL_SPLIT=natural SAMPLE_COUNT=120 \
        RESULT_ROOT="$RESULT_ROOT/step${step}" CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" \
            "$EVAL_SCRIPT"
    fi
done

"$PYTHON" - "$RESULT_ROOT" "$FINAL_STEP" <<'PY'
import json, pathlib, sys
root, final_step = pathlib.Path(sys.argv[1]), int(sys.argv[2])
results = {str(s): json.loads((root / f"step{s}" / "natural_n120" / "result.json").read_text())
           for s in range(0, final_step + 1, 2)}
em = {s: r["micro_em"] for s, r in results.items()}
summary = {
    "protocol": "deterministic natural-120",
    "initialization": "SFT checkpoint; RL and outcome-privileged 7B think+search OPD enabled together from step 1",
    "micro_em_by_step": em,
    f"delta_step{final_step}_vs_sft": em[str(final_step)] - em["0"],
    "all_error_checks_clean": all(not any(r["error_checks"].values()) for r in results.values()),
    "opd_variant": "top positive sibling + answer-redacted outcome/evidence + think/search split loss + online directional gate",
}
(root / "progression.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "OPD_DENSE_SFT_RUN_COMPLETE=$(date -Is)" | tee -a "$PIPELINE_LOG"
