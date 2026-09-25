#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/luwa/Documents/Tree-GRPO
RUN_NAME=${EXPERIMENT_NAME:-tracecredit-nodeskill-sampled-nll-opd-smoke}
export EXPERIMENT_NAME="$RUN_NAME"
PIPELINE_LOG="$ROOT/verl_log/${RUN_NAME}.pipeline.log"
mkdir -p "$ROOT/verl_log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
export MODEL_DIR="$ROOT/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
export INIT_CHECKPOINT=
export RESUME_GLOBAL_STEP=0
export RESUME_ACTOR_STATE=false
export TOTAL_TRAINING_STEPS=5
export SAVE_FREQ=1
export SAVE_AT_END=true
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}
export RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}

export SELF_OPD_ENABLED=true
export SELF_OPD_LOSS=sampled_nll
export SELF_OPD_COEF=0.001
export SELF_OPD_BETA=5.0
export SELF_OPD_DELTA_MIN=0.0
export SELF_OPD_THINK_COEF=0.2
export SELF_OPD_QUERY_COEF=1.0
export SELF_OPD_MIN_VALUE_GAP=0.1
export SELF_OPD_MIN_RAW_ADVANTAGE=0.0
export SELF_OPD_MAX_EVENTS=3
export SELF_OPD_MAX_TEACHER_LENGTH=512
export SELF_OPD_MAX_ACTION_TOKENS=256
export SELF_OPD_MAX_QUERY_TOKENS=64
export SELF_OPD_MAX_EVIDENCE_TOKENS=128
export SELF_OPD_TEACHER_CONTEXT=node_skill
export SELF_OPD_EVENT_SELECTION=contrast
export SELF_OPD_ANALYZER_URL=http://127.0.0.1:8127

# Conservative 24-GiB settings: keep sharded actor params/grads on device,
# offload the substantially larger Adam state, and cap vLLM allocations.
export ACTOR_PARAM_OFFLOAD=false
export ACTOR_GRAD_OFFLOAD=false
export OPTIMIZER_OFFLOAD=true
export REF_PARAM_OFFLOAD=true
export GPU_MEMORY_UTILIZATION=0.35
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=3072
export ROLLOUT_MAX_NUM_SEQS=32
export DAPO_TARGET_EFFECTIVE_PROMPTS=12
export DAPO_MAX_CHUNKS=24
export ROLLOUT_ACCUMULATION_STEPS=8
export LOGGER="['console']"

"$ROOT/scripts/preflight_tracecredit_nodeskill_opd.sh"
"$ROOT/train_multihopqa_branch_credit_dapo_step20_to30.sh"

SMOKE_CKPT="$ROOT/verl_checkpoints/$RUN_NAME/actor/global_step_5"
SMOKE_LOG="$ROOT/verl_log/$RUN_NAME.log"
validate_checkpoint() {
    local checkpoint=$1 rank kind
    for kind in model optim extra_state; do
        for ((rank = 0; rank < N_GPUS_PER_NODE; rank++)); do
            [[ -s "$checkpoint/${kind}_world_size_${N_GPUS_PER_NODE}_rank_${rank}.pt" ]] || return 1
        done
    done
}

[[ -s "$SMOKE_LOG" ]] && ! rg -qi 'out of memory|cuda.*error|traceback|nan|ray task error' "$SMOKE_LOG"
validate_checkpoint "$SMOKE_CKPT"
"$ROOT/.conda/envs/treegrpo/bin/python" - "$SMOKE_LOG" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
def positive(key):
    values = [float(x) for x in re.findall(re.escape(key) + r"[^0-9.eE+-]*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)", text)]
    return any(value > 0 for value in values)
assert positive("self_opd/loss"), "smoke produced no positive sampled-NLL loss"
assert positive("self_opd/weighted_tokens_per_update"), "smoke produced no weighted OPD tokens"
assert positive("self_opd/gated_query_tokens_per_update") or positive("self_opd/search_tokens_per_update"), "smoke produced no gated/search tokens"
PY

echo "SMOKE_SUCCESS=$(date -Is) checkpoint=$SMOKE_CKPT"

# Continue from the validated smoke checkpoint with optimizer state restored.
FORMAL_RUN_NAME="${RUN_NAME}-formal"
export EXPERIMENT_NAME="$FORMAL_RUN_NAME"
export INIT_CHECKPOINT="$SMOKE_CKPT"
export RESUME_GLOBAL_STEP=5
export RESUME_ACTOR_STATE=true
export TOTAL_TRAINING_STEPS=21
export SAVE_FREQ=1
export SAVE_AT_END=true
"$ROOT/train_multihopqa_branch_credit_dapo_step20_to30.sh"
echo "FORMAL_SUCCESS=$(date -Is) experiment=$FORMAL_RUN_NAME"
