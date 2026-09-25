#!/usr/bin/env bash
set -euo pipefail

# Reproducible hard gate for generated-correction OPD.  This script deliberately
# stops after a failed 7B audit; callers must not launch training on exit 2.
GPU_ID="${GPU_ID:-3}"
PYTHON_BIN="${PYTHON_BIN:-/home/luwa/.conda/envs/dsclr/bin/python}"
MODEL="${MODEL:-models/Qwen2.5-7B-Instruct}"
OUTPUT="${OUTPUT:-verl_log/teacher_correction_retrieval_7b_v2.json}"
ROLLOUT_DIR="${ROLLOUT_DIR:-rollouts/multihop-tree-dapo-branch-credit-step20to30-retry-20260906-092827}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${PWD}:${PWD}/scripts" "${PYTHON_BIN}" \
  scripts/audit_teacher_correction_retrieval.py \
  --trees "${ROLLOUT_DIR}/step_000021_chunk_*_trees.jsonl" \
  --selected "${ROLLOUT_DIR}/step_000021_chunk_*_selected.jsonl" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --max-events 32 \
  --batch-size 2 \
  --max-context 1536 \
  --max-new-tokens 160 \
  --retrieval-url http://127.0.0.1:8000/retrieve \
  --retrieval-topk 3 \
  --device cuda:0

if ! "${PYTHON_BIN}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["verdict"]["gate_pass"] else 2)' "${OUTPUT}"; then
  echo "[preflight] FAILED: OPD training remains blocked; see ${OUTPUT}" >&2
  exit 2
fi

echo "[preflight] PASSED: 7B teacher is eligible for the subsequent 3B comparison." 
