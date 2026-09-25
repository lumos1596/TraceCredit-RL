#!/usr/bin/env bash
# Run the fixed-budget random and uncertainty-balanced probes sequentially.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"

for mode in random uncertainty_balanced; do
    "$PROJECT_DIR/run_expand_mode_credit_probe.sh" "$mode"
done

exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/compare_branch_credit_efficiency.py" \
    --random "$PROJECT_DIR/rollouts/branch-credit-expand-probe-random-step60" \
    --new "$PROJECT_DIR/rollouts/branch-credit-expand-probe-uncertainty_balanced-step60" \
    --output "$PROJECT_DIR/branch_credit_expand_mode_comparison.json"
