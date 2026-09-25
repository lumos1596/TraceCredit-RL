#!/usr/bin/env bash
# Run the calibrated selector without parameter updates, then compare it with
# the frozen random-selector baseline that supplied the aggregate priors.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
RANDOM_DIR="$PROJECT_DIR/rollouts/branch-credit-expand-probe-random-step60"
NEW_DIR="$PROJECT_DIR/rollouts/branch-credit-expand-probe-outcome_prior-step60"
OUTPUT="$PROJECT_DIR/branch_credit_outcome_prior_comparison.json"

[[ -d "$RANDOM_DIR" ]] || {
    echo "missing frozen random baseline: $RANDOM_DIR" >&2
    exit 2
}

export ROLLOUT_ACCUMULATION_STEPS=16
"$PROJECT_DIR/run_expand_mode_credit_probe.sh" outcome_prior

new_file_count=$(find "$NEW_DIR" -maxdepth 1 -name '*trees.jsonl' -type f | wc -l)
if [[ "$new_file_count" -ne 16 ]]; then
    echo "incomplete outcome_prior rollout: expected 16 tree files, got $new_file_count" >&2
    exit 1
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/compare_branch_credit_efficiency.py" \
    --random "$RANDOM_DIR" \
    --new "$NEW_DIR" \
    --new-label outcome_prior \
    --output "$OUTPUT"

echo "OUTCOME_PRIOR_PROBE_COMPLETE=$(date --iso-8601=seconds)"
echo "RESULT=$OUTPUT"
