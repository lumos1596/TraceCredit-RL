#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the two teacher-data producers, then run the reviewed build, five-card
# SFT on physical GPUs 0 through 4, followed by smoke/full evaluation on GPUs 3
# and 4. Every transition is gated by independently validated artifacts; a failed
# stage never advances the state. GPU 5 remains reserved for the retriever and is
# not exposed to SFT or passed to evaluation.

ROOT="/home/luwa/Documents/Tree-GRPO"
PYTHON="${PYTHON_BIN:-${ROOT}/.conda/envs/treegrpo/bin/python}"
SINGLE_DIR="${ROOT}/data/teacher_singlehop_qwen7b_prod_20260911"
MULTI_DIR="${ROOT}/data/teacher_multihop_dsflash_prod_20260911"
DATA_DIR="${ROOT}/data/teacher_sft_mixed_1500_20260911"
CKPT_DIR="${ROOT}/verl_checkpoints/teacher-mixed-sft-qwen2.5-3b-first-round-20260911"
STATE_DIR="${ROOT}/pipeline_state/teacher_mixed_sft_first_round_20260911"
LOG_FILE="${STATE_DIR}/pipeline.log"
POLL_INTERVAL=60
STALE_SECONDS=3600
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_teacher_mixed_sft_pipeline_20260911.sh [options]

Options:
  --dry-run               Check current state once; launch or write nothing.
  --poll-interval SEC     Seconds between generation checks (default: 60).
  --stale-seconds SEC     Fail if an incomplete summary is older than this
                          many seconds (default: 3600; 0 disables the check).
  -h, --help              Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --poll-interval) POLL_INTERVAL="${2:?missing value for --poll-interval}"; shift 2 ;;
    --stale-seconds) STALE_SECONDS="${2:?missing value for --stale-seconds}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$POLL_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "poll interval must be a positive integer" >&2; exit 2; }
[[ "$STALE_SECONDS" =~ ^[0-9]+$ ]] || { echo "stale seconds must be a non-negative integer" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }

generation_state() {
  "$PYTHON" - "$SINGLE_DIR/summary.json" "$MULTI_DIR/summary.json" "$STALE_SECONDS" <<'PY'
import json, sys, time
from pathlib import Path

specs = [(Path(sys.argv[1]), "single", 1000), (Path(sys.argv[2]), "multi", 500)]
stale_seconds = int(sys.argv[3])
ready = True
for path, label, target in specs:
    if not path.is_file():
        print(f"GENERATION {label}: waiting; summary missing: {path}")
        ready = False
        continue
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        status = summary["status"]
        accepted = int(summary["candidates"]["accepted_selected"])
    except Exception as exc:
        raise SystemExit(f"invalid {label} summary {path}: {exc}")
    age = max(0, time.time() - path.stat().st_mtime)
    print(f"GENERATION {label}: accepted={accepted}/{target} status={status} age={age:.0f}s")
    if accepted > target:
        raise SystemExit(f"{label} accepted count overshot target: {accepted}>{target}")
    if status in {"question_pool_exhausted", "failed", "error"}:
        raise SystemExit(f"{label} generation terminal failure: status={status}, accepted={accepted}")
    if accepted == target:
        if status != "target_reached":
            raise SystemExit(f"{label} has target count but non-success status={status}")
    else:
        ready = False
        if stale_seconds and age > stale_seconds:
            raise SystemExit(f"{label} generation appears stale: summary age {age:.0f}s > {stale_seconds}s")
print("READY=1" if ready else "READY=0")
PY
}

validate_accepted() {
  "$PYTHON" - "$SINGLE_DIR/accepted.jsonl" "$MULTI_DIR/accepted.jsonl" <<'PY'
import json, re, sys
from collections import Counter
from pathlib import Path

def norm(value):
    return re.sub(r"\s+", " ", str(value).strip().casefold())

def load(path):
    rows=[]
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try: row=json.loads(line)
            except Exception as exc: raise SystemExit(f"invalid JSON at {path}:{lineno}: {exc}")
            if not isinstance(row, dict): raise SystemExit(f"non-object at {path}:{lineno}")
            if row.get("eligible") is not True: raise SystemExit(f"ineligible record at {path}:{lineno}")
            rows.append(row)
    return rows

single_path, multi_path = map(Path, sys.argv[1:])
for path in (single_path, multi_path):
    if not path.is_file(): raise SystemExit(f"missing accepted data: {path}")
single, multi = load(single_path), load(multi_path)
if len(single) != 1000: raise SystemExit(f"single accepted count {len(single)} != 1000")
if len(multi) != 500: raise SystemExit(f"multi accepted count {len(multi)} != 500")
if Counter(row.get("data_source") for row in multi) != Counter({"hotpotqa":200,"2wikimultihopqa":175,"musique":125}):
    raise SystemExit(f"wrong multi source quotas: {Counter(row.get('data_source') for row in multi)}")
if len({str(r.get("question_id")) for r in single}) != len(single): raise SystemExit("duplicate single question_id")
if len({str(r.get("question_id")) for r in multi}) != len(multi): raise SystemExit("duplicate multi question_id")
keys=[norm(r.get("question_key") or r.get("question")) for r in single+multi]
if not all(keys) or len(set(keys)) != len(keys): raise SystemExit("empty or duplicate normalized question across accepted data")
candidates=[str(r.get("candidate_id")) for r in single+multi]
if len(set(candidates)) != len(candidates): raise SystemExit("duplicate candidate_id across accepted data")
print("ACCEPTED_DATA_OK total=1500 nq=1000 hotpotqa=200 2wikimultihopqa=175 musique=125")
PY
}

validate_built() {
  "$PYTHON" - "$DATA_DIR" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq

d=Path(sys.argv[1]); stats_path=d/"stats.json"; manifest_path=d/"manifest.jsonl"
for path in (d/"train.parquet", d/"validation.parquet", stats_path, manifest_path):
    if not path.is_file(): raise SystemExit(f"missing built artifact: {path}")
stats=json.loads(stats_path.read_text(encoding="utf-8"))
if stats.get("status") != "completed": raise SystemExit("builder stats are not completed")
if stats.get("counts") != {"total":1500,"train":1350,"validation":150}: raise SystemExit(f"wrong stats counts: {stats.get('counts')}")
if stats.get("quotas") != {"nq":1000,"hotpotqa":200,"2wikimultihopqa":175,"musique":125}: raise SystemExit(f"wrong stats quotas: {stats.get('quotas')}")
if pq.read_metadata(d/"train.parquet").num_rows != 1350: raise SystemExit("train parquet row count != 1350")
if pq.read_metadata(d/"validation.parquet").num_rows != 150: raise SystemExit("validation parquet row count != 150")
tokenization=stats.get("tokenization", {})
if tokenization.get("max_length") != 2560: raise SystemExit(f"wrong tokenization max_length: {tokenization.get('max_length')}")
if int(tokenization.get("max_token_length", 0)) > 2560: raise SystemExit(f"built data exceeds max_length: {tokenization.get('max_token_length')}")
splits={"train":set(),"validation":set()}; count=0
for lineno,line in enumerate(manifest_path.open(encoding="utf-8"),1):
    row=json.loads(line); split=row.get("split")
    if split not in splits: raise SystemExit(f"bad manifest split at line {lineno}")
    key=row.get("question_sha256")
    if not key: raise SystemExit(f"missing question hash at line {lineno}")
    if key in splits[split]: raise SystemExit(f"duplicate question within {split}")
    splits[split].add(key); count+=1
if count != 1500 or len(splits["train"]) != 1350 or len(splits["validation"]) != 150: raise SystemExit("manifest counts are invalid")
if splits["train"] & splits["validation"]: raise SystemExit("train/validation question leakage")
print("BUILT_DATA_OK train=1350 validation=150 overlap=0")
PY
}

validate_checkpoints() {
  for path in "$CKPT_DIR/global_step_162/config.json" "$CKPT_DIR/best/config.json"; do
    [[ -f "$path" ]] || { echo "Missing required checkpoint artifact: $path" >&2; return 1; }
  done
  echo "CHECKPOINTS_OK final=global_step_162 best=$(readlink -f "$CKPT_DIR/best")"
}

validate_evaluation() {
  local nq_count="$1"
  local multi_count="$2"
  "$PYTHON" - "$ROOT/evaluation/teacher_mixed_sft_first_round" "$nq_count" "$multi_count" <<'PY'
import json, sys
from pathlib import Path

root, nq_count, multi_count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
labels = ["original_3b", "search_r1_sft_step350", "teacher_mixed_sft_best", "teacher_mixed_sft_final"]
for label in labels:
    paths = [
        (root/label/f"nq_n{nq_count}"/"result.json", "em", nq_count),
        (root/label/f"multihop_balanced_n{multi_count}"/"result.json", "micro_em", multi_count),
    ]
    for path, metric, expected in paths:
        if not path.is_file(): raise SystemExit(f"missing evaluation result: {path}")
        result=json.loads(path.read_text(encoding="utf-8"))
        if result.get(metric) is None: raise SystemExit(f"missing {metric} in {path}")
        checks=result.get("error_checks", {})
        if any(checks.values()): raise SystemExit(f"failed error checks in {path}: {checks}")
        observed=result.get("sample_count")
        if observed is not None and int(observed) != expected:
            raise SystemExit(f"sample count mismatch in {path}: {observed} != {expected}")
print(f"EVALUATION_OK nq={nq_count} multihop={multi_count} models={len(labels)}")
PY
}

if ((DRY_RUN)); then
  echo "DRY_RUN: no files or workloads will be created"
  state="$(generation_state)"
  printf '%s\n' "$state"
  if grep -q '^READY=1$' <<<"$state"; then
    validate_accepted
    [[ ! -e "$DATA_DIR" ]] || validate_built
    [[ ! -e "$CKPT_DIR" ]] || validate_checkpoints
  else
    echo "DRY_RUN_WAITING: generation quotas are not complete"
  fi
  exit 0
fi

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/pipeline.lock"
flock -n 9 || { echo "Another pipeline instance already holds $STATE_DIR/pipeline.lock" >&2; exit 3; }
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[$(date -Is)] PIPELINE_EXIT rc=$rc"; exit $rc' EXIT

mark_done() { printf '%s\n' "$(date -Is)" >"$STATE_DIR/$1.done"; }
run_stage() {
  local name="$1"; shift
  echo "[$(date -Is)] STAGE_START $name"
  "$@"
  mark_done "$name"
  echo "[$(date -Is)] STAGE_DONE $name"
}

echo "[$(date -Is)] PIPELINE_START pid=$$"
while true; do
  state="$(generation_state)"
  printf '%s\n' "$state"
  grep -q '^READY=1$' <<<"$state" && break
  sleep "$POLL_INTERVAL"
done
validate_accepted
mark_done generation

if validate_built >/dev/null 2>&1; then
  echo "[$(date -Is)] STAGE_REUSE build (validated existing output)"
  mark_done build
else
  if [[ -e "$DATA_DIR" ]]; then
    echo "Refusing to overwrite invalid or partial data directory: $DATA_DIR" >&2
    exit 1
  fi
  run_stage build "$PYTHON" "$ROOT/scripts/data_process/build_teacher_mixed_sft_dataset.py" \
    --single-accepted "$SINGLE_DIR/accepted.jsonl" \
    --multi-accepted "$MULTI_DIR/accepted.jsonl" \
    --output-dir "$DATA_DIR" --tokenizer-path "$ROOT/models/Qwen2.5-3B-Instruct" --seed 42
  validate_built
fi

if validate_checkpoints >/dev/null 2>&1; then
  echo "[$(date -Is)] STAGE_REUSE sft (validated existing checkpoints)"
  mark_done sft
else
  if [[ -e "$CKPT_DIR" ]] && find "$CKPT_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to restart SFT over a nonempty incomplete checkpoint directory: $CKPT_DIR" >&2
    exit 1
  fi
  run_stage sft env CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$ROOT/train_teacher_mixed_sft_first_round_20260911.sh"
  validate_checkpoints
fi

if validate_evaluation 3 12 >/dev/null 2>&1; then
  echo "[$(date -Is)] STAGE_REUSE smoke (validated existing outputs)"
  mark_done smoke
else
  run_stage smoke env CUDA_VISIBLE_DEVICES=3,4 SMOKE=1 "$ROOT/evaluation/teacher_mixed_sft_first_round/run_comparison.sh"
  validate_evaluation 3 12
fi

# Smoke and full results use sample-count-qualified directories (n3/n12 versus
# n120/n480); the comparison summary is intentionally replaced by the full run.
if validate_evaluation 120 480 >/dev/null 2>&1; then
  echo "[$(date -Is)] STAGE_REUSE full_eval (validated existing outputs)"
  mark_done full_eval
else
  run_stage full_eval env CUDA_VISIBLE_DEVICES=3,4 "$ROOT/evaluation/teacher_mixed_sft_first_round/run_comparison.sh"
  validate_evaluation 120 480
fi

[[ -f "$ROOT/evaluation/teacher_mixed_sft_first_round/comparison.json" ]] || {
  echo "Full evaluation returned without comparison.json" >&2; exit 1;
}
mark_done complete
echo "[$(date -Is)] PIPELINE_COMPLETE"
