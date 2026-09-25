#!/usr/bin/env bash
# Supervisor for the answer-conditioned 7B teacher pipeline (42 -> FINAL_STEP).
#
# Runs detached (setsid). Every minute it checks the pipeline state:
#   - done when the final step's eval result exists;
#   - if the pipeline process died before that, relaunch it (the training
#     script auto-resumes from the latest complete checkpoint, and finished
#     evals are skipped), up to MAX_RELAUNCH times.
set -u

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PIPELINE=${PIPELINE:-run_opd_answer_teacher_dense_step20_to40.sh}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_DIR/evaluation/formal_em/opd_answer7b_dense_step20to40"}
FINAL_STEP=${FINAL_STEP:-40}
MAX_RELAUNCH=4
MAX_MINUTES=2160   # 36h
LOG="$PROJECT_DIR/verl_log/answer7b_supervisor.log"

log() { echo "[supervisor] $(date '+%F %T') $*" >> "$LOG"; }

log "start: pipeline=$PIPELINE target FINAL_STEP=$FINAL_STEP result=$RESULT_ROOT/step${FINAL_STEP}/natural_n120/result.json"

attempts=0
for ((minute = 0; minute < MAX_MINUTES; minute++)); do
    if [[ -s "$RESULT_ROOT/step${FINAL_STEP}/natural_n120/result.json" ]]; then
        log "done: step${FINAL_STEP} eval result present"
        exit 0
    fi
    if ! pgrep -f "$PIPELINE" > /dev/null 2>&1; then
        if (( attempts >= MAX_RELAUNCH )); then
            log "give up: relaunch budget exhausted ($attempts)"
            exit 2
        fi
        attempts=$((attempts + 1))
        log "pipeline not running (attempt $attempts/$MAX_RELAUNCH), launching FINAL_STEP=$FINAL_STEP"
        cd "$PROJECT_DIR" || exit 3
        FINAL_STEP="$FINAL_STEP" \
            setsid nohup "./$PIPELINE" \
            >> "$PROJECT_DIR/verl_log/answer7b_supervisor_pipeline.log" 2>&1 \
            < /dev/null &
        log "launched pid=$!"
        sleep 300   # let it boot; next loop sees the new process via pgrep
    fi
    sleep 60
done

log "timeout after ${MAX_MINUTES} minutes"
exit 4
