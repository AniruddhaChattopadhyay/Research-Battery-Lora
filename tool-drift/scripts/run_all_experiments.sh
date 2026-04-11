#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MASTER_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

passed=0
failed=0
declare -a FAILED_CONFIGS=()

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

run_experiment() {
    local script="$1"
    local config="$2"
    local label="$3"
    local logfile="$LOG_DIR/${label}_${TIMESTAMP}.log"

    log "START  $label"
    if python "$script" --config "$config" > "$logfile" 2>&1; then
        log "PASS   $label"
        ((passed++)) || true
    else
        log "FAIL   $label (see $logfile)"
        ((failed++)) || true
        FAILED_CONFIGS+=("$label")
    fi
}

log "=========================================="
log "Tool-Drift Retry: final experiment"
log "=========================================="

run_experiment scripts/run_pilot_bfcl.py configs/bfcl_stage3_200_llama4scout.yaml "bfcl_200_llama4scout"

# ── Summary ──
log "=========================================="
log "DONE  passed=$passed  failed=$failed"
if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    log "Failed: ${FAILED_CONFIGS[*]}"
fi
log "Master log: $MASTER_LOG"
log "=========================================="
