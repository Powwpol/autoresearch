#!/usr/bin/env bash
# bench_rl.sh — 4 conditions × 3 seeds = 12 RL runs (GRPO/DPO × ±sWELU+KWT)
# Usage: bash bench_rl.sh [OUT_DIR]
# CONFIDENTIEL — KWT/sWELU = IP interne BCUB3/POWWPOL. NE PAS DIFFUSER.
set -euo pipefail
OUT_DIR="${1:-/workspace/results_rl}"
mkdir -p "$OUT_DIR"

WARMUP_SEC="${WARMUP_SEC:-60}"
RL_SEC="${RL_SEC:-120}"
DEPTH="${DEPTH:-12}"
RL_BATCH="${RL_BATCH:-16}"
G="${G:-4}"
RL_SEQ_LEN="${RL_SEQ_LEN:-512}"

echo "=== bench_rl.sh  OUT=$OUT_DIR  warmup=${WARMUP_SEC}s  rl=${RL_SEC}s ==="

# Conditions: ALGO  SWELU  KWT
CONDITIONS=(
  "grpo 0 0"
  "grpo 1 1"
  "dpo  0 0"
  "dpo  1 1"
)

SEEDS=(42 43 44)

for COND in "${CONDITIONS[@]}"; do
  read -r ALGO SWELU KWT <<< "$COND"
  for SEED in "${SEEDS[@]}"; do
    ALGO_TRIM="${ALGO// /}"
    LABEL="${ALGO_TRIM}_s${SWELU}k${KWT}_seed${SEED}"
    OUT_FILE="${OUT_DIR}/results_${LABEL}.json"
    TRACE_FILE="${OUT_DIR}/trace_${LABEL}.jsonl"

    if [[ -f "$OUT_FILE" ]]; then
      echo "  SKIP (exists) $LABEL"
      continue
    fi

    echo "  RUN $LABEL ..."
    env ALGO="$ALGO_TRIM" \
        SWELU_ENABLED="$SWELU" \
        KWT_ENABLED="$KWT" \
        SEED="$SEED" \
        WARMUP_SEC="$WARMUP_SEC" \
        RL_SEC="$RL_SEC" \
        DEPTH="$DEPTH" \
        RL_BATCH="$RL_BATCH" \
        G="$G" \
        RL_SEQ_LEN="$RL_SEQ_LEN" \
        LABEL="$LABEL" \
        OUT_FILE="$OUT_FILE" \
        TRACE_FILE="$TRACE_FILE" \
      python3 /workspace/autoresearch/train_rl.py
    echo "  DONE $LABEL → $OUT_FILE"
  done
done

echo "=== All runs complete.  Results in $OUT_DIR ==="
ls -lh "$OUT_DIR"/*.json
