#!/bin/bash
# bench_kwt.sh — KWT benchmark : baseline vs KWT sur le même pod GPU
# Usage : bash bench_kwt.sh [SEED] [WORKSPACE]
# Résultats écrits dans $WORKSPACE/results_kwt_bench.json
# CONFIDENTIEL — KWT IP interne BCUB3/POWWPOL, NE PAS DIFFUSER

set -euo pipefail
SEED="${1:-42}"
WS="${2:-/workspace/autoresearch}"
cd "$WS"

echo "=== KWT BENCHMARK — $(date -u '+%Y-%m-%dT%H:%M:%SZ') ===" | tee /workspace/bench_kwt.log
echo "SEED=$SEED  WS=$WS"
python3 --version && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -- Data preparation (idempotent) --
if [ ! -f ~/.cache/autoresearch/val.bin ]; then
    echo "--- [1/3] prepare.py ..." | tee -a /workspace/bench_kwt.log
    uv run prepare.py 2>&1 | tee /workspace/prepare.log
    echo "prepare.py DONE" | tee -a /workspace/bench_kwt.log
else
    echo "--- [1/3] cache found, skipping prepare.py"
fi

# -- Baseline --
echo "--- [2/3] BASELINE (KWT_ENABLED=0, SEED=$SEED) ..." | tee -a /workspace/bench_kwt.log
date -u
SEED="$SEED" KWT_ENABLED=0 uv run train.py > /workspace/run_baseline.log 2>&1 || true
echo "BASELINE DONE $(date -u)" | tee -a /workspace/bench_kwt.log
grep "^val_bpb:\|^peak_vram_mb:\|^num_steps:\|^mfu_percent:\|^total_tokens" /workspace/run_baseline.log || true

# -- KWT run --
echo "--- [3/3] KWT (KWT_ENABLED=1, SEED=$SEED) ..." | tee -a /workspace/bench_kwt.log
date -u
SEED="$SEED" KWT_ENABLED=1 KWT_LOG_PATH=/workspace/kwt_trace.jsonl uv run train.py > /workspace/run_kwt.log 2>&1 || true
echo "KWT DONE $(date -u)" | tee -a /workspace/bench_kwt.log
grep "^val_bpb:\|^peak_vram_mb:\|^num_steps:\|^mfu_percent:\|^total_tokens" /workspace/run_kwt.log || true

# -- Parse results into JSON --
python3 - <<'PYEOF'
import re, json

def parse_log(path):
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return {"error": "file not found"}
    d = {}
    for k in ["val_bpb","peak_vram_mb","num_steps","training_seconds",
              "mfu_percent","total_tokens_M","num_params_M","depth"]:
        m = re.search(rf"^{k}:\s+([0-9.]+)", txt, re.M)
        if m:
            d[k] = float(m.group(1))
    # Extract step-level loss proxy (val_bpb proxy per 50 steps)
    steps, losses = [], []
    for m in re.finditer(r"val_bpb \(step (\d+)\): ([0-9.]+)", txt):
        steps.append(int(m.group(1))); losses.append(float(m.group(2)))
    if steps:
        d["step_curve"] = {"steps": steps, "loss_proxy": losses}
    return d

base = parse_log("/workspace/run_baseline.log")
kwt  = parse_log("/workspace/run_kwt.log")
result = {
    "seed": 42,
    "baseline": base,
    "kwt": kwt,
    "delta_val_bpb": round(
        (kwt.get("val_bpb", float("nan")) - base.get("val_bpb", float("nan"))), 6
    ) if "val_bpb" in base and "val_bpb" in kwt else None,
}
with open("/workspace/results_kwt_bench.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
PYEOF

echo "=== BENCH COMPLETE ===" | tee -a /workspace/bench_kwt.log
echo "Results: /workspace/results_kwt_bench.json"
