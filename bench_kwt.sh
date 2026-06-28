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

# ── Dependency bootstrap ────────────────────────────────────────────────────
# Strategy: use pre-installed torch if GPU-capable; else install from cu128.
# This makes the script work on both runpod/base (no torch) and runpod/pytorch:* (has torch).
echo "--- [0/3] Dependency setup ..." | tee -a /workspace/bench_kwt.log

_torch_ok() {
    python3 -c "import torch; assert torch.cuda.is_available(), 'cuda not available'" 2>/dev/null
}

if _torch_ok; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    echo "torch $TORCH_VER with CUDA already available — skip torch install"
    # Install all non-torch deps from pyproject.toml (torch pre-installed; skip it)
    pip install --quiet tiktoken sentencepiece numpy pandas pyarrow rustbpe requests kernels wandb 2>&1 | tail -5
else
    echo "No GPU torch — installing via uv (pyproject.toml: torch==2.9.1 cu128)..."
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    which uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    uv sync 2>&1 | tail -10
fi

echo "dep setup DONE" | tee -a /workspace/bench_kwt.log

# Determine run prefix: uv run if uv available + venv, else plain python3
if command -v uv >/dev/null 2>&1 && [ -d .venv ]; then
    RUN_PY="uv run python3"
else
    RUN_PY="python3"
fi
echo "RUN_PY=$RUN_PY"

# -- Data preparation (idempotent) --
if [ ! -f ~/.cache/autoresearch/val.bin ]; then
    echo "--- [1/3] prepare.py ..." | tee -a /workspace/bench_kwt.log
    $RUN_PY prepare.py 2>&1 | tee /workspace/prepare.log
    echo "prepare.py DONE" | tee -a /workspace/bench_kwt.log
else
    echo "--- [1/3] cache found, skipping prepare.py"
fi

# -- Baseline --
echo "--- [2/3] BASELINE (KWT_ENABLED=0, SEED=$SEED) ..." | tee -a /workspace/bench_kwt.log
date -u
SEED="$SEED" KWT_ENABLED=0 $RUN_PY train.py > /workspace/run_baseline.log 2>&1 || true
echo "BASELINE DONE $(date -u)" | tee -a /workspace/bench_kwt.log
grep "^val_bpb:\|^peak_vram_mb:\|^num_steps:\|^mfu_percent:\|^total_tokens" /workspace/run_baseline.log || true

# -- KWT run --
echo "--- [3/3] KWT (KWT_ENABLED=1, SEED=$SEED) ..." | tee -a /workspace/bench_kwt.log
date -u
SEED="$SEED" KWT_ENABLED=1 KWT_LOG_PATH=/workspace/kwt_trace.jsonl $RUN_PY train.py > /workspace/run_kwt.log 2>&1 || true
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
