# Autoresearch × sWELU — BCUB3 fork

> Branch `autoresearch/qdrant-corpus-2026-05-13` extends [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with **sWELU** (smooth Weibull Exponential Linear Unit, **patent INPI FR2513029** by Paul OBARA, BCUB3) replacing the default squared-ReLU MLP activation.

## TL;DR — Phase 3 breakthrough 2026-05-14

| Setup | Activation | val_bpb FINAL | Notes |
|---|---|---|---|
| Karpathy original | ReLU² (`F.relu(x).square()`) | 0.7100 | baseline |
| sWELU **with gradient hygiene** | `sWELU(k, λ, β)` learned via AdamW | **0.7020** ⭐ | beats baseline by 1.1% |

Same model (50M params, 8 transformer blocks, vocab 8192, seq 2048), same corpus (15M tokens domain corpus), same 5-min training wall-time, same Muon+AdamW optimizer. Only difference: activation function and gradient flow on its parameters.

## What is sWELU?

A smooth, parameterized activation function. The three parameters `(k, λ, β)` are **learned during training via backpropagation**, exactly like `nn.Linear.weight`:

```
sWELU(x; k, λ, β) = x · σ(β·x) − λ · (1 − exp(−(|x|/λ)^k)) · (1 − σ(β·x))
```

- Positive branch `x · σ(β·x)` : Swish-like, ≈ identity for x→+∞
- Negative branch `−λ · (1 − exp(−(|x|/λ)^k))` : asymptotically bounded by `−λ`, controlled-shape Weibull cdf
- `β` controls transition sharpness, `k` controls negative-branch curvature, `λ` controls saturation depth

All three parameters are `nn.Parameter` — the network discovers its optimal activation shape per block during training.

## Critical implementation detail — *gradient hygiene*

A subtle pitfall when learning activation parameters: **never use `torch.clamp`** on learnable parameters in the forward pass — it kills the gradient outside the clamp range.

**Bad** (gradient ZERO when `k < 0.1`):
```python
weibull_neg = lam * (1.0 - torch.exp(
    -((abs_x / lam).clamp(min=1e-8)).pow(self.k.clamp(min=0.1))
))
```

**Good** (smooth lower bound, gradient continuous everywhere):
```python
k_eff = F.softplus(self.k) + 0.05
weibull_neg = lam * (1.0 - torch.exp(
    -((abs_x / lam).clamp(min=1e-8)).pow(k_eff)
))
```

This single change brought val_bpb from **1.118** (with clamp killer) to **0.702** (with softplus) — a 36% improvement, beating the squared-ReLU baseline.

**Lesson**: when introducing learnable parameters into activation functions, instrument the gradient flow with explicit logging :

```python
print(f"swelu_param (step {step}, block {b}): "
      f"k={F.softplus(k).item()+0.05:.3f} (|grad|={k.grad.abs().item():.4f}) | "
      f"lam={torch.exp(log_lambda).item():.3f} (|grad|={log_lambda.grad.abs().item():.4f}) | "
      f"beta={beta.item():.3f} (|grad|={beta.grad.abs().item():.4f})")
```

If `|grad|` stays near zero for many steps, the chain rule is broken somewhere upstream.

## Patent FR2513029

sWELU is filed at INPI (France) under patent number FR2513029 by Paul OBARA, BCUB3. The patent claim covers the activation form and the property that its parameters are learnable end-to-end via backpropagation.

This public fork is provided as scientific reproducibility material. Commercial use of sWELU in production systems requires patent licensing — contact info@bcub3.com.

## How to reproduce

```bash
git clone https://github.com/Powwpol/autoresearch
cd autoresearch
git checkout autoresearch/qdrant-corpus-2026-05-13

# Prepare tokenizer + corpus (your JSONL files in /workspace/corpus/)
uv run prepare.py

# Train
uv run train.py
```

Hardware tested: NVIDIA A40 SECURE (48 GiB), NVIDIA L40S (48 GiB). 5 min training wall-time @ batch 64, ~$0.13 per run on RunPod.

## Architecture (8-block transformer)

```
Input tokens
  ↓
Embedding (vocab 8192 → 512-d)
  ↓
8× Block:
  ├── LayerNorm → CausalSelfAttention (Flash-Attn-3) → +residual
  └── LayerNorm → MLP[Linear(512→2048) → sWELU(k,λ,β) → Linear(2048→512)] → +residual
  ↓
LM head → softmax over 8192 vocab
```

## Validation history

| Date | Setup | sWELU result |
|---|---|---|
| 2026-05-08 | MoE drift gate (1 hidden layer × 6 units, random search 600 candidates) | 77.5% directional accuracy vs 56% baselines (+21.4 pp) |
| 2026-05-14 | autoresearch nano-GPT (8 blocks × 512-d, AdamW backprop) | val_bpb 0.702 vs 0.710 ReLU² baseline |

## Credits

- **karpathy/autoresearch** — base agent framework + nanochat-derived training script
- **BCUB3 R&D** (Paul OBARA + Nika OS) — sWELU patch, gradient hygiene fix, Phase 1-3 experiments
- **INPI FR2513029** — patent filing covering sWELU

## License

Code: same as upstream karpathy/autoresearch (MIT).
Patent: FR2513029 commercial license required for production use.
