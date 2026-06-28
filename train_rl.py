#!/usr/bin/env python3
"""
train_rl.py — GRPO / DPO benchmark nanochat-GPT + KWT-NS + sWELU.

Conditions (env vars):
  ALGO=grpo|dpo          SWELU_ENABLED=0|1   KWT_ENABLED=0|1
  SEED=int               WARMUP_SEC=int       RL_SEC=int
  DEPTH=int              RL_BATCH=int         G=int
  BETA_DPO=float         RL_SEQ_LEN=int
  OUT_FILE=path          TRACE_FILE=path

CONFIDENTIEL — KWT/sWELU = IP interne BCUB3/POWWPOL (FR2513029). NE PAS DIFFUSER.
"""
import os, sys, time, json, math, copy
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Env config ─────────────────────────────────────────────────────────────────
SEED          = int(os.environ.get("SEED",          "42"))
ALGO          = os.environ.get("ALGO",              "grpo").lower()
SWELU_ENABLED = os.environ.get("SWELU_ENABLED",     "1") == "1"
KWT_ENABLED   = os.environ.get("KWT_ENABLED",       "1") == "1"
WARMUP_SEC    = int(os.environ.get("WARMUP_SEC",    "60"))
RL_SEC        = int(os.environ.get("RL_SEC",        "120"))
OUT_FILE      = os.environ.get("OUT_FILE",  "/workspace/results_rl.json")
TRACE_FILE    = os.environ.get("TRACE_FILE","/workspace/rl_trace.jsonl")
DEPTH         = int(os.environ.get("DEPTH",         "12"))
RL_BATCH      = int(os.environ.get("RL_BATCH",      "16"))
WARMUP_BATCH  = int(os.environ.get("WARMUP_BATCH",  "32"))
G             = int(os.environ.get("G",              "4"))
BETA_DPO      = float(os.environ.get("BETA_DPO",    "0.1"))
RL_SEQ_LEN    = int(os.environ.get("RL_SEQ_LEN",    "512"))
MATRIX_LR     = float(os.environ.get("MATRIX_LR",   "3e-4"))
SCALAR_LR     = float(os.environ.get("SCALAR_LR",   "3e-5"))
RL_LR_SCALE   = float(os.environ.get("RL_LR_SCALE", "0.1"))
LABEL         = os.environ.get("LABEL",
    f"{ALGO}_s{int(SWELU_ENABLED)}k{int(KWT_ENABLED)}_seed{SEED}")

assert ALGO in ("grpo", "dpo"), f"ALGO={ALGO!r} must be grpo|dpo"

# ── Seed + device ──────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
import random; random.seed(SEED)
np.random.seed(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

print(f"train_rl  label={LABEL}  algo={ALGO}  swelu={SWELU_ENABLED}  kwt={KWT_ENABLED}  seed={SEED}",
      flush=True)

# ── Imports from prepare.py ────────────────────────────────────────────────────
sys.path.insert(0, "/workspace/autoresearch")
from prepare import Tokenizer, make_dataloader

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"vocab_size={vocab_size}", flush=True)

# ── Architecture (exact copy from train.py, SDPA-only, SWELU_ENABLED switch) ──

ASPECT_RATIO   = 64
HEAD_DIM       = 64
WINDOW_PATTERN = "SSSL"


@dataclass
class GPTConfig:
    sequence_len:   int   = 2048
    vocab_size:     int   = 8192
    n_layer:        int   = 12
    n_head:         int   = 6
    n_kv_head:      int   = 6
    n_embd:         int   = 768
    window_pattern: str   = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd    = config.n_embd
        self.head_dim  = config.n_embd // config.n_head
        self.c_q    = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.c_k    = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v    = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,                   bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (nn.Linear(self.ve_gate_channels, config.n_kv_head, bias=False)
                        if has_ve(layer_idx, config.n_layer) else None)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True).transpose(1, 2)
        return self.c_proj(y.contiguous().view(B, T, -1))


class sWELU(nn.Module):
    """FR2513029 — CONFIDENTIEL IP Paul OBARA BCUB3."""
    def __init__(self, k_init=1.5, lambda_init=1.0, beta_init=1.0):
        super().__init__()
        self.k          = nn.Parameter(torch.tensor(float(k_init)))
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(float(lambda_init))))
        self.beta       = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        lam    = torch.exp(self.log_lambda)
        gate   = torch.sigmoid(self.beta * x)
        k_eff  = F.softplus(self.k) + 0.05
        weibull = lam * (1.0 - torch.exp(-((x.abs() / lam).clamp(min=1e-8)).pow(k_eff)))
        return x * gate - weibull * (1.0 - gate)


class SqReLU(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        _k   = float(os.environ.get("SWELU_K",      "1.5"))
        _lam = float(os.environ.get("SWELU_LAMBDA", "1.0"))
        _b   = float(os.environ.get("SWELU_BETA",   "1.0"))
        self.activation = sWELU(_k, _lam, _b) if SWELU_ENABLED else SqReLU()

    def forward(self, x):
        return self.c_proj(self.activation(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config      = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer  = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h":   nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head       = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim
        self.value_embeds  = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rot(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rot(self, seq_len, head_dim, base=10000, device=None):
        dev = device or self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=dev)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        tbl = {"L": (long_w, 0), "S": (short_w, 0)}
        ws = [tbl[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        ws[-1] = (long_w, 0)
        return ws

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.transformer.wte.weight, 0, 1.0)
        nn.init.normal_(self.lm_head.weight, 0, 0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for blk in self.transformer.h:
            for W in [blk.attn.c_q.weight, blk.attn.c_k.weight, blk.attn.c_v.weight,
                      blk.mlp.c_fc.weight]:
                nn.init.uniform_(W, -s, s)
            nn.init.zeros_(blk.attn.c_proj.weight)
            nn.init.zeros_(blk.mlp.c_proj.weight)
            if blk.attn.ve_gate is not None:
                nn.init.zeros_(blk.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
            ve.to(dtype=torch.bfloat16)
        self.transformer.wte.to(dtype=torch.bfloat16)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rot(self.rotary_seq_len, head_dim)

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x  = norm(self.transformer.wte(idx))
        x0 = x
        for i, blk in enumerate(self.transformer.h):
            x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x  = blk(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
        return logits


# ── KWTNonStationary — CONFIDENTIEL (IP BCUB3/POWWPOL) ───────────────────────
class KWTNonStationary:
    def __init__(self, window=30, max_boost=0.3, min_hist=12, deadzone=0.6, smooth=5):
        self.hist      = deque(maxlen=max(window, min_hist))
        self.window    = window
        self.max_boost = max_boost
        self.min_hist  = min_hist
        self.deadzone  = deadzone
        self.smooth    = max(1, smooth)
        self.last_scale = 1.0
        self.fit_fail   = 0
        self.n_boosted  = 0

    @staticmethod
    def _weibull_cdf(x, k, lam):
        return 1.0 - np.exp(-(np.clip(x, 1e-6, None) / max(lam, 1e-3)) ** max(k, 0.3))

    def scale(self):
        if len(self.hist) < self.min_hist:
            self.last_scale = 1.0
            return 1.0
        w      = np.asarray(self.hist, dtype=float)
        recent = float(np.mean(w[-self.smooth:]))
        try:
            from scipy.optimize import curve_fit
            p, _ = curve_fit(self._weibull_cdf, np.sort(w),
                             np.linspace(0.02, 0.98, len(w)),
                             p0=[1.5, w.mean() + 1e-3], maxfev=2000,
                             bounds=([0.3, 1e-3], [6.0, 10 * w.mean() + 1.0]))
            pw = float(self._weibull_cdf(recent, *p))
        except Exception:
            self.fit_fail += 1
            pw = float(recent > np.median(w))
        if pw <= self.deadzone:
            self.last_scale = 1.0
            return 1.0
        b = w.mean() / (w.mean() + w.std() + 1e-6)
        q = 1.0 - pw
        f = float(np.clip((pw * b - q) / (b + 1e-6), 0.0, self.max_boost))
        if f > 0:
            self.n_boosted += 1
        self.last_scale = 1.0 + f
        return self.last_scale

    def push(self, v):
        if v == v and abs(v) < 1e6:
            self.hist.append(float(v))

    def push_and_scale(self, v):
        self.push(v)
        return self.scale()


# ── Model + optimizer ──────────────────────────────────────────────────────────
def build_config():
    base_dim   = DEPTH * ASPECT_RATIO
    model_dim  = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads  = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=RL_SEQ_LEN, vocab_size=vocab_size,
        n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_config()
print(f"config: {asdict(config)}", flush=True)

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
num_params = sum(p.numel() for p in model.parameters())
print(f"params={num_params:,}  swelu={SWELU_ENABLED}  kwt={KWT_ENABLED}", flush=True)

matrix_params = [p for p in model.parameters() if p.ndim >= 2]
scalar_params  = [p for p in model.parameters() if p.ndim < 2]
optimizer = torch.optim.AdamW(
    [{"params": matrix_params, "lr": MATRIX_LR, "initial_lr": MATRIX_LR},
     {"params": scalar_params,  "lr": SCALAR_LR,  "initial_lr": SCALAR_LR}],
    betas=(0.9, 0.95), weight_decay=0.1, eps=1e-10,
)

kwt = KWTNonStationary()


# ── LR schedule helpers ────────────────────────────────────────────────────────
WARMUP_RATIO   = 0.10
FINAL_LR_FRAC  = 0.10

def get_lrm(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    if progress < 1.0:
        p = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p))
    return FINAL_LR_FRAC


# ── Phase 1: SL warmup ─────────────────────────────────────────────────────────
print(f"\n=== Phase 1: SL warmup ({WARMUP_SEC}s) ===", flush=True)
model.train()
warmup_loader = make_dataloader(tokenizer, WARMUP_BATCH, RL_SEQ_LEN, "train")
x_w, y_w, _ = next(warmup_loader)

warmup_time  = 0.0
warmup_steps = 0
warmup_loss  = float("nan")

while True:
    t0 = time.time()
    with autocast_ctx:
        loss = model(x_w, y_w)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    progress = min(warmup_time / max(WARMUP_SEC, 1), 1.0)
    lrm = get_lrm(progress)
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * lrm

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    x_w, y_w, _ = next(warmup_loader)

    torch.cuda.synchronize()
    dt = time.time() - t0
    if warmup_steps > 5:
        warmup_time += dt

    warmup_loss = loss.item()
    if warmup_steps % 50 == 0:
        print(f"  warmup {warmup_steps:04d} | loss={warmup_loss:.4f} | t={warmup_time:.1f}s",
              flush=True)
    warmup_steps += 1
    if warmup_time >= WARMUP_SEC:
        break

print(f"Warmup done: {warmup_steps} steps  final_loss={warmup_loss:.4f}", flush=True)


# ── Phase 2: RL ────────────────────────────────────────────────────────────────
ref_model = None
if ALGO == "dpo":
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print("DPO: ref_model frozen", flush=True)

rl_loader = make_dataloader(tokenizer, RL_BATCH, RL_SEQ_LEN, "train")

# Lower LR for RL phase
for g in optimizer.param_groups:
    g["lr"] = g["initial_lr"] * RL_LR_SCALE
    g["initial_lr"] = g["lr"]

print(f"\n=== Phase 2: RL ({RL_SEC}s)  algo={ALGO}  G={G}  rl_lr={optimizer.param_groups[0]['lr']:.2e} ===",
      flush=True)

model.train()
rl_time        = 0.0
rl_step        = 0
kwt_activations = 0
reward_curve   = {"steps": [], "rewards": []}
reward_mean    = 0.0

trace_fh = open(TRACE_FILE, "w")

while True:
    x_rl, y_rl, _ = next(rl_loader)
    prompts = x_rl             # [B, T]
    targets = y_rl[:, -1]     # [B] — next token after last position (cloze)
    B = prompts.size(0)

    t0 = time.time()

    if ALGO == "grpo":
        # Sample G completions from current policy (one forward, G stochastic draws)
        with torch.no_grad(), autocast_ctx:
            logits_sg = model(prompts)[:, -1, :].float()   # [B, V]
        probs    = F.softmax(logits_sg, dim=-1)             # [B, V]
        sampled  = [torch.multinomial(probs, 1).squeeze(1) for _ in range(G)]  # G×[B]
        rewards  = [(s == targets).float() for s in sampled]
        rewards_t = torch.stack(rewards)                   # [G, B]
        adv = (rewards_t - rewards_t.mean(0, keepdim=True)) / (rewards_t.std(0, keepdim=True) + 1e-8)

        # Policy gradient loss (with grad)
        with autocast_ctx:
            logits_pg = model(prompts)[:, -1, :].float()   # [B, V]
        log_p = F.log_softmax(logits_pg, dim=-1)
        pg_losses = []
        for gi in range(G):
            lp_i = log_p.gather(1, sampled[gi].unsqueeze(1)).squeeze(1)  # [B]
            pg_losses.append(-(lp_i * adv[gi].detach()).mean())
        rl_loss   = torch.stack(pg_losses).mean()
        reward_mean = rewards_t.mean().item()

    else:  # dpo
        with torch.no_grad(), autocast_ctx:
            logits_sg = model(prompts)[:, -1, :].float()
        probs       = F.softmax(logits_sg, dim=-1)
        completions = [torch.multinomial(probs, 1).squeeze(1) for _ in range(G)]

        chosen_tok, rejected_tok, valid_idx = [], [], []
        for b in range(B):
            ok  = [c[b].item() for c in completions if c[b].item() == targets[b].item()]
            bad = [c[b].item() for c in completions if c[b].item() != targets[b].item()]
            if ok and bad:
                chosen_tok.append(ok[0])
                rejected_tok.append(bad[0])
                valid_idx.append(b)

        if not valid_idx:
            # All G samples correct OR all wrong — skip step, record approx reward
            reward_mean = (probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                           ).mean().item()
            torch.cuda.synchronize()
            rl_step += 1
            if rl_step % 20 == 0:
                print(f"  rl {rl_step:04d} | DPO skip (no pair)  reward~={reward_mean:.4f}",
                      flush=True)
            continue

        vi         = torch.tensor(valid_idx, device=device)
        vp         = prompts[vi]
        chosen_t   = torch.tensor(chosen_tok,   device=device, dtype=torch.long)
        rejected_t = torch.tensor(rejected_tok, device=device, dtype=torch.long)

        with autocast_ctx:
            logits_p = model(vp)[:, -1, :].float()
        log_p    = F.log_softmax(logits_p, dim=-1)
        lp_ch    = log_p.gather(1, chosen_t.unsqueeze(1)).squeeze(1)
        lp_rj    = log_p.gather(1, rejected_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad(), autocast_ctx:
            logits_ref = ref_model(vp)[:, -1, :].float()
        log_ref     = F.log_softmax(logits_ref, dim=-1)
        lp_ch_ref   = log_ref.gather(1, chosen_t.unsqueeze(1)).squeeze(1).detach()
        lp_rj_ref   = log_ref.gather(1, rejected_t.unsqueeze(1)).squeeze(1).detach()

        logits_dpo = BETA_DPO * ((lp_ch - lp_ch_ref) - (lp_rj - lp_rj_ref))
        rl_loss    = -F.logsigmoid(logits_dpo).mean()
        reward_mean = (logits_dpo.detach() > 0).float().mean().item()

    # KWT: push (1-reward) as non-stationarity signal; high reward drop = non-stationary
    kwt_signal = 1.0 - reward_mean
    kwt_scale  = kwt.push_and_scale(kwt_signal) if KWT_ENABLED else 1.0
    if kwt_scale > 1.0:
        for g in optimizer.param_groups:
            g["lr"] *= kwt_scale
        kwt_activations += 1

    optimizer.zero_grad()
    rl_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if kwt_scale > 1.0:
        for g in optimizer.param_groups:
            g["lr"] /= kwt_scale

    torch.cuda.synchronize()
    dt = time.time() - t0
    if rl_step > 5:
        rl_time += dt

    f_star = max(0.0, kwt_scale - 1.0)
    trace_fh.write(json.dumps({
        "step": rl_step, "reward": reward_mean, "loss": rl_loss.item(),
        "kwt_scale": kwt_scale, "f_star": f_star,
    }) + "\n")
    trace_fh.flush()

    if rl_step % 10 == 0:
        reward_curve["steps"].append(rl_step)
        reward_curve["rewards"].append(reward_mean)
        print(f"  rl {rl_step:04d} | reward={reward_mean:.4f} | loss={rl_loss.item():.4f} | "
              f"kwt={kwt_scale:.3f} | t={rl_time:.1f}s", flush=True)

    rl_step += 1
    if rl_time >= RL_SEC:
        break

trace_fh.close()

peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"\n=== Done  label={LABEL}  reward={reward_mean:.4f}  "
      f"kwt_act={kwt_activations}/{rl_step}  vram={peak_vram_mb:.0f}MB ===", flush=True)

result = {
    "label":            LABEL,
    "algo":             ALGO,
    "swelu":            int(SWELU_ENABLED),
    "kwt":              int(KWT_ENABLED),
    "seed":             SEED,
    "warmup_steps":     warmup_steps,
    "warmup_final_loss": warmup_loss,
    "rl_steps":         rl_step,
    "rl_final_reward":  reward_mean,
    "rl_reward_curve":  reward_curve,
    "kwt_activations":  kwt_activations,
    "kwt_activations_pct": 100 * kwt_activations / max(rl_step, 1),
    "peak_vram_mb":     peak_vram_mb,
    "depth":            DEPTH,
    "num_params":       num_params,
    "rl_seq_len":       RL_SEQ_LEN,
    "G":                G,
    "beta_dpo":         BETA_DPO,
}
Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_FILE).write_text(json.dumps(result, indent=2))
print(f"Saved → {OUT_FILE}", flush=True)
