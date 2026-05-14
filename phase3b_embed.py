"""phase3b_embed.py — small bottleneck embedding model, sWELU vs ReLU² A/B.

Phase 3b BCUB3 R&D : test sWELU hypothesis "shines in expressivity bottleneck".

Architecture (deliberately small to bottleneck the activation) :
  - Encoder: 4 transformer blocks × 256-d × 4 heads
  - Vocab: BPE 8192 (from prepare.py tokenizer)
  - MLP activation: sWELU(k,λ,β) [A] OR ReLU² [B] (env var ACTIVATION=swelu|relu_sq)
  - Output: mean-pool tokens → projection 256→128 → L2-normalize → embedding

Training :
  - InfoNCE contrastive, in-batch negatives
  - Batch 64 pairs, 5 min wall-time budget
  - AdamW + cosine LR

Eval :
  - Spearman correlation on STSb val (cosine sim vs human 0-5 rating)
  - Recall@10 on val pairs (100 query→positive)

Outputs : /workspace/results_phase3b.json + /workspace/run.log
"""
from __future__ import annotations
import os, json, math, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

ACTIVATION   = os.environ.get("ACTIVATION", "swelu")
PAIRS_TRAIN  = os.environ.get("PAIRS_TRAIN", "/workspace/corpus/synthetic_pairs.train.jsonl")
PAIRS_VAL    = os.environ.get("PAIRS_VAL", "/workspace/corpus/synthetic_pairs.val.jsonl")
STSB_PATH    = os.environ.get("STSB_PATH", "/workspace/corpus/stsb_val.jsonl")
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", "/root/.cache/autoresearch/tokenizer")
BATCH        = int(os.environ.get("BATCH", "64"))
N_LAYERS     = int(os.environ.get("N_LAYERS", "4"))
N_HEADS      = int(os.environ.get("N_HEADS", "4"))
N_EMBD       = int(os.environ.get("N_EMBD", "256"))
PROJ_DIM     = int(os.environ.get("PROJ_DIM", "128"))
MAX_LEN      = int(os.environ.get("MAX_LEN", "128"))
LR           = float(os.environ.get("LR", "3e-4"))
TEMP         = float(os.environ.get("TEMP", "0.05"))
TIME_BUDGET  = int(os.environ.get("TIME_BUDGET", "300"))
EVAL_EVERY   = int(os.environ.get("EVAL_EVERY", "50"))
SEED         = int(os.environ.get("SEED", "1337"))
OUT_DIR      = Path(os.environ.get("OUT_DIR", "/workspace"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)

from prepare import Tokenizer
tokenizer = Tokenizer.load(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.vocab_size
PAD_ID = 0

def encode(text, max_len=MAX_LEN):
    ids = tokenizer.encode(str(text))[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


class sWELU(nn.Module):
    def __init__(self, k_init=1.5, lambda_init=1.0, beta_init=1.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(float(k_init)))
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(float(lambda_init))))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        lam = torch.exp(self.log_lambda)
        gate = torch.sigmoid(self.beta * x)
        abs_x = torch.abs(x)
        k_eff = F.softplus(self.k) + 0.05
        weibull_neg = lam * (1.0 - torch.exp(-((abs_x / lam).clamp(min=1e-8)).pow(k_eff)))
        return x * gate - weibull_neg * (1.0 - gate)


class ReluSq(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


def make_act():
    if ACTIVATION == "swelu": return sWELU()
    if ACTIVATION == "relu_sq": return ReluSq()
    if ACTIVATION == "gelu": return nn.GELU()
    raise ValueError(ACTIVATION)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.act = make_act()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, mask=None):
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, key_padding_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(MAX_LEN, N_EMBD)
        self.blocks = nn.ModuleList([Block(N_EMBD, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.proj = nn.Linear(N_EMBD, PROJ_DIM, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        pad_mask = (input_ids == PAD_ID)
        for block in self.blocks:
            x = block(x, mask=pad_mask)
        x = self.ln_f(x)
        mask = (~pad_mask).float().unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x


def load_pairs(path):
    pairs = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "query" in d and "positive" in d:
                        pairs.append((d["query"], d["positive"]))
                except Exception: continue
    except FileNotFoundError: pass
    return pairs

print(f"[init] loading pairs...", flush=True)
train_pairs = load_pairs(PAIRS_TRAIN)
val_pairs = load_pairs(PAIRS_VAL)
print(f"[init] train={len(train_pairs)} val={len(val_pairs)}", flush=True)

def batch_iter(pairs, bs, shuffle=True):
    idx = list(range(len(pairs)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [pairs[j] for j in idx[i:i+bs]]
        if len(chunk) < 2: continue
        q = torch.tensor([encode(p[0]) for p in chunk], dtype=torch.long).to(DEVICE)
        p = torch.tensor([encode(p[1]) for p in chunk], dtype=torch.long).to(DEVICE)
        yield q, p


def load_stsb():
    rows = []
    try:
        with open(STSB_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    rows.append((d["sentence1"], d["sentence2"], float(d["score"])))
                except Exception: continue
    except FileNotFoundError: pass
    return rows

stsb = load_stsb()
print(f"[init] stsb rows = {len(stsb)}", flush=True)


@torch.no_grad()
def eval_spearman(model):
    if not stsb: return float("nan")
    model.eval()
    cos_sims, labels = [], []
    for i in range(0, len(stsb), 32):
        batch = stsb[i:i+32]
        s1 = torch.tensor([encode(b[0]) for b in batch], dtype=torch.long).to(DEVICE)
        s2 = torch.tensor([encode(b[1]) for b in batch], dtype=torch.long).to(DEVICE)
        e1 = model(s1); e2 = model(s2)
        cos_sims.extend((e1*e2).sum(dim=-1).cpu().tolist())
        labels.extend([b[2] for b in batch])
    rho, _ = spearmanr(cos_sims, labels)
    return float(rho)


@torch.no_grad()
def eval_recall(model, k=10):
    if len(val_pairs) < 2: return float("nan")
    model.eval()
    n = min(100, len(val_pairs))
    q = torch.tensor([encode(val_pairs[i][0]) for i in range(n)], dtype=torch.long).to(DEVICE)
    p = torch.tensor([encode(val_pairs[i][1]) for i in range(n)], dtype=torch.long).to(DEVICE)
    qe = model(q); pe = model(p)
    sim = qe @ pe.T
    correct = sum(1 for i in range(n) if i in sim[i].topk(k).indices.cpu().tolist())
    return correct / n


model = SmallEncoder().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"[init] model params = {n_params/1e6:.2f}M, activation = {ACTIVATION}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

t_start = time.time()
step = 0
results = {"activation": ACTIVATION, "n_params_M": n_params/1e6, "steps": [],
           "loss": [], "spearman": [], "recall_at_10": []}
model.train()
print(f"[train] starting, budget={TIME_BUDGET}s", flush=True)

while time.time() - t_start < TIME_BUDGET:
    for q, p in batch_iter(train_pairs, BATCH):
        if time.time() - t_start >= TIME_BUDGET: break
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            qe = model(q); pe = model(p)
            logits = qe @ pe.T / TEMP
            labels = torch.arange(logits.size(0), device=DEVICE)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % EVAL_EVERY == 0 or step == 1:
            rho = eval_spearman(model)
            r10 = eval_recall(model, k=10)
            results["steps"].append(step)
            results["loss"].append(float(loss.item()))
            results["spearman"].append(rho)
            results["recall_at_10"].append(r10)
            elapsed = time.time() - t_start
            print(f"step {step:05d} | loss={loss.item():.4f} | spearman={rho:.4f} | recall@10={r10:.3f} | elapsed={elapsed:.0f}s", flush=True)
            # ralph-loop compat : emit val_bpb line every 50 steps (negated spearman, lower better)
            print(f"val_bpb (step {step}): {-rho:.6f}", flush=True)
            model.train()
            if ACTIVATION == "swelu":
                for bi, b in enumerate(model.blocks):
                    act = b.mlp.act
                    k_val = float(F.softplus(act.k).item() + 0.05)
                    lam_val = float(torch.exp(act.log_lambda).item())
                    beta_val = float(act.beta.item())
                    print(f"swelu_param (step {step}, block {bi}): k={k_val:.3f} | lam={lam_val:.3f} | beta={beta_val:.3f}", flush=True)

final_rho = eval_spearman(model)
final_r10 = eval_recall(model, k=10)
results["final_spearman"] = final_rho
results["final_recall_at_10"] = final_r10
results["total_seconds"] = time.time() - t_start
results["num_steps"] = step

print(f"\n=== FINAL Phase 3b — activation={ACTIVATION} ===", flush=True)
print(f"final_spearman: {final_rho:.4f}", flush=True)
print(f"final_recall_at_10: {final_r10:.4f}", flush=True)
print(f"val_bpb: {-final_rho:.4f}", flush=True)
print(f"training_seconds: {results['total_seconds']:.1f}", flush=True)
print(f"num_steps: {step}", flush=True)
print(f"num_params_M: {n_params/1e6:.2f}", flush=True)
(OUT_DIR / "results_phase3b.json").write_text(json.dumps(results, indent=2))
