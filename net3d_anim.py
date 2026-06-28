#!/usr/bin/env python3
"""
net3d_anim.py — Animation 3D nanochat-GPT (12-layer Transformer) pendant KWT benchmark.
Source : results_kwt_bench.json + kwt_trace.jsonl (sortie de bench_kwt.sh)
Gauche  : Transformer 12 couches en 3D — activité des couches proportionnelle au gradient
           de loss, KWT scale > 1.0 affiché comme "spike" lumineux
Droite  : val_bpb proxy baseline vs +KWT (courbes animées) + KWT scale en bas

Utilisation :
  python3 net3d_anim.py --results /workspace/results_kwt_bench.json \
                        --kwt-trace /workspace/kwt_trace.jsonl \
                        --out /workspace/viz

CONFIDENTIEL — KWT = IP interne BCUB3/POWWPOL. NE PAS DIFFUSER.
"""
import argparse, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path

# ── Architecture nanochat-GPT ─────────────────────────────────────────────────
N_LAYER = 12
N_HEAD  = 6
N_EMBD  = 768

# ── Couleurs charte ─────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_BASE    = "#4c72b0"
C_KWT     = "#dd8452"
C_SCALE   = "#55a868"
C_NEURON  = "#c9d1d9"
C_EDGE    = "#30363d"
C_ACTIVE  = "#f79300"

# ── Build transformer 3D positions ──────────────────────────────────────────
def make_transformer_positions():
    """12 couches empilées en Z; chaque couche = anneau de neurones symboliques."""
    POS = []
    n_per_layer = 8  # neurones symboliques par couche
    for li in range(N_LAYER):
        z = li * 1.5
        theta = np.linspace(0, 2 * math.pi, n_per_layer, endpoint=False)
        r = 1.0
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        zs = np.full(n_per_layer, z)
        POS.append(np.column_stack([xs, ys, zs]))
    return POS


def make_edges(POS):
    """Arêtes légères entre couches consécutives (connexions symboliques)."""
    edges = []
    for li in range(N_LAYER - 1):
        for i in range(len(POS[li])):
            for j in range(0, len(POS[li + 1]), 2):  # skip-connect pour lisibilité
                edges.append((li, i, li + 1, j))
    return edges


def load_results(results_path, kwt_trace_path):
    r = json.loads(Path(results_path).read_text())
    base_curve = r.get("baseline", {}).get("step_curve") or {"steps": [], "loss_proxy": []}
    kwt_curve  = r.get("kwt", {}).get("step_curve")      or {"steps": [], "loss_proxy": []}
    delta      = r.get("delta_val_bpb")
    base_final = r.get("baseline", {}).get("val_bpb")
    kwt_final  = r.get("kwt", {}).get("val_bpb")

    # KWT trace
    kwt_trace = []
    try:
        for line in Path(kwt_trace_path).read_text().strip().split("\n"):
            if line.strip():
                kwt_trace.append(json.loads(line))
    except Exception:
        pass
    return {
        "base_steps": base_curve["steps"],
        "base_losses": base_curve["loss_proxy"],
        "kwt_steps": kwt_curve["steps"],
        "kwt_losses": kwt_curve["loss_proxy"],
        "delta": delta,
        "base_final": base_final,
        "kwt_final": kwt_final,
        "kwt_trace": kwt_trace,
    }


def make_animation(data, out_dir):
    POS   = make_transformer_positions()
    edges = make_edges(POS)

    base_steps  = np.array(data["base_steps"])
    base_losses = np.array(data["base_losses"])
    kwt_steps   = np.array(data["kwt_steps"])
    kwt_losses  = np.array(data["kwt_losses"])
    kwt_trace   = data["kwt_trace"]

    # Map step → kwt_scale (for highlighting)
    scale_map = {}
    for row in kwt_trace:
        scale_map[row["step"]] = row.get("kwt_scale", 1.0)

    n_frames = max(len(base_steps), len(kwt_steps), 1)

    # Loss range for y-axis
    all_losses = list(base_losses) + list(kwt_losses)
    y_lo = min(all_losses) * 0.97 if all_losses else 0.9
    y_hi = max(all_losses) * 1.02 if all_losses else 1.1

    fig = plt.figure(figsize=(14, 6), facecolor=C_BG)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08, wspace=0.35)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_facecolor(C_BG)
    ax_loss = fig.add_subplot(2, 2, 2)
    ax_scale = fig.add_subplot(2, 2, 4)
    for ax in [ax_loss, ax_scale]:
        ax.set_facecolor(C_BG)
        ax.tick_params(colors="#8b949e"); ax.xaxis.label.set_color("#8b949e"); ax.yaxis.label.set_color("#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")

    def draw(fr):
        ax3d.cla()
        ax3d.set_facecolor(C_BG)

        # Current KWT step and scale
        cur_kwt_step = kwt_steps[min(fr, len(kwt_steps) - 1)] if len(kwt_steps) > 0 else 0
        kwt_scale = scale_map.get(cur_kwt_step, 1.0)
        active_layer = int(fr / max(n_frames - 1, 1) * N_LAYER)  # sweep effect
        kwt_active = kwt_scale > 1.001

        # Edges (faint)
        for (l0, i0, l1, j0) in edges:
            p0, p1 = POS[l0][i0], POS[l1][j0]
            alpha = 0.12
            ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                      color=C_EDGE, alpha=alpha, lw=0.4)

        # Neurons
        for li, P in enumerate(POS):
            is_act = (li == active_layer % N_LAYER)
            if kwt_active and is_act:
                col = C_ACTIVE; sz = 80; alph = 0.98
            elif is_act:
                col = C_BASE; sz = 55; alph = 0.9
            else:
                col = C_NEURON; sz = 28; alph = 0.55
            ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], c=col, s=sz, alpha=alph, depthshade=False, edgecolors="none")

        # Attention head indicator at active layer
        if kwt_active:
            lp = POS[active_layer % N_LAYER]
            ax3d.scatter(lp[:N_HEAD, 0], lp[:N_HEAD, 1], lp[:N_HEAD, 2],
                         c=C_KWT, s=120, alpha=0.85, depthshade=False, edgecolors="none")

        ax3d.set_axis_off()
        azim = (fr * 2.5) % 360
        ax3d.view_init(elev=22, azim=azim)
        title_scale = f"  KWT scale={kwt_scale:.3f}" if kwt_active else ""
        ax3d.set_title(
            f"nanochat-GPT · 12L×6H×768d{title_scale}\n"
            f"orange = KWT actif  bleu = attention  blanc = couche",
            color="#c9d1d9", fontsize=8.5, pad=2,
        )

        # Loss curves
        ax_loss.cla()
        ax_loss.set_facecolor(C_BG)
        ax_loss.tick_params(colors="#8b949e"); ax_loss.xaxis.label.set_color("#8b949e"); ax_loss.yaxis.label.set_color("#8b949e")
        for sp in ax_loss.spines.values(): sp.set_color("#30363d")
        if len(base_steps) > 0:
            end = min(fr + 1, len(base_steps))
            ax_loss.plot(base_steps[:end], base_losses[:end], color=C_BASE, lw=1.6, label="Baseline")
        if len(kwt_steps) > 0:
            end = min(fr + 1, len(kwt_steps))
            ax_loss.plot(kwt_steps[:end], kwt_losses[:end], color=C_KWT, lw=1.6, label="+KWT", alpha=0.9)
        ax_loss.set_ylim(y_lo, y_hi)
        ax_loss.set_ylabel("val_bpb proxy", fontsize=8)
        ax_loss.set_title("val_bpb (proxy step-level)", color="#c9d1d9", fontsize=9)
        ax_loss.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9", edgecolor="#30363d")

        # KWT scale
        ax_scale.cla()
        ax_scale.set_facecolor(C_BG)
        ax_scale.tick_params(colors="#8b949e"); ax_scale.xaxis.label.set_color("#8b949e"); ax_scale.yaxis.label.set_color("#8b949e")
        for sp in ax_scale.spines.values(): sp.set_color("#30363d")
        if kwt_trace:
            t_steps = [x["step"] for x in kwt_trace]
            t_scale = [x.get("kwt_scale", 1.0) for x in kwt_trace]
            cur_t = min(fr + 1, len(t_steps))
            ax_scale.plot(t_steps[:cur_t], t_scale[:cur_t], color=C_SCALE, lw=1.4)
            ax_scale.axhline(1.0, color="#555", lw=0.8, ls="--")
            ax_scale.set_ylim(0.98, max(t_scale) * 1.02 if t_scale else 1.1)
        ax_scale.set_xlabel("step", fontsize=8)
        ax_scale.set_ylabel("KWT scale", fontsize=8)
        ax_scale.set_title("KWT LR scale (1.0 = baseline)", color="#c9d1d9", fontsize=9)

        return []

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, blit=False, interval=80)
    out = Path(out_dir) / "kwt_net3d_gpt.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=12, bitrate=2400)
    anim.save(str(out), writer=writer)
    print(f"VIDEO -> {out}")
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",   required=True, help="results_kwt_bench.json")
    ap.add_argument("--kwt-trace", required=True, help="kwt_trace.jsonl")
    ap.add_argument("--out",       default="/workspace/viz", help="output dir")
    args = ap.parse_args()

    data = load_results(args.results, args.kwt_trace)
    out_path = make_animation(data, args.out)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
