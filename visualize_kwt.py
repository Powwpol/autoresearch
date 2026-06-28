#!/usr/bin/env python3
"""
visualize_kwt.py — Génère vidéo + plots baseline vs KWT pour le benchmark nanochat-GPT.

Sorties :
  kwt_curves.png        — plot statique comparaison val_bpb proxy
  kwt_training.mp4      — animation des courbes en temps réel
  kwt_scale_dynamics.png — dynamique du scale KWT (si kwt_trace.jsonl dispo)

Usage :
    python3 visualize_kwt.py --results /workspace/results_kwt_bench.json
                             --kwt-trace /workspace/kwt_trace.jsonl
                             --out /workspace
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

sys.path.insert(0, "/home/nika/vault/scripts")
try:
    from bcub3_brand import apply_bcub3_style, brand_axes, add_brand_header, C as _C, SERIES as _SERIES
    apply_bcub3_style()
    C_BASE  = _C["teal_deep"]    # #5E9384
    C_KWT   = _C["coral_deep"]   # #C97A55
    C_SCALE = _C["signal"]       # #15803D
    _BRAND  = True
except ImportError:
    _BRAND  = False
    C_BASE  = "#4c72b0"
    C_KWT   = "#dd8452"
    C_SCALE = "#55a868"


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_kwt_trace(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def make_static_plot(results: dict, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.97, wspace=0.3)

    base = results.get("baseline", {})
    kwt  = results.get("kwt", {})
    delta = results.get("delta_val_bpb")

    ax_curve = axes[0]
    ax_bar   = axes[1]

    for ax in axes:
        if _BRAND:
            brand_axes(ax)

    # Step-level curves
    for label, run, color in [("Baseline", base, C_BASE), ("Baseline+KWT", kwt, C_KWT)]:
        curve = run.get("step_curve")
        if curve:
            ax_curve.plot(curve["steps"], curve["loss_proxy"],
                          label=label, color=color, linewidth=2.0)

    ax_curve.set_xlabel("Pas d'entraînement")
    ax_curve.set_ylabel("val_bpb proxy (loss EMA)")
    ax_curve.set_title("Courbes d'entraînement")
    ax_curve.legend()

    # Final val_bpb bar chart
    labels, vals, colors = [], [], []
    for label, run, color in [("Baseline", base, C_BASE), ("Baseline+KWT", kwt, C_KWT)]:
        v = run.get("val_bpb")
        if v is not None:
            labels.append(label); vals.append(v); colors.append(color)

    if vals:
        ec = _C["anthracite"] if _BRAND else "white"
        bars = ax_bar.bar(labels, vals, color=colors, width=0.4, edgecolor=ec, linewidth=0.8)
        ax_bar.set_ylabel("val_bpb final (bits/byte, ↓ mieux)")
        ymin = min(vals) * 0.995; ymax = max(vals) * 1.005
        ax_bar.set_ylim(ymin, ymax)
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, val + (ymax - ymin) * 0.003,
                        f"{val:.6f}", ha="center", va="bottom", fontsize=10)
        sign = "+" if delta and delta >= 0 else ""
        ax_bar.set_title(f"val_bpb final  (Δ = {sign}{delta:.6f})" if delta is not None else "val_bpb final")

    if _BRAND:
        add_brand_header(fig,
                         "KWT-NS vs Baseline — nanochat-GPT",
                         "val_bpb proxy · 5 min H100 PCIe · seed=42")
    else:
        fig.suptitle("KWT vs Baseline — nanochat-GPT val_bpb proxy (5 min H100)", fontsize=13)

    out = Path(out_dir) / "kwt_curves.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[static] saved → {out}")
    return str(out)


def make_video(results: dict, out_dir: str) -> str:
    base_curve = results.get("baseline", {}).get("step_curve")
    kwt_curve  = results.get("kwt", {}).get("step_curve")

    if not base_curve or not kwt_curve:
        print("[video] step curves missing — skipping animation")
        return ""

    xs_b, ys_b = base_curve["steps"], base_curve["loss_proxy"]
    xs_k, ys_k = kwt_curve["steps"],  kwt_curve["loss_proxy"]
    n_frames = max(len(xs_b), len(xs_k))
    delta = results.get("delta_val_bpb")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    if _BRAND:
        brand_axes(ax)
        txt_col = _C["anthracite"]
    else:
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
        txt_col = "#e6edf3"

    ax.set_xlabel("Pas d'entraînement")
    ax.set_ylabel("val_bpb proxy (loss EMA)")

    all_ys = ys_b + ys_k
    y_min = min(all_ys) * 0.98; y_max = max(all_ys) * 1.02
    x_max = max(max(xs_b or [1]), max(xs_k or [1]))
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(y_min, y_max)

    line_b, = ax.plot([], [], color=C_BASE, linewidth=2.2, label="Baseline")
    line_k, = ax.plot([], [], color=C_KWT,  linewidth=2.2, label="Baseline+KWT-NS")
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, color=txt_col, fontsize=10, va="top")
    ax.legend()

    if _BRAND:
        sign = "+" if delta and delta >= 0 else ""
        subtitle = f"Δ = {sign}{delta:.6f} bpb  ·  5 min H100 PCIe  ·  seed=42" if delta is not None else "5 min H100 PCIe · seed=42"
        add_brand_header(fig, "KWT-NS vs Baseline — nanochat-GPT", subtitle)

    def init():
        line_b.set_data([], [])
        line_k.set_data([], [])
        txt.set_text("")
        return line_b, line_k, txt

    def animate(frame):
        f_b = min(frame, len(xs_b) - 1)
        f_k = min(frame, len(xs_k) - 1)
        line_b.set_data(xs_b[:f_b + 1], ys_b[:f_b + 1])
        line_k.set_data(xs_k[:f_k + 1], ys_k[:f_k + 1])
        step = xs_b[f_b] if xs_b else 0
        txt.set_text(f"Pas {step}\nBase: {ys_b[f_b]:.5f}\n KWT: {ys_k[f_k]:.5f}")
        return line_b, line_k, txt

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, blit=True, interval=80
    )
    out = Path(out_dir) / "kwt_training.mp4"
    writer = animation.FFMpegWriter(fps=12, bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "baseline",
                    "-movflags", "+faststart",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    anim.save(str(out), writer=writer)
    plt.close(fig)
    print(f"[video] saved → {out}")
    return str(out)


def make_kwt_scale_plot(trace: list[dict], out_dir: str) -> str:
    if not trace:
        return ""
    steps  = [r["step"] for r in trace]
    fstars = [r.get("f_star", 0.0) for r in trace]
    scales = [r.get("kwt_scale", 1.0) for r in trace]
    lrms   = [r.get("lrm", 1.0) for r in trace]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.subplots_adjust(top=0.82, bottom=0.1, left=0.1, right=0.97, hspace=0.12)

    if _BRAND:
        brand_axes(ax1); brand_axes(ax2)

    ax1.fill_between(steps, 1.0, scales, alpha=0.25, color=C_KWT)
    ax1.plot(steps, scales, color=C_KWT, linewidth=1.8, label="kwt_scale = 1+f*")
    ax1.axhline(1.0, color=(_C["gray_mid"] if _BRAND else "#888"), linestyle="--",
                linewidth=0.9, label="baseline (1.0)")
    ax1.set_ylabel("LR scale factor")
    ax1.set_ylim(0.95, max(scales) * 1.05)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.fill_between(steps, 0, fstars, alpha=0.25, color=C_SCALE)
    ax2.plot(steps, fstars, color=C_SCALE, linewidth=1.8, label="f* (Kelly fraction)")
    ax2.plot(steps, lrms, color=C_BASE, linewidth=1.0, linestyle=":", label="lrm (base schedule)")
    ax2.set_xlabel("Pas d'entraînement")
    ax2.set_ylabel("Valeur")
    ax2.legend(loc="upper right", fontsize=9)

    if _BRAND:
        add_brand_header(fig, "Dynamique KWT-NS — LR scale + Kelly f*",
                         "nanochat-GPT · 5 min H100 PCIe · deadzone=0.6")
    else:
        fig.suptitle("Dynamique KWT — scale LR et fraction de Kelly f*", fontsize=12)

    out = Path(out_dir) / "kwt_scale_dynamics.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[scale] saved → {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Visualise KWT vs baseline benchmark")
    ap.add_argument("--results",   default="/workspace/results_kwt_bench.json")
    ap.add_argument("--kwt-trace", default="/workspace/kwt_trace.jsonl")
    ap.add_argument("--out",       default="/workspace")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)
    trace   = load_kwt_trace(args.kwt_trace)

    print(f"[results] baseline val_bpb={results.get('baseline', {}).get('val_bpb', 'N/A')}")
    print(f"[results] kwt val_bpb     ={results.get('kwt', {}).get('val_bpb', 'N/A')}")
    print(f"[results] delta           ={results.get('delta_val_bpb', 'N/A')}")
    if trace:
        active = sum(1 for r in trace if r.get("f_star", 0) > 0.01)
        print(f"[trace]   {len(trace)} records, KWT active {active}/{len(trace)} steps ({100*active/len(trace):.1f}%)")

    static_path = make_static_plot(results, args.out)
    video_path  = make_video(results, args.out)
    scale_path  = make_kwt_scale_plot(trace, args.out)

    print("\n=== Sorties générées ===")
    for p in [static_path, video_path, scale_path]:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
