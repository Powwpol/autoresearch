#!/usr/bin/env python3
"""
visualize_rl.py — Animation BCUB3 reward curves benchmark RL (GRPO/DPO × sWELU+KWT).

Usage:
    python3 visualize_rl.py --results-dir /workspace/results_rl --out /workspace/viz_rl

CONFIDENTIEL — KWT/sWELU = IP interne BCUB3/POWWPOL. NE PAS DIFFUSER.
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, "/home/nika/vault/scripts")
try:
    from bcub3_brand import (apply_bcub3_style, brand_axes, add_brand_header,
                              C as _C, SERIES, save_anim)
    apply_bcub3_style()
    _BRAND = True
except ImportError:
    _BRAND = False
    _C     = {"teal_deep": "#5E9384", "coral_deep": "#C97A55", "cream": "#FDFBF8",
               "anthracite": "#2C3E42", "signal": "#15803D", "gray_mid": "#566569"}
    def save_anim(anim, out, fps=7, end_hold_s=2.0):
        import subprocess
        raw = out.replace(".mp4", "_raw.mp4")
        anim.save(raw, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        subprocess.run(["/usr/bin/ffmpeg", "-y", "-i", raw,
                        "-vf", f"tpad=stop_mode=clone:stop_duration={end_hold_s}",
                        "-c:v", "libx264", "-profile:v", "baseline",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-loglevel", "error", out], check=True)


COND_LABELS = {
    "grpo_base": "GRPO baseline",
    "grpo_kwt":  "GRPO + sWELU+KWT",
    "dpo_base":  "DPO baseline",
    "dpo_kwt":   "DPO + sWELU+KWT",
}
COND_COLORS = {
    "grpo_base": _C["teal_deep"],
    "grpo_kwt":  _C["coral_deep"],
    "dpo_base":  _C["signal"],
    "dpo_kwt":   _C["gray_mid"],
}


def load_data(results_dir):
    data = defaultdict(list)
    for p in sorted(Path(results_dir).glob("results_*.json")):
        try:
            r = json.loads(p.read_text())
            algo = r["algo"]
            suf  = "kwt" if r["kwt"] else "base"
            data[f"{algo}_{suf}"].append(r)
        except Exception:
            pass
    return data


def align_curves(runs):
    """Return (steps_x, mean_arr, std_arr) aligned by index."""
    if not runs:
        return None, None, None
    max_n = max(len(r["rl_reward_curve"]["rewards"]) for r in runs)
    arrs  = []
    for r in runs:
        rc = r["rl_reward_curve"]["rewards"]
        if rc:
            pad = [rc[-1]] * (max_n - len(rc))
            arrs.append(rc + pad)
    if not arrs:
        return None, None, None
    arr = np.array(arrs)
    return list(range(max_n)), arr.mean(0), arr.std(0)


def make_animation(data, out_dir):
    conds = ["grpo_base", "grpo_kwt", "dpo_base", "dpo_kwt"]
    curves = {}
    for ck in conds:
        runs = data.get(ck, [])
        if runs:
            curves[ck] = align_curves(runs)

    if not curves:
        print("No data to animate", flush=True)
        return

    n_frames = max(len(v[0]) for v in curves.values() if v[0] is not None)

    all_y = []
    for _, mean, std in curves.values():
        if mean is not None:
            all_y.extend((mean + std).tolist())
            all_y.extend((mean - std).tolist())
    y_lo = max(0.0, min(all_y) * 0.97) if all_y else 0.0
    y_hi = min(1.0, max(all_y) * 1.03) if all_y else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.07, right=0.97, wspace=0.3)
    ax_grpo, ax_dpo = axes

    if _BRAND:
        for ax in axes:
            brand_axes(ax)

    grpo_pairs = [("grpo_base", "grpo_kwt")]
    dpo_pairs  = [("dpo_base",  "dpo_kwt")]

    def _init_ax(ax, title):
        if _BRAND:
            brand_axes(ax)
        ax.set_xlim(0, n_frames + 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Steps (×10)", fontsize=8)
        ax.set_ylabel("Reward moyen (3 seeds ± σ)", fontsize=8)
        ax.set_title(title, fontsize=10, color=_C["anthracite"])

    _init_ax(ax_grpo, "GRPO")
    _init_ax(ax_dpo,  "DPO")

    line_objs = {}
    fill_objs = {}
    for ck in conds:
        if ck not in curves or curves[ck][0] is None:
            continue
        ax = ax_grpo if ck.startswith("grpo") else ax_dpo
        col = COND_COLORS[ck]
        lbl = COND_LABELS[ck]
        ln, = ax.plot([], [], color=col, lw=2.2, label=lbl)
        line_objs[ck] = ln
    for ax in axes:
        ax.legend(fontsize=8)

    txt_grpo = ax_grpo.text(0.03, 0.96, "", transform=ax_grpo.transAxes,
                             color=_C["anthracite"], fontsize=8, va="top")
    txt_dpo  = ax_dpo.text(0.03, 0.96, "", transform=ax_dpo.transAxes,
                            color=_C["anthracite"], fontsize=8, va="top")

    if _BRAND:
        add_brand_header(fig, "Benchmark RL — GRPO vs DPO × ±sWELU+KWT",
                         "FineWeb cloze · 3 seeds · nanochat-GPT 50M · H100 SECURE")

    def draw(fr):
        artists = []
        for ck, (xs, mean, std) in curves.items():
            if xs is None:
                continue
            end = min(fr + 1, len(xs))
            xv  = xs[:end]
            mv  = mean[:end]
            line_objs[ck].set_data(xv, mv)
            artists.append(line_objs[ck])

        # text updates
        step = fr * 10
        grpo_parts, dpo_parts = [], []
        for ck, (xs, mean, std) in curves.items():
            if xs is None:
                continue
            end = min(fr + 1, len(mean)) - 1
            if end < 0:
                continue
            suffix = f"{COND_LABELS[ck]}: {mean[end]:.4f}"
            if ck.startswith("grpo"):
                grpo_parts.append(suffix)
            else:
                dpo_parts.append(suffix)
        txt_grpo.set_text(f"step≈{step}\n" + "\n".join(grpo_parts))
        txt_dpo.set_text( f"step≈{step}\n" + "\n".join(dpo_parts))
        artists += [txt_grpo, txt_dpo]
        return artists

    anim_obj = animation.FuncAnimation(
        fig, draw, frames=n_frames, blit=True, interval=90
    )
    out = Path(out_dir) / "rl_reward_anim.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_anim(anim_obj, str(out), fps=7, end_hold_s=2.0)
    plt.close(fig)
    print(f"[anim] {out}", flush=True)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/workspace/results_rl")
    ap.add_argument("--out",         default="/workspace/viz_rl")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    data = load_data(args.results_dir)
    out  = make_animation(data, args.out)
    if out:
        print(f"Done → {out}", flush=True)


if __name__ == "__main__":
    main()
