#!/usr/bin/env python3
"""
analyze_rl.py — Analyse statistique Wilcoxon/Cliff + charts BCUB3 du benchmark RL.

Usage:
    python3 analyze_rl.py --results-dir /workspace/results_rl --out /workspace/viz_rl

Sorties:
    rl_summary.png          bar-chart 4 conditions × reward final
    rl_curves.png           reward curves seed-moyennées
    rl_kwt_scale.png        activations KWT par condition
    rl_report.md            rapport avec Wilcoxon+Cliff verdict

CONFIDENTIEL — KWT/sWELU = IP interne BCUB3/POWWPOL. NE PAS DIFFUSER.
"""
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/nika/vault/scripts")
try:
    from bcub3_brand import apply_bcub3_style, brand_axes, add_brand_header, C as _C, SERIES
    apply_bcub3_style()
    _BRAND = True
except ImportError:
    _BRAND = False
    _C     = {"teal_deep": "#5E9384", "coral_deep": "#C97A55", "cream": "#FDFBF8",
               "anthracite": "#2C3E42", "signal": "#15803D", "gray_mid": "#566569",
               "teal": "#7DB5A5", "coral": "#E99971"}
    SERIES = [_C["teal_deep"], _C["coral_deep"], _C["signal"], _C["gray_mid"]]


# ── Palette conditions ─────────────────────────────────────────────────────────
COND_COLORS = {
    "grpo_base": _C["teal_deep"],
    "grpo_kwt":  _C["coral_deep"],
    "dpo_base":  _C["signal"],
    "dpo_kwt":   _C["gray_mid"],
}
COND_LABELS = {
    "grpo_base": "GRPO baseline",
    "grpo_kwt":  "GRPO + sWELU+KWT",
    "dpo_base":  "DPO baseline",
    "dpo_kwt":   "DPO + sWELU+KWT",
}


def cond_key(r):
    a = r["algo"]
    suf = "kwt" if r["kwt"] else "base"
    return f"{a}_{suf}"


def load_results(results_dir):
    data = defaultdict(list)
    for p in sorted(Path(results_dir).glob("results_*.json")):
        try:
            r = json.loads(p.read_text())
            ck = cond_key(r)
            data[ck].append(r)
        except Exception as e:
            print(f"WARN: skip {p.name}: {e}", flush=True)
    return data


def wilcoxon_cliff(a, b):
    """Paired Wilcoxon + Cliff's delta on matched seed vectors."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None, None, None
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = a - b
    if np.all(diffs == 0):
        return 1.0, 0.0, "trivial"
    stat, pval = wilcoxon(diffs, alternative="two-sided")
    # Cliff's delta
    pairs_a_gt = sum(ai > bi for ai in a for bi in b)
    pairs_b_gt = sum(bi > ai for ai in a for bi in b)
    n2 = len(a) * len(b)
    delta = (pairs_a_gt - pairs_b_gt) / n2
    if abs(delta) < 0.147:
        mag = "negligible"
    elif abs(delta) < 0.33:
        mag = "small"
    elif abs(delta) < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return stat, pval, delta, mag


def make_summary_plot(data, out_dir):
    conds = ["grpo_base", "grpo_kwt", "dpo_base", "dpo_kwt"]
    means, stds, labels, colors = [], [], [], []
    for ck in conds:
        runs = data.get(ck, [])
        rewards = [r["rl_final_reward"] for r in runs]
        if not rewards:
            means.append(0); stds.append(0)
        else:
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        labels.append(COND_LABELS.get(ck, ck))
        colors.append(COND_COLORS.get(ck, "#888"))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.1, right=0.97)
    if _BRAND:
        brand_axes(ax)

    x = np.arange(len(conds))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  width=0.55, edgecolor=_C["anthracite"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Reward final moyen (cloze, ↑ mieux)", fontsize=9)
    ax.set_ylim(0, max(means) * 1.25 if means else 1.0)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005,
                f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=8,
                color=_C["anthracite"])

    if _BRAND:
        add_brand_header(fig, "Benchmark RL — GRPO vs DPO × ±sWELU+KWT",
                         "3 seeds · reward cloze FineWeb · H100 SECURE")
    else:
        fig.suptitle("Benchmark RL — GRPO vs DPO × ±sWELU+KWT  (3 seeds)", fontsize=12)

    out = Path(out_dir) / "rl_summary.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out}", flush=True)
    return str(out)


def make_curves_plot(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.97, wspace=0.28)
    for ax in axes:
        if _BRAND:
            brand_axes(ax)

    pairs = [("grpo_base", "grpo_kwt", axes[0], "GRPO"),
             ("dpo_base",  "dpo_kwt",  axes[1], "DPO")]

    for base_ck, kwt_ck, ax, title in pairs:
        for ck in (base_ck, kwt_ck):
            runs = data.get(ck, [])
            if not runs:
                continue
            max_steps = max(len(r["rl_reward_curve"]["steps"]) for r in runs)
            # Align curves by index; pad shorter ones with last value
            all_rewards = []
            for r in runs:
                rc = r["rl_reward_curve"]["rewards"]
                if rc:
                    pad = [rc[-1]] * (max_steps - len(rc))
                    all_rewards.append(rc + pad)
            if not all_rewards:
                continue
            arr  = np.array(all_rewards)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ref_steps = runs[0]["rl_reward_curve"]["steps"]
            steps_x = list(range(len(mean)))
            ax.plot(steps_x, mean, label=COND_LABELS[ck], color=COND_COLORS[ck], lw=2.0)
            ax.fill_between(steps_x, mean - std, mean + std,
                            color=COND_COLORS[ck], alpha=0.15)

        ax.set_xlabel("Step (×10)", fontsize=8)
        ax.set_ylabel("Reward moyen (cloze)", fontsize=8)
        ax.set_title(title, fontsize=10, color=_C["anthracite"])
        ax.legend(fontsize=8)

    if _BRAND:
        add_brand_header(fig, "Courbes de reward RL — moyenne ± std 3 seeds",
                         "FineWeb cloze · 512 tokens · nanochat-GPT 50M")

    out = Path(out_dir) / "rl_curves.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out}", flush=True)
    return str(out)


def make_kwt_plot(data, out_dir):
    conds = ["grpo_kwt", "dpo_kwt"]
    labels, act_means, act_stds = [], [], []
    for ck in conds:
        runs = data.get(ck, [])
        acts = [r["kwt_activations_pct"] for r in runs]
        labels.append(COND_LABELS[ck])
        act_means.append(np.mean(acts) if acts else 0)
        act_stds.append(np.std(acts)  if acts else 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.12, right=0.97)
    if _BRAND:
        brand_axes(ax)

    x = np.arange(len(conds))
    ax.bar(x, act_means, yerr=act_stds, capsize=5,
           color=[COND_COLORS[c] for c in conds],
           width=0.45, edgecolor=_C["anthracite"], linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% steps KWT activé (scale>1)", fontsize=9)
    ax.axhline(0, color=_C["gray_mid"], lw=0.8, ls="--")
    for xi, (m, s) in enumerate(zip(act_means, act_stds)):
        ax.text(xi, m + s + 0.3, f"{m:.1f}%", ha="center", fontsize=9, color=_C["anthracite"])

    if _BRAND:
        add_brand_header(fig, "Activations KWT-NS en régime RL",
                         "% steps où f* > 0 → LR boosté")
    out = Path(out_dir) / "rl_kwt_scale.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out}", flush=True)
    return str(out)


def _sorted_by_seed(runs):
    return sorted(runs, key=lambda r: r["seed"])


def make_report(data, out_dir):
    lines = [
        "# RAPPORT RL — GRPO/DPO × sWELU+KWT",
        "",
        "**Protocole** : nanochat-GPT 50M · FineWeb cloze · "
        "3 seeds (42,43,44) · GRPO G=4 · DPO β=0.1 · "
        "warmup 60s SL + 120s RL · H100 SECURE · Wilcoxon+Cliff",
        "",
        "## Résultats par condition",
        "",
        "| Condition | Seeds | Reward moy | Reward std | KWT act% | Wilcoxon p | Cliff δ | Mag |",
        "|-----------|-------|-----------|-----------|---------|-----------|--------|-----|",
    ]

    stats = {}
    for ck, runs in data.items():
        rewards = [r["rl_final_reward"] for r in runs]
        acts    = [r["kwt_activations_pct"] for r in runs]
        lbl = COND_LABELS.get(ck, ck)
        stats[ck] = {
            "rewards": rewards,
            "mean":    np.mean(rewards) if rewards else 0,
            "std":     np.std(rewards)  if rewards else 0,
            "acts":    np.mean(acts)    if acts    else 0,
            "n":       len(rewards),
        }
        lines.append(
            f"| {lbl} | {[r['seed'] for r in runs]} | "
            f"{stats[ck]['mean']:.5f} | {stats[ck]['std']:.5f} | "
            f"{stats[ck]['acts']:.1f}% | — | — | — |"
        )

    lines += ["", "## Tests statistiques (Wilcoxon paired + Cliff δ)", ""]

    for base_ck, kwt_ck, algo in [("grpo_base","grpo_kwt","GRPO"),
                                   ("dpo_base", "dpo_kwt", "DPO")]:
        base_runs = _sorted_by_seed(data.get(base_ck, []))
        kwt_runs  = _sorted_by_seed(data.get(kwt_ck,  []))
        min_n = min(len(base_runs), len(kwt_runs))
        if min_n < 3:
            lines.append(f"### {algo}: données insuffisantes (n={min_n})")
            continue
        base_r = [r["rl_final_reward"] for r in base_runs[:min_n]]
        kwt_r  = [r["rl_final_reward"] for r in kwt_runs[:min_n]]
        res = wilcoxon_cliff(kwt_r, base_r)
        if res[0] is None:
            lines.append(f"### {algo}: scipy absent — test ignoré")
            continue
        stat, pval, delta, mag = res
        sign = "✅ sig" if pval < 0.05 else "— n.s."
        direction = "KWT > base" if delta > 0 else "base > KWT"
        lines += [
            f"### {algo}",
            f"- n={min_n} seeds · W={stat:.2f} · **p={pval:.4f}** ({sign}) · "
            f"Cliff δ={delta:+.3f} ({mag}) · {direction}",
            f"- KWT rewards: {[f'{x:.5f}' for x in kwt_r]}",
            f"- Base rewards: {[f'{x:.5f}' for x in base_r]}",
            f"- Δ moyen: {np.mean(kwt_r)-np.mean(base_r):+.5f}",
            "",
        ]

    # Verdict
    lines += ["", "## Verdict", ""]
    any_sig = False
    for base_ck, kwt_ck, algo in [("grpo_base","grpo_kwt","GRPO"),
                                   ("dpo_base", "dpo_kwt", "DPO")]:
        base_runs = _sorted_by_seed(data.get(base_ck, []))
        kwt_runs  = _sorted_by_seed(data.get(kwt_ck,  []))
        min_n = min(len(base_runs), len(kwt_runs))
        if min_n < 3:
            continue
        base_r = [r["rl_final_reward"] for r in base_runs[:min_n]]
        kwt_r  = [r["rl_final_reward"] for r in kwt_runs[:min_n]]
        res = wilcoxon_cliff(kwt_r, base_r)
        if res[0] is None:
            continue
        _, pval, delta, mag = res
        if pval < 0.05 and abs(delta) >= 0.147:
            any_sig = True
            direction = "GAIN" if delta > 0 else "PERTE"
            lines.append(f"- **{algo} : {direction} stat-sig** "
                         f"(p={pval:.4f}, Cliff δ={delta:+.3f} {mag})")
        else:
            lines.append(f"- **{algo} : NEUTRE** "
                         f"(p={pval:.4f}, Cliff δ={delta:+.3f} {mag})")

    if any_sig:
        lines.append("")
        lines.append("> ⚠️ Résultat significatif détecté — vérifier reproductibilité "
                     "avant de muter le référentiel.")
    else:
        lines.append("")
        lines.append("> Aucun gain stat-sig de sWELU+KWT en régime RL sur cette configuration. "
                     "Verdict : **NEUTRE**. Pas de mutation référentiel.")

    out = Path(out_dir) / "rl_report.md"
    out.write_text("\n".join(lines))
    print(f"[report] {out}", flush=True)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/workspace/results_rl")
    ap.add_argument("--out",         default="/workspace/viz_rl")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    data = load_results(args.results_dir)
    if not data:
        print(f"No results found in {args.results_dir}", flush=True)
        return

    for ck, runs in data.items():
        rewards_str = [f"{r['rl_final_reward']:.4f}" for r in runs]
        print(f"  {ck}: {len(runs)} runs  rewards={rewards_str}")

    make_summary_plot(data, args.out)
    make_curves_plot(data,  args.out)
    make_kwt_plot(data,     args.out)
    make_report(data,       args.out)
    print(f"\nDone → {args.out}", flush=True)


if __name__ == "__main__":
    main()
