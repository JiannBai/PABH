#!/usr/bin/env python3
"""Generate paper figures (WWW'27 draft) from experiment logs & E2 archives."""
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = Path("/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE")
LOG = B / "verl_log"
E2 = Path("/mnt/tidalfs-hssh01/dataset/bxd/research/PABPO/research_archive/e2_results")
OUT = B / "paper_www2027" / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "figure.dpi": 200,
})
C_BASE = "#4C72B0"
C_OURS = "#C44E52"
C_MID = "#55A868"
C_GREY = "#8C8C8C"


def extract_curve(patterns):
    """Parse step:val/success_rate from logs (ordered; later logs override)."""
    curve = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(LOG / pat))):
            txt = open(f, errors="ignore").read().replace("\r", "\n")
            for m in re.finditer(r"step:(\d+) .*?val/success_rate:([0-9.]+)", txt):
                curve[int(m.group(1))] = float(m.group(2))
    xs = sorted(curve)
    return np.array(xs), np.array([curve[x] for x in xs]) * 100


# ============================ Fig 1: main bars ============================
def fig_main():
    groups = [
        ("ALFWorld\n1.5B", 89.8, 0, 95.2, 1.4),
        ("ALFWorld\n7B", 93.3, 2.8, 95.3, 0.0),
        ("WebShop\n1.5B", 69.7, 1.1, 71.1, 0.6),
        ("WebShop\n7B", 73.1, 0.5, 74.8, 2.3),
        ("ScienceWorld\n1.5B", 86.7, 1.1, 89.6, 0.7),
    ]
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    b1 = ax.bar(x - w / 2, [g[1] for g in groups], w, yerr=[g[2] for g in groups],
                label="BiPACE-Q (repro)", color=C_BASE, capsize=2.5, error_kw=dict(lw=0.9))
    b2 = ax.bar(x + w / 2, [g[3] for g in groups], w, yerr=[g[4] for g in groups],
                label="PABH (ours)", color=C_OURS, capsize=2.5, error_kw=dict(lw=0.9))
    for i, g in enumerate(groups):
        d = g[3] - g[1]
        ax.text(i + w / 2, g[3] + g[4] + 0.7, f"+{d:.1f}", ha="center", fontsize=7.5,
                color=C_OURS, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("peak val success (%)")
    ax.set_ylim(55, 102)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_bars.pdf")
    plt.close(fig)


# ======================= Fig 2: SW 1.5B curves ===========================
def fig_curves():
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    arms = [
        ("BiPACE-Q seed0", ["sciworld_1.5b_baseline_seed0_node1613826_20260928_181206.log",
                            "sciworld_1.5b_baseline_seed0_node1613826_20260929_092957.log"], C_BASE, "-"),
        ("BiPACE-Q seed1", ["sciworld_1.5b_baseline_seed1_node1613826_20261005_120547.log",
                            "sciworld_1.5b_baseline_seed1_node1613826_20261005_205329.log"], C_BASE, "--"),
        ("PABH seed0", ["sciworld_1.5b_bisimhead_prm_seed0_node1613826_*"], C_OURS, "-"),
        ("PABH seed1", ["sciworld_1.5b_bisimhead_prm_seed1_node1613826_*"], C_OURS, "--"),
        ("PABH seed2", ["sciworld_1.5b_bisimhead_prm_seed2_node1613826_*"], C_OURS, ":"),
    ]
    for name, pats, color, ls in arms:
        xs, ys = extract_curve(pats)
        if len(xs):
            ax.plot(xs, ys, ls, color=color, label=name, alpha=0.95)
    ax.set_xlabel("PPO step"); ax.set_ylabel("val success (%)")
    ax.set_ylim(20, 95)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("ScienceWorld 1.5B (SFT cold start)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sw_curves.pdf")
    plt.close(fig)


# ===================== Fig 3: E2 purity (2 panels) =======================
RUN_META = {  # file -> (val_peak, scale, label)
    "1.5B_TRUEBASE_rawactorhidden.json": (89.8, "1.5B", "raw"),
    "1.5B_dg_anchor.json": (87.5, "1.5B", "head+DG"),
    "BiPACE_headPRM_1.5b_seed0.json": (93.0, "1.5B", "head+PRM"),
    "BiPACE_headPRM_1.5b_seed1.json": (96.1, "1.5B", "head+PRM"),
    "BiPACE_headPRM_1.5b_seed2.json": (96.1, "1.5B", "head+PRM"),
    "1.5B_headPRM_seed3_s250.json": (94.5, "1.5B", "head+PRM"),
    "1.5B_headPRM_seed5_s250.json": (96.1, "1.5B", "head+PRM"),
    "PABPO_B3v2_1.5b_seed0.json": (92.2, "1.5B", "B3 clamp"),
    "7B_rawanchor_seed0.json": (91.4, "7B", "rawanchor"),
    "7B_rawanchor_seed1.json": (95.3, "7B", "rawanchor"),
    "7B_midarm_gtext_seed0.json": (93.8, "7B", "midarm"),
    "7B_midarm_gtext_seed1.json": (95.3, "7B", "midarm"),
    "7B_headPRM_seed1.json": (95.3, "7B", "head+PRM"),
    "7B_headPRM_seed2.json": (95.3, "7B", "head+PRM"),
    "7B_stabA_s200_seed1.json": (95.3, "7B", "head+PRM"),
}


def fig_purity():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    # panel 1: scatter conf_clus vs val
    pts = []
    for f, (val, scale, lab) in RUN_META.items():
        d = json.load(open(E2 / f))
        cc = np.mean([r["conf_cluster_frac"] for r in d]) * 100
        pts.append((cc, val, scale))
    for scale, mk, color in [("1.5B", "o", C_BASE), ("7B", "s", C_OURS)]:
        xs = [p[0] for p in pts if p[2] == scale]
        ys = [p[1] for p in pts if p[2] == scale]
        a1.scatter(xs, ys, marker=mk, color=color, s=26, alpha=0.85, label=scale)
        if len(xs) > 2:
            z = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 20)
            a1.plot(xx, np.polyval(z, xx), color=color, lw=1.0, alpha=0.6)
    a1.set_xlabel("progress-confused clusters (%)")
    a1.set_ylabel("peak val success (%)")
    a1.legend(frameon=False, title="scale")
    a1.set_title("(a) contamination vs. success (15 runs)", fontsize=8.5)
    a1.spines[["top", "right"]].set_visible(False)
    # panel 2: conf_pair over training (mean across 1.5B/7B)
    for scale, color in [("1.5B", C_BASE), ("7B", C_OURS)]:
        series = []
        for f, (_, sc, _) in RUN_META.items():
            if sc != scale:
                continue
            d = json.load(open(E2 / f))
            by = {}
            for r in d:
                seg = (r["ppo_step"] - 1) // 50
                by.setdefault(seg, []).append(r["conf_pair_frac"] * 100)
            series.append({k: np.mean(v) for k, v in by.items()})
        segs = sorted({k for s in series for k in s})
        xs = [(s + 0.5) * 50 for s in segs]
        ys = [np.mean([s[k] for s in series if k in s]) for k in segs]
        a2.plot(xs, ys, "-o", color=color, ms=3.5, label=scale)
    a2.set_xlabel("PPO step")
    a2.set_ylabel("progress-confused pairs (%)")
    a2.legend(frameon=False, title="scale")
    a2.set_title("(b) contamination grows with training", fontsize=8.5)
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_purity.pdf")
    plt.close(fig)


# ===================== Fig 4: TextCraft noise =============================
def fig_textcraft():
    runs = [
        ("base (ours)", 70.3, C_BASE), ("head lr1e-3", 62.5, C_OURS), ("head lr1e-4", 60.9, C_OURS),
        ("g+PRM", 60.9, C_OURS), ("g+dg", 64.1, C_OURS), ("dg+resamp", 62.5, C_OURS),
        ("base (peer #1)", 56.2, C_BASE), ("bisim orig", 65.6, C_OURS), ("base (peer #2)", 56.2, C_BASE),
        ("head $\\tau$.005", 71.9, C_OURS), ("g+dg (peer)", 62.5, C_OURS), ("g+PRM (peer)", 62.5, C_OURS),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    xs = np.arange(len(runs))
    colors = [r[2] for r in runs]
    ax.bar(xs, [r[1] for r in runs], color=colors, width=0.7)
    ax.axhline(65.1, color="k", lw=1.0, ls="--")
    ax.text(len(runs) - 0.5, 66.2, "official BiPACE-Q 65.1$\\pm$7.0", ha="right", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in runs], rotation=38, ha="right")
    ax.set_ylabel("peak val success (%)")
    ax.set_ylim(50, 78)
    ax.spines[["top", "right"]].set_visible(False)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=C_BASE, label="BiPACE-Q baseline"),
                       mpatches.Patch(color=C_OURS, label="head variants")],
              frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_textcraft_noise.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_main(); print("fig_main_bars.pdf")
    fig_curves(); print("fig_sw_curves.pdf")
    fig_purity(); print("fig_purity.pdf")
    fig_textcraft(); print("fig_textcraft_noise.pdf")
