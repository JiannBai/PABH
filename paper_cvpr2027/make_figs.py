#!/usr/bin/env python3
"""Generate the retained CVPR progress-purity mechanism figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = Path("/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE")
E2 = Path("/mnt/tidalfs-hssh01/dataset/bxd/research/PABPO/research_archive/e2_results")
OUT = B / "paper_cvpr2027" / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "figure.dpi": 200,
    "pdf.fonttype": 42,
})
C_BASE = "#315C81"
C_OURS = "#7157C8"

# Full-run retrospective diagnostics. These values are not used to recompute
# the in-budget means in Table 1; see the main-table protocol note.
RUN_META = {
    "1.5B_TRUEBASE_rawactorhidden.json": (89.8, "1.5B"),
    "1.5B_dg_anchor.json": (87.5, "1.5B"),
    "BiPACE_headPRM_1.5b_seed0.json": (93.0, "1.5B"),
    "BiPACE_headPRM_1.5b_seed1.json": (96.1, "1.5B"),
    "BiPACE_headPRM_1.5b_seed2.json": (96.1, "1.5B"),
    "1.5B_headPRM_seed3_s250.json": (94.5, "1.5B"),
    "1.5B_headPRM_seed5_s250.json": (96.1, "1.5B"),
    "PABPO_B3v2_1.5b_seed0.json": (92.2, "1.5B"),
    "7B_rawanchor_seed0.json": (92.2, "7B"),  # union of all resume segments
    "7B_rawanchor_seed1.json": (95.3, "7B"),
    "7B_midarm_gtext_seed0.json": (93.8, "7B"),
    "7B_midarm_gtext_seed1.json": (95.3, "7B"),
    "7B_headPRM_seed1.json": (95.3, "7B"),
    "7B_headPRM_seed2.json": (95.3, "7B"),
    "7B_stabA_s200_seed1.json": (95.3, "7B"),
}


def fig_purity():
    def _seg_means(data, key):
        by = {}
        for r in data:
            seg = (r["ppo_step"] - 1) // 50
            by.setdefault(seg, []).append(r[key] * 100)
        return {k: float(np.mean(v)) for k, v in by.items()}

    fig, a1 = plt.subplots(figsize=(4.4, 2.7))
    colors = {"1.5B": "#8B969C", "7B": "#476D78"}

    # Coverage stays flat while pair-level progress contamination grows.
    # Both metrics are percentages of the exact-match grouping.
    for scale in ("1.5B", "7B"):
        cov, pur = {}, {}
        for filename, (_, s) in RUN_META.items():
            if s != scale:
                continue
            with open(E2 / filename) as handle:
                data = json.load(handle)
            for k, v in _seg_means(data, "singleton_frac").items():
                cov.setdefault(k, []).append(v)
            for k, v in _seg_means(data, "conf_pair_frac").items():
                pur.setdefault(k, []).append(v)
        segs = sorted(set(cov) | set(pur))
        xs = [(s + 0.5) * 50 for s in segs]
        cov_y = [np.nanmean(cov.get(s, [np.nan])) for s in segs]
        pur_y = [np.nanmean(pur.get(s, [np.nan])) for s in segs]
        a1.plot(xs, cov_y, "--", color=colors[scale], lw=1.4,
                label=f"{scale} singleton coverage")
        a1.plot(xs, pur_y, "-o", color=colors[scale], ms=3.5, lw=1.5,
                label=f"{scale} pair contamination")
    a1.set_xlabel("PPO step")
    a1.set_ylabel("share of exact-match grouping (%)")
    a1.set_ylim(0, 22)
    a1.legend(frameon=False, fontsize=7.2, loc="upper left")
    a1.set_title("singleton coverage hides growing progress contamination", fontsize=8.8)
    a1.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_purity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_purity()
    print("fig_purity.pdf")
