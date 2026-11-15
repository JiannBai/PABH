#!/usr/bin/env python3
"""Editable, publication-style comparison of step-credit grouping methods."""
from pathlib import Path
from make_method_ppt import DualCanvas, C

OUT = Path(__file__).resolve().parent / "figs"
PPT = OUT / "fig_method_comparison_editable.pptx"
PNG = OUT / "fig_method_comparison_preview.png"
PDF = OUT / "fig_method_comparison.pdf"

NAVY = "285E8E"
BLUE = "EAF3FB"
ORANGE = "D97732"
ORANGE_BG = "FFF2E7"
PURPLE = "6D4CC4"
PURPLE_BG = "F1EDFC"
TEAL = "14868C"
RED = "C95B5B"
GRAY = "7C8896"


def panel(c, x, title, subtitle, accent, fill, ours=False):
    c.rect(x, 0.20, 3.08, 4.18, "FFFFFF", accent if ours else "C9D3DF",
           radius=0.12, width=1.8 if ours else 1.0)
    c.rect(x, 0.20, 3.08, 0.64, accent, None, radius=0.12)
    c.rect(x, 0.64, 3.08, 0.20, accent, None, radius=0)
    c.text(x+0.12, 0.28, 2.84, 0.43, title, 21, "FFFFFF", True, "center")
    c.text(x+0.15, 0.94, 2.78, 0.34, subtitle, 12.2, C["ink"], True, "center")
    c.rect(x+0.20, 3.63, 2.68, 0.50, fill, None, radius=0.08)


def state(c, x, y, progress, color, w=1.18):
    c.rect(x, y, w, 0.66, "FFFFFF", color, radius=0.08, width=1.2)
    c.text(x+0.06, y+0.05, w-0.12, 0.25, 'obs  "o*"', 11.5, C["ink"], True, "center")
    c.text(x+0.06, y+0.33, w-0.12, 0.23, progress, 11, color, True, "center")


def build():
    c = DualCanvas(PPT, PNG, PDF, width=13.333, height=4.58)
    xs = [0.18, 3.49, 6.80, 10.11]
    panel(c, xs[0], "GRPO", "Compare complete episodes", GRAY, "F0F2F5")
    panel(c, xs[1], "GiGPO", "Match identical observations", NAVY, BLUE)
    panel(c, xs[2], "BiPACE", "Cluster raw hidden states", ORANGE, ORANGE_BG)
    panel(c, xs[3], "PABH  (ours)", "Learn progress-aware peers", PURPLE, PURPLE_BG, True)

    # GRPO: three full trajectories under one episode-level baseline.
    x = xs[0]
    for row, ret in enumerate(["R₁", "R₂", "R₃"]):
        y = 1.48 + row*0.60
        c.text(x+0.18, y, 0.30, 0.30, f"τ{row+1}", 12, C["muted"], True, "center")
        for k in range(4):
            xx = x+0.55+k*0.45
            c.rect(xx, y, 0.30, 0.30, BLUE if k < 3 else PURPLE_BG,
                   "AAB7C5", radius=0.035, width=0.8)
            if k < 3:
                c.line(xx+0.30, y+0.15, xx+0.43, y+0.15, GRAY, 1.0, True)
        c.text(x+2.42, y, 0.36, 0.30, ret, 12, NAVY, True, "center")
    c.line(x+0.52, 3.28, x+2.35, 3.28, GRAY, 1.2)
    c.text(x+0.50, 3.30, 1.88, 0.24, "one episode group", 11.2, C["muted"], False, "center")
    c.text(x+0.23, 3.68, 2.62, 0.38, "Coarse credit for every step", 11.8, C["ink"], True, "center")

    # GiGPO: exact match can merge unequal progress; other states remain singleton.
    x = xs[1]
    c.rect(x+0.20, 1.42, 2.68, 0.88, "FFF7F7", RED, radius=0.08, width=1.2, dash=True)
    state(c, x+0.30, 1.53, "early  g₁", ORANGE)
    state(c, x+1.58, 1.53, "later  g₂", TEAL)
    c.text(x+0.26, 2.39, 2.56, 0.30, "same text → mixed progress", 11.5, RED, True, "center")
    state(c, x+0.86, 2.79, "other  o′", NAVY, 1.36)
    c.text(x+2.23, 2.95, 0.62, 0.25, "singleton", 10.5, C["muted"], False, "center")
    c.text(x+0.23, 3.68, 2.62, 0.38, "Sparse or progress-confounded", 11.8, NAVY, True, "center")

    # BiPACE: approximate grouping improves coverage but not necessarily purity.
    x = xs[2]
    c.rect(x+0.32, 1.45, 2.44, 1.76, "FAFBFC", ORANGE, radius=0.16, width=1.2, dash=True)
    for dx, dy, col in [(0.65,0.42,TEAL),(1.15,0.28,ORANGE),(1.77,0.50,TEAL),
                        (0.82,1.08,ORANGE),(1.48,1.22,TEAL),(2.06,0.98,ORANGE)]:
        c.ellipse(x+dx, 1.45+dy, 0.15, col, "FFFFFF", 0.8)
    c.text(x+0.42, 3.26, 2.24, 0.26, "one mixed cluster in raw h", 11.2, C["muted"], True, "center")
    c.text(x+0.23, 3.68, 2.62, 0.38, "More coverage; untrained metric", 11.5, ORANGE, True, "center")

    # PABH: learned projection separates progress before PACE consumes peers.
    x = xs[3]
    state(c, x+0.20, 1.48, "early  g₁", ORANGE, 1.22)
    state(c, x+0.20, 2.43, "later  g₂", TEAL, 1.22)
    c.line(x+1.45, 2.18, x+1.73, 2.18, PURPLE, 1.6, True)
    c.rect(x+1.76, 1.88, 0.50, 0.60, PURPLE, None, radius=0.10)
    c.text(x+1.76, 1.88, 0.50, 0.60, "W", 20, "FFFFFF", True, "center")
    c.line(x+2.28, 2.18, x+2.50, 2.18, PURPLE, 1.6, True)
    c.rect(x+2.52, 1.45, 0.38, 0.78, ORANGE_BG, ORANGE, radius=0.12, width=1.1, dash=True)
    c.ellipse(x+2.64, 1.75, 0.14, ORANGE, "FFFFFF", 0.7)
    c.rect(x+2.52, 2.43, 0.38, 0.78, "E8F6F5", TEAL, radius=0.12, width=1.1, dash=True)
    c.ellipse(x+2.64, 2.73, 0.14, TEAL, "FFFFFF", 0.7)
    c.text(x+0.20, 3.31, 2.68, 0.24, "separated peers in learned z", 11.2, PURPLE, True, "center")
    c.text(x+0.23, 3.68, 2.62, 0.38, "Progress-pure peers for PACE", 11.8, PURPLE, True, "center")

    c.save()
    print(PPT); print(PNG); print(PDF)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
