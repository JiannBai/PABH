#!/usr/bin/env python3
"""Generate the concise GRPO/GiGPO/BiPACE/PABH comparison figure."""

from pathlib import Path
from make_method_ppt import DualCanvas, C, progress

OUT = Path(__file__).resolve().parent / "figs"
PPT = OUT / "fig_method_comparison_editable.pptx"
PNG = OUT / "fig_method_comparison_preview.png"
PDF = OUT / "fig_method_comparison.pdf"


def card(c, x, title, subtitle, accent, badge=None):
    c.rect(x, 1.08, 2.88, 3.96, C["white"], C["line"], radius=0.13, width=0.9)
    c.rect(x, 1.08, 2.88, 0.055, accent, None, radius=0)
    c.text(x+0.17, 1.24, 2.54, 0.32, title, 14.0, C["ink"], True, "center")
    c.text(x+0.17, 1.58, 2.54, 0.23, subtitle, 8.0, C["muted"], False, "center")
    if badge:
        c.rect(x+0.90, 4.63, 1.08, 0.25, accent, None, radius=0.09)
        c.text(x+0.90, 4.63, 1.08, 0.25, badge, 7.0, C["white"], True, "center")


def state(c, x, y, obs, filled, color):
    c.rect(x, y, 0.89, 0.66, C["white"], color, radius=0.08, width=0.9)
    c.rect(x+0.09, y+0.09, 0.71, 0.20, C["node_purple"], None, radius=0.04)
    c.text(x+0.09, y+0.09, 0.71, 0.20, obs, 6.8, C["purple_dark"], True, "center")
    progress(c, x+0.18, y+0.41, filled, color, seg=0.12, gap=0.035)


def build():
    c = DualCanvas(PPT, PNG, PDF, width=13.333, height=5.35)
    c.text(0.30, 0.17, 10.7, 0.40,
           "From Outcome Groups to Progress-Aware Behavioral Peers",
           21.5, C["ink"], True)
    c.text(0.32, 0.62, 10.5, 0.24,
           "The rollout budget stays fixed; the notion of comparable states changes.",
           10.2, C["muted"])
    c.rect(11.15, 0.28, 1.72, 0.34, C["soft"], None, radius=0.10)
    c.text(11.15, 0.28, 1.72, 0.34, "same rollouts  •  same PPO", 7.4, C["muted"], True, "center")

    card(c, 0.28, "GRPO", "episode-relative credit", C["gray"])
    card(c, 3.58, "GiGPO", "exact anchor-state groups", C["navy"])
    card(c, 6.88, "BiPACE", "raw-hidden behavioral groups", C["navy"])
    card(c, 10.18, "PABH  (ours)", "trained progress-aware groups", C["purple"], "PROGRESS-AWARE")

    # GRPO: compare complete trajectories, never individual states.
    for i, ly in enumerate([2.02, 2.55, 3.08]):
        c.text(0.49, ly+0.05, 0.28, 0.25, f"τ{i+1}", 8.0, C["muted"], True, "center")
        xs = [0.89, 1.31, 1.73, 2.15]
        for k, x in enumerate(xs):
            c.rect(x, ly, 0.29, 0.31,
                   [C["blue_light"], C["yellow_light"], C["pink_light"], C["green_light"]][k],
                   C["white"], radius=0.05, width=0.6)
            if k < 3:
                c.line(x+0.29, ly+0.155, x+0.40, ly+0.155, C["gray"], 0.7, True)
        c.text(2.57, ly+0.03, 0.28, 0.25, f"R{i+1}", 7.4, C["navy"], True, "center")
    c.line(0.81, 3.56, 2.51, 3.56, C["gray"], 1.1)
    c.line(0.81, 3.50, 0.81, 3.56, C["gray"], 1.1)
    c.line(2.51, 3.50, 2.51, 3.56, C["gray"], 1.1)
    c.text(0.82, 3.63, 1.68, 0.23, "one group of full episodes", 7.4, C["muted"], False, "center")
    c.rect(0.69, 4.05, 2.05, 0.38, C["gray_light"], None, radius=0.09)
    c.text(0.69, 4.05, 2.05, 0.38, "coarse: every step shares A_episode", 7.3, C["muted"], True, "center")

    # GiGPO: exact text match can merge different progress and split other peers.
    c.rect(3.83, 2.02, 2.38, 0.93, C["orange_light"], C["orange"], radius=0.12, width=0.9, dash=True)
    state(c, 3.98, 2.15, "same o*", 1, C["orange"])
    state(c, 5.15, 2.15, "same o*", 3, C["teal"])
    c.text(4.01, 3.03, 2.02, 0.22, "exact match → mixed progress", 7.2, C["orange"], True, "center")
    state(c, 4.56, 3.40, "different o", 3, C["teal"])
    c.text(5.53, 3.52, 0.52, 0.25, "singleton", 7.0, C["muted"], False, "center")
    c.rect(3.98, 4.05, 2.08, 0.38, C["navy_light"], None, radius=0.09)
    c.text(3.98, 4.05, 2.08, 0.38, "step credit, but rigid state key", 7.3, C["navy"], True, "center")

    # BiPACE: more approximate peers, but raw geometry remains untrained.
    c.rect(7.18, 2.03, 2.28, 1.52, C["soft"], C["gray"], radius=0.28, width=0.9, dash=True)
    for cx, cy, color in [(7.55, 2.42, C["teal"]), (8.05, 2.34, C["orange"]),
                          (8.63, 2.51, C["teal"]), (7.73, 2.94, C["orange"]),
                          (8.37, 3.08, C["teal"]), (8.93, 2.91, C["orange"])]:
        c.ellipse(cx, cy, 0.11, color, C["white"], 0.7)
    c.text(7.25, 3.65, 2.14, 0.23, "more coverage, still mixed", 7.2, C["muted"], True, "center")
    c.rect(7.30, 4.05, 2.04, 0.38, C["navy_light"], None, radius=0.09)
    c.text(7.30, 4.05, 2.04, 0.38, "raw h + PACE: untrained metric", 7.3, C["navy"], True, "center")

    # PABH: progress state and learned metric produce pure behavioral peers.
    state(c, 10.42, 2.02, "same o*", 1, C["orange"])
    state(c, 10.42, 2.90, "same o*", 3, C["teal"])
    c.line(11.36, 2.69, 11.72, 2.69, C["purple"], 1.4, True)
    c.rect(11.74, 2.39, 0.55, 0.58, C["purple"], None, radius=0.13)
    c.text(11.74, 2.39, 0.55, 0.58, "W", 15.0, C["white"], True, "center")
    c.rect(12.37, 2.00, 0.48, 0.75, C["orange_light"], C["orange"], radius=0.17, width=0.8, dash=True)
    c.ellipse(12.61, 2.37, 0.105, C["orange"], C["white"], 0.6)
    c.rect(12.37, 2.94, 0.48, 0.75, C["teal_light"], C["teal"], radius=0.17, width=0.8, dash=True)
    c.ellipse(12.61, 3.31, 0.105, C["teal"], C["white"], 0.6)
    c.text(10.43, 3.86, 2.39, 0.22, "progress-pure peers for PACE", 7.3, C["purple_dark"], True, "center")
    c.rect(10.56, 4.17, 2.12, 0.30, C["purple_light"], None, radius=0.08)
    c.text(10.56, 4.17, 2.12, 0.30, "g_text + PRM-trained z", 7.4, C["purple_dark"], True, "center")

    c.save()
    print(PPT)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
