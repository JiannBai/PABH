#!/usr/bin/env python3
"""Generate restrained, paper-focused versions of Figures 1 and 2.

The original figures and generators are preserved under figs/original_v1/.
All deliverables remain editable vector PPT/PDF plus a PNG preview.
"""

from pathlib import Path

from make_method_ppt import C, DualCanvas, network


OUT = Path(__file__).resolve().parent / "figs"

# One muted accent plus neutral gray. Shape/fill, rather than extra colors,
# encodes progress and grouping.
C.update({
    "ink": "25343B", "muted": "626F76", "line": "C9D1D5",
    "panel": "FFFFFF", "soft": "F5F7F7", "white": "FFFFFF",
    "accent": "476D78", "accent_dark": "304F58", "accent_light": "E9F0F1",
    "navy": "476D78", "navy_light": "E9F0F1",
    "purple": "476D78", "purple_dark": "304F58", "purple_light": "E9F0F1",
    "teal": "476D78", "teal_dark": "304F58", "teal_light": "E9F0F1",
    "orange": "8B969C", "orange_light": "F0F2F3",
    "gray": "8B969C", "gray_light": "F0F2F3",
    "node_purple": "E9F0F1", "blue_light": "E9F0F1",
    "yellow_light": "F0F2F3", "pink_light": "F0F2F3",
    "green_light": "E9F0F1",
})


def panel(c, x, y, w, h, title, subtitle=None, ours=False):
    border = C["accent"] if ours else C["line"]
    c.rect(x, y, w, h, C["white"], border, radius=0.10,
           width=1.15 if ours else 0.85)
    c.text(x + 0.16, y + 0.12, w - 0.32, 0.30, title, 14.2,
           C["accent_dark"] if ours else C["ink"], True)
    if subtitle:
        c.text(x + 0.16, y + 0.45, w - 0.32, 0.22, subtitle, 9.7, C["muted"])


def takeaway(c, x, y, w, text, ours=False):
    c.rect(x, y, w, 0.38, C["accent_light"] if ours else C["soft"],
           None, radius=0.06)
    c.text(x + 0.05, y, w - 0.10, 0.38, text, 10.0,
           C["accent_dark"] if ours else C["muted"], ours, "center")


def progress_dot(c, cx, cy, late=True, r=0.105, number=None):
    if late:
        c.ellipse(cx, cy, r, C["accent"], C["white"], 0.65)
        if number is not None:
            c.text(cx - r, cy - r, 2 * r, 2 * r, str(number), 9.2,
                   C["white"], True, "center")
    else:
        c.ellipse(cx, cy, r, C["white"], C["gray"], 1.15)
        if number is not None:
            c.text(cx - r, cy - r, 2 * r, 2 * r, str(number), 9.2,
                   C["muted"], True, "center")


def small_state(c, x, y, w, late, label):
    c.rect(x, y, w, 0.60, C["white"], C["line"], radius=0.07, width=0.8)
    c.text(x + 0.07, y + 0.06, w - 0.36, 0.22, 'same obs  "o*"', 9.0,
           C["ink"], True)
    progress_dot(c, x + 0.18, y + 0.43, late, r=0.065)
    c.text(x + 0.29, y + 0.31, w - 0.34, 0.23, label, 8.8,
           C["accent_dark"] if late else C["muted"], late)


def legend(c, x, y, size=9.5):
    progress_dot(c, x, y + 0.10, True, r=0.055)
    c.text(x + 0.10, y, 1.25, 0.20, "later progress g₂", size, C["muted"])
    progress_dot(c, x + 1.48, y + 0.10, False, r=0.055)
    c.text(x + 1.58, y, 1.40, 0.20, "earlier progress g₁", size, C["muted"])


def save_with_font_scale(c, scale):
    """Keep the layout/aspect ratio but enlarge text relative to the final PDF."""
    c.fig.set_size_inches(c.width * scale, c.height * scale)
    c.save()


def build_comparison():
    ppt = OUT / "fig_method_comparison_editable.pptx"
    png = OUT / "fig_method_comparison_preview.png"
    pdf = OUT / "fig_method_comparison.pdf"
    c = DualCanvas(ppt, png, pdf, width=13.333, height=3.78)
    xs = [0.20, 3.48, 6.76, 10.04]
    w = 3.12
    panel(c, xs[0], 0.18, w, 3.22, "GRPO", "complete-episode comparison")
    panel(c, xs[1], 0.18, w, 3.22, "GiGPO", "exact-observation comparison")
    panel(c, xs[2], 0.18, w, 3.22, "BiPACE", "raw-hidden comparison")
    panel(c, xs[3], 0.18, w, 3.22, "PABH (ours)", "learned progress-aware comparison", True)

    # GRPO: all steps in complete trajectories share episode-level credit.
    x = xs[0]
    for i, yy in enumerate([0.96, 1.43, 1.90]):
        c.text(x + 0.14, yy, 0.32, 0.25, f"τ{i+1}", 10.0, C["muted"], True, "center")
        for k in range(4):
            xx = x + 0.57 + 0.46 * k
            c.rect(xx, yy, 0.29, 0.25, C["soft"], C["line"], radius=0.04, width=0.65)
            if k < 3:
                c.line(xx + 0.30, yy + 0.125, xx + 0.43, yy + 0.125, C["gray"], 0.7, True)
        c.text(x + 2.51, yy, 0.38, 0.25, f"R{i+1}", 9.5, C["muted"], True, "center")
    c.line(x + 0.52, 2.36, x + 2.39, 2.36, C["gray"], 0.9)
    c.line(x + 0.52, 2.30, x + 0.52, 2.36, C["gray"], 0.9)
    c.line(x + 2.39, 2.30, x + 2.39, 2.36, C["gray"], 0.9)
    c.text(x + 0.48, 2.42, 1.96, 0.22, "one group of full episodes", 9.3, C["muted"], False, "center")
    takeaway(c, x + 0.22, 2.91, w - 0.44, "coarse credit; no state peers")

    # GiGPO: identical feedback can conceal different progress.
    x = xs[1]
    c.rect(x + 0.17, 0.88, 2.78, 1.36, C["white"], C["gray"],
           radius=0.10, width=0.95, dash=True)
    small_state(c, x + 0.31, 1.04, 1.16, False, "g₁")
    small_state(c, x + 1.64, 1.04, 1.16, True, "g₂")
    c.text(x + 0.31, 1.74, 2.49, 0.22, "exact match mixes g₁ and g₂", 9.5,
           C["muted"], True, "center")
    progress_dot(c, x + 1.08, 2.48, True, r=0.075)
    c.text(x + 1.24, 2.35, 1.42, 0.25, "different obs → singleton", 9.0, C["muted"])
    takeaway(c, x + 0.22, 2.91, w - 0.44, "step groups, but a rigid state key")

    # BiPACE: broader raw-hidden grouping still mixes progress.
    x = xs[2]
    c.rect(x + 0.24, 0.86, 2.64, 1.52, C["soft"], C["gray"],
           radius=0.26, width=0.95, dash=True)
    points = [(0.62, 1.22, True), (1.20, 1.12, False), (1.83, 1.31, True),
              (0.82, 1.79, False), (1.48, 1.66, True), (2.20, 1.87, False)]
    for dx, yy, late in points:
        progress_dot(c, x + dx, yy, late, r=0.105)
    c.text(x + 0.40, 2.47, 2.32, 0.22, "one mixed cluster in raw h", 9.5,
           C["muted"], True, "center")
    takeaway(c, x + 0.22, 2.91, w - 0.44, "more coverage; untrained geometry")

    # PABH: the learned metric separates the same records by progress.
    x = xs[3]
    small_state(c, x + 0.17, 0.87, 1.04, False, "g₁")
    small_state(c, x + 0.17, 1.63, 1.04, True, "g₂")
    c.line(x + 1.24, 1.56, x + 1.43, 1.56, C["accent"], 1.2, True)
    c.rect(x + 1.46, 1.29, 0.48, 0.55, C["accent"], None, radius=0.09)
    c.text(x + 1.46, 1.29, 0.48, 0.55, "W", 15.0, C["white"], True, "center")
    c.line(x + 1.97, 1.56, x + 2.14, 1.56, C["accent"], 1.2, True)
    for yy, late in [(0.91, True), (1.68, False)]:
        c.rect(x + 2.17, yy, 0.72, 0.58, C["white"],
               C["accent"] if late else C["gray"], radius=0.12, width=0.9, dash=True)
        progress_dot(c, x + 2.53, yy + 0.29, late, r=0.10)
    c.text(x + 0.31, 2.47, 2.48, 0.22, "learned z separates g₁ and g₂", 9.5,
           C["accent_dark"], True, "center")
    takeaway(c, x + 0.22, 2.91, w - 0.44, "progress-aware peers for PACE", True)

    c.text(0.24, 3.48, 6.60, 0.20,
           "Same rollouts and PPO; only the comparison unit changes.",
           9.5, C["muted"])
    legend(c, 9.65, 3.46, size=9.2)
    save_with_font_scale(c, 0.90)
    print(ppt)
    print(png)
    print(pdf)


def example_card(c, x, y, late, title, events):
    c.rect(x, y, 1.42, 1.57, C["white"], C["line"], radius=0.08, width=0.85)
    c.text(x + 0.08, y + 0.09, 1.26, 0.25, title, 10.5,
           C["accent_dark"] if late else C["muted"], True, "center")
    c.rect(x + 0.09, y + 0.42, 1.24, 0.34, C["soft"], None, radius=0.04)
    c.text(x + 0.11, y + 0.42, 1.20, 0.34, '"pick up kettle 1"', 8.5,
           C["ink"], False, "center")
    progress_dot(c, x + 0.22, y + 0.99, late, r=0.07)
    c.text(x + 0.33, y + 0.82, 1.00, 0.31, events, 9.0,
           C["accent_dark"] if late else C["muted"], late)
    c.text(x + 0.12, y + 1.24, 1.18, 0.20, "progress events", 8.2, C["muted"], False, "center")


def cluster_points(c, x, y, separated=False):
    if not separated:
        c.rect(x, y, 1.34, 1.05, C["white"], C["gray"], radius=0.22, width=0.9, dash=True)
        for dx, dy, late in [(0.29, 0.33, True), (0.68, 0.25, False),
                             (0.98, 0.49, True), (0.43, 0.75, False),
                             (0.79, 0.73, True)]:
            progress_dot(c, x + dx, y + dy, late, r=0.075)
    else:
        for yy, late in [(y, True), (y + 0.65, False)]:
            c.rect(x, yy, 1.34, 0.53, C["white"],
                   C["accent"] if late else C["gray"], radius=0.16, width=0.9, dash=True)
            for dx in [0.38, 0.70, 1.00]:
                progress_dot(c, x + dx, yy + 0.265, late, r=0.075)


def build_method():
    ppt = OUT / "fig_pabh_method_editable.pptx"
    png = OUT / "fig_pabh_method_preview.png"
    pdf = OUT / "fig_pabh_method.pdf"
    c = DualCanvas(ppt, png, pdf, width=13.333, height=6.28)

    panel(c, 0.18, 0.18, 3.45, 5.47, "1  Progress-event state", "add task history to the anchor", True)
    panel(c, 3.78, 0.18, 5.25, 5.47, "2  Learn the state metric", "fit progress-aware distances", True)
    panel(c, 9.18, 0.18, 3.97, 5.47, "3  BiGPO + PACE", "inherited credit estimator")

    # Concrete motivating example and anchor construction.
    c.rect(0.39, 0.90, 3.02, 0.34, C["soft"], None, radius=0.06)
    c.text(0.45, 0.90, 2.90, 0.34, "Example: place two kettles on the table", 9.3,
           C["muted"], True, "center")
    example_card(c, 0.39, 1.30, False, "state A · g₁", "{take(k₁)}")
    example_card(c, 1.99, 1.30, True, "state B · g₂", "{take(k₁), put(k₂)}")
    c.text(1.80, 1.86, 0.20, 0.30, "≠", 16.0, C["muted"], True, "center")
    c.text(0.43, 2.94, 2.94, 0.24, "same feedback, different progress", 10.5,
           C["accent_dark"], True, "center")

    c.rect(0.39, 3.37, 1.03, 0.55, C["accent_light"], None, radius=0.07)
    c.text(0.39, 3.37, 1.03, 0.55, "[g_text ; o]", 11.0,
           C["accent_dark"], True, "center", font="Cambria Math")
    c.text(0.39, 3.96, 1.03, 0.20, "anchor only", 8.5, C["muted"], False, "center")
    c.line(1.46, 3.64, 1.68, 3.64, C["accent"], 1.1, True)
    network(c, 1.84, 3.64, C["accent"])
    c.line(2.28, 3.64, 2.51, 3.64, C["accent"], 1.1, True)
    c.ellipse(2.78, 3.64, 0.20, C["accent"], C["white"], 0.7)
    c.text(2.58, 3.44, 0.40, 0.40, "h", 12.5, C["white"], True, "center")
    c.text(1.63, 3.96, 1.39, 0.20, "late-layer actor", 8.5, C["muted"], False, "center")

    c.rect(0.39, 4.40, 3.02, 0.60, C["soft"], C["line"], radius=0.07, width=0.75)
    c.text(0.51, 4.47, 2.78, 0.23, "new progress event  →  process reward rᵖ", 9.5,
           C["ink"], True, "center", font="Cambria Math")
    c.text(0.51, 4.72, 2.78, 0.18, "used only for metric supervision", 8.4, C["muted"], False, "center")
    c.text(0.42, 5.22, 2.96, 0.20, "policy prompt and PPO reward unchanged", 8.8,
           C["muted"], False, "center")

    # h and PRM paths into the metric-learning panel.
    c.line(3.41, 3.64, 3.62, 3.64, C["accent"], 1.2)
    c.line(3.62, 3.64, 3.62, 1.72, C["accent"], 1.2)
    c.line(3.62, 1.72, 3.99, 1.72, C["accent"], 1.2, True)
    c.line(3.41, 4.70, 3.66, 4.70, C["gray"], 1.0)
    c.line(3.66, 4.70, 3.66, 4.86, C["gray"], 1.0)
    c.line(3.66, 4.86, 4.03, 4.86, C["gray"], 1.0, True)

    # Raw versus learned geometry.
    c.text(4.01, 0.89, 1.48, 0.28, "raw hidden h", 11.5, C["ink"], True, "center")
    cluster_points(c, 4.08, 1.19, separated=False)
    c.text(4.01, 2.39, 1.48, 0.24, "mixed cluster", 9.3, C["muted"], False, "center")

    c.line(5.54, 1.72, 5.82, 1.72, C["accent"], 1.3, True)
    c.rect(5.86, 1.41, 0.73, 0.62, C["accent"], None, radius=0.10)
    c.text(5.86, 1.41, 0.73, 0.62, "W", 17.0, C["white"], True, "center")
    c.text(5.68, 2.13, 1.10, 0.22, "linear head", 9.2, C["accent_dark"], True, "center")
    c.text(5.57, 2.39, 1.33, 0.22, "only W updated", 8.5, C["muted"], False, "center")
    c.line(6.62, 1.72, 6.94, 1.72, C["accent"], 1.3, True)

    c.text(7.01, 0.89, 1.76, 0.28, "z = Norm(Wh)", 11.5,
           C["accent_dark"], True, "center", font="Cambria Math")
    cluster_points(c, 7.22, 1.12, separated=True)
    c.text(7.01, 2.53, 1.76, 0.22, "progress-separated peers", 9.3,
           C["accent_dark"], False, "center")
    legend(c, 5.10, 2.90, size=8.8)

    # MICo pair fitting: current distance matches PRM + EMA successor distance.
    c.line(4.01, 3.27, 8.78, 3.27, C["line"], 0.8)
    c.text(4.04, 3.36, 4.71, 0.26, "MICo pair fitting · same rollout batch", 10.5,
           C["ink"], True, "center")
    progress_dot(c, 4.48, 4.08, True, r=0.14)
    c.text(4.34, 3.94, 0.28, 0.28, "i", 10.0, C["white"], True, "center")
    progress_dot(c, 5.63, 4.08, False, r=0.14)
    c.text(5.49, 3.94, 0.28, 0.28, "j", 10.0, C["muted"], True, "center")
    c.line(4.62, 4.08, 5.49, 4.08, C["accent"], 1.6)
    c.text(4.16, 4.29, 0.67, 0.20, "rᵢᵖ = +0.5", 8.8, C["muted"], False, "center", font="Cambria Math")
    c.text(5.29, 4.29, 0.67, 0.20, "rⱼᵖ = −0.1", 8.8, C["muted"], False, "center", font="Cambria Math")
    c.line(5.93, 4.08, 6.43, 4.08, C["gray"], 0.9, True)
    c.text(5.96, 3.80, 0.45, 0.18, "next", 8.0, C["muted"], False, "center")
    progress_dot(c, 6.85, 4.08, True, r=0.14)
    c.text(6.71, 3.94, 0.28, 0.28, "i⁺", 9.0, C["white"], True, "center")
    progress_dot(c, 8.00, 4.08, False, r=0.14)
    c.text(7.86, 3.94, 0.28, 0.28, "j⁺", 9.0, C["muted"], True, "center")
    c.line(6.99, 4.08, 7.86, 4.08, C["gray"], 1.1, False, True)
    c.text(6.51, 4.29, 1.84, 0.20, "EMA target successors", 8.6, C["muted"], False, "center")

    c.rect(4.03, 4.67, 1.75, 0.37, C["soft"], None, radius=0.05)
    c.text(4.03, 4.67, 1.75, 0.37, "current:  ‖zᵢ − zⱼ‖", 9.6,
           C["accent_dark"], True, "center", font="Cambria Math")
    c.rect(6.02, 4.67, 2.72, 0.37, C["accent_light"], None, radius=0.05)
    c.text(6.02, 4.67, 2.72, 0.37, "target:  |Δrᵖ| + γ‖z̄ᵢ⁺ − z̄ⱼ⁺‖", 9.4,
           C["accent_dark"], True, "center", font="Cambria Math")
    c.text(4.06, 5.20, 4.66, 0.26,
           "L_head = ( current distance − target distance )²",
           11.0, C["accent_dark"], True, "center", font="Cambria Math")

    # Purified cluster and action-conditioned credit example.
    c.line(8.78, 1.72, 9.40, 1.72, C["accent"], 1.4, True)
    c.rect(9.43, 0.94, 3.47, 1.74, C["soft"], C["accent"], radius=0.10, width=0.95, dash=True)
    c.text(9.57, 1.04, 3.19, 0.25, "one learned BiGPO cluster", 10.5,
           C["accent_dark"], True, "center")
    node_xs = [9.86, 10.66, 11.46, 12.44]
    nums = [1, 2, 4, 3]
    actions = ["a", "a", "a", "b"]
    returns = ["0.9", "0.7", "0.8", "0.2"]
    for xx, num, action, ret in zip(node_xs, nums, actions, returns):
        progress_dot(c, xx, 1.60, True, r=0.15, number=num)
        c.rect(xx - 0.15, 1.88, 0.30, 0.24,
               C["accent"] if action == "a" else C["gray"], None, radius=0.04)
        c.text(xx - 0.15, 1.88, 0.30, 0.24, action, 9.0, C["white"], True, "center")
        c.text(xx - 0.24, 2.22, 0.48, 0.20, f"R = {ret}", 8.5, C["muted"], False, "center")
    c.line(9.70, 2.50, 11.62, 2.50, C["accent"], 0.9)
    c.text(9.73, 2.51, 1.86, 0.18, "same-action peers", 8.3, C["muted"], False, "center")

    c.rect(9.43, 2.84, 1.61, 0.73, C["accent_light"], None, radius=0.07)
    c.text(9.49, 2.91, 1.49, 0.29, "Q̂(s,a) = 0.80", 11.5,
           C["accent_dark"], True, "center", font="Cambria Math")
    c.text(9.49, 3.25, 1.49, 0.18, "mean{0.9, 0.7, 0.8}", 8.0, C["muted"], False, "center")
    c.rect(11.23, 2.84, 1.67, 0.73, C["soft"], None, radius=0.07)
    c.text(11.29, 2.91, 1.55, 0.29, "V̂(s) = 0.65", 11.5,
           C["ink"], True, "center", font="Cambria Math")
    c.text(11.29, 3.25, 1.55, 0.18, "mean of all four returns", 8.0, C["muted"], False, "center")

    c.text(9.46, 3.82, 3.40, 0.52,
           "A_step = Q̂(s,a) − V̂(s)\n= 0.80 − 0.65 = +0.15",
           12.0, C["ink"], True, "center", font="Cambria Math")
    c.rect(9.43, 4.57, 3.47, 0.65, C["soft"], C["line"], radius=0.07, width=0.75)
    c.text(9.55, 4.62, 3.23, 0.30, "A = A_episode + ω · A_step", 11.5,
           C["ink"], True, "center", font="Cambria Math")
    c.text(9.55, 4.96, 3.23, 0.18, "extrinsic discounted returns only", 8.4,
           C["muted"], False, "center")

    # A narrow invariant strip instead of another full pipeline.
    c.rect(0.18, 5.82, 12.97, 0.34, C["soft"], None, radius=0.06)
    c.text(0.38, 5.82, 3.70, 0.34, "PPO update unchanged", 9.8, C["ink"], True)
    c.text(4.48, 5.82, 3.90, 0.34, "policy prompt unchanged", 9.5, C["muted"], False, "center")
    c.text(8.63, 5.82, 4.28, 0.34, "PRM shapes the metric, not the PPO reward", 9.5,
           C["accent_dark"], True, "right")

    save_with_font_scale(c, 0.84)
    print(ppt)
    print(png)
    print(pdf)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_comparison()
    build_method()
