#!/usr/bin/env python3
"""Generate the redesigned PABH method figure with an editable PPT source."""
from pathlib import Path
from make_method_ppt import DualCanvas, C

OUT = Path(__file__).resolve().parent / "figs"
PPT = OUT / "fig_pabh_method_editable.pptx"
PNG = OUT / "fig_pabh_method_preview.png"
PDF = OUT / "fig_pabh_method.pdf"

NAVY = "2D6B78"
BLUE_BG = "F0F5F7"
PURPLE = "476C4E"
PURPLE_BG = "EEF4EA"
TEAL = "1F5B2A"
TEAL_BG = "EAF3E7"
ORANGE = "3B9FC4"
ORANGE_BG = "EAF5FA"
GRAY = "7A858F"


def panel(c, x, number, title, subtitle, accent, bg):
    c.rect(x, 0.20, 4.18, 5.78, "FFFFFF", "C8D2DE", radius=0.12, width=1.0)
    c.rect(x, 0.20, 4.18, 0.64, GRAY, None, radius=0.12)
    c.rect(x, 0.64, 4.18, 0.20, GRAY, None, radius=0)
    c.ellipse(x+0.35, 0.52, 0.18, "FFFFFF", None, 0)
    c.text(x+0.17, 0.34, 0.36, 0.36, str(number), 15, GRAY, True, "center")
    c.text(x+0.64, 0.28, 3.36, 0.43, title, 19, "FFFFFF", True)
    c.text(x+0.24, 0.91, 3.70, 0.24, subtitle, 11.2, C["ink"], True, "center")


def state_card(c, x, y, title, events, color, bg):
    c.rect(x, y, 3.62, 0.90, bg, color, radius=0.08, width=1.1)
    c.text(x+0.12, y+0.08, 3.38, 0.25, title, 12.3, C["ink"], True, "center")
    c.text(x+0.12, y+0.40, 3.38, 0.32, events, 11.2, color, True, "center")


def build():
    c = DualCanvas(PPT, PNG, PDF, width=13.333, height=6.55)
    panel(c, 0.18, 1, "Progress-aware state", "Same observation, different task history", NAVY, BLUE_BG)
    panel(c, 4.58, 2, "Learn the state metric", "Process-anchored MICo pair fitting", PURPLE, PURPLE_BG)
    panel(c, 8.98, 3, "Assign step credit", "BiGPO clusters + PACE estimator", TEAL, TEAL_BG)

    # Panel 1: two records, their progress state, and the anchor-only forward pass.
    state_card(c, 0.46, 1.20, 'A:  obs  "pick up kettle 1"', "g₁ = {take(k₁)}", ORANGE, ORANGE_BG)
    state_card(c, 0.46, 2.24, 'B:  obs  "pick up kettle 1"', "g₂ = {take(k₁), put(k₂)}", TEAL, TEAL_BG)
    c.text(0.52, 3.24, 3.48, 0.30, "identical text  ≠  identical state", 13, NAVY, True, "center")
    c.rect(0.46, 3.70, 1.33, 0.63, BLUE_BG, NAVY, radius=0.08, width=1.0)
    c.text(0.46, 3.70, 1.33, 0.63, "[g_text ; o]", 15, NAVY, True, "center")
    c.line(1.84, 4.01, 2.12, 4.01, NAVY, 1.4, True)
    c.rect(2.16, 3.70, 1.23, 0.63, "FFFFFF", NAVY, radius=0.08, width=1.0)
    c.text(2.16, 3.70, 1.23, 0.63, "actor\nlate layer", 11.5, NAVY, True, "center")
    c.line(3.43, 4.01, 3.68, 4.01, NAVY, 1.4, True)
    c.ellipse(3.96, 4.01, 0.24, NAVY, None, 0)
    c.text(3.72, 3.77, 0.48, 0.48, "h", 18, "FFFFFF", True, "center")
    c.text(0.48, 4.43, 3.60, 0.28, "anchor channel only · no actor gradient", 11.3, C["muted"], False, "center")
    c.rect(0.46, 4.91, 3.62, 0.68, "F5F7FA", None, radius=0.08)
    c.text(0.56, 4.96, 3.42, 0.27, "new progress event  →  process reward  rᵖ", 12.1, C["ink"], True, "center")
    c.text(0.56, 5.25, 3.42, 0.22, "supervises the metric, not the policy reward", 10.8, C["muted"], False, "center")

    # Panel 2: current projection, learned embedding separation, and pair target.
    c.text(4.90, 1.94, 0.96, 0.24, "raw h", 11.5, C["muted"], True, "center")
    c.line(5.94, 1.58, 6.22, 1.58, PURPLE, 1.5, True)
    c.rect(6.26, 1.27, 0.64, 0.64, PURPLE, None, radius=0.10)
    c.text(6.26, 1.27, 0.64, 0.64, "W", 22, "FFFFFF", True, "center")
    c.line(6.94, 1.58, 7.22, 1.58, PURPLE, 1.5, True)
    c.rect(7.27, 1.23, 1.48, 0.70, PURPLE_BG, PURPLE, radius=0.08, width=1.0)
    c.text(7.27, 1.23, 1.48, 0.70, "z = Norm(Wh)", 13.2, PURPLE, True, "center")
    for dx, dy, col in [(5.10,1.42,TEAL),(5.46,1.27,ORANGE),(5.52,1.70,TEAL)]:
        c.ellipse(dx, dy, 0.075, col, "FFFFFF", 0.7)
    c.rect(4.92, 2.22, 1.64, 0.78, ORANGE_BG, ORANGE, radius=0.12, width=1.0, dash=True)
    c.ellipse(5.69, 2.59, 0.08, ORANGE, "FFFFFF", 0.7)
    c.rect(7.06, 2.22, 1.64, 0.78, TEAL_BG, TEAL, radius=0.12, width=1.0, dash=True)
    c.ellipse(7.84, 2.59, 0.08, TEAL, "FFFFFF", 0.7)
    c.text(4.88, 3.05, 3.88, 0.27, "learned z separates early and later progress", 11.5, PURPLE, True, "center")
    c.line(4.94, 3.49, 8.42, 3.49, "D8E0E9", 1.0)
    c.text(4.88, 3.63, 3.88, 0.30, "MICo pair fitting · same rollout batch", 13.5, C["ink"], True, "center")
    c.rect(4.88, 4.09, 1.76, 0.69, BLUE_BG, None, radius=0.07)
    c.text(4.88, 4.09, 1.76, 0.69, "current\n‖zᵢ − zⱼ‖", 12.2, NAVY, True, "center")
    c.text(6.65, 4.20, 0.30, 0.35, "→", 16, C["muted"], True, "center")
    c.rect(6.98, 4.09, 1.78, 0.69, PURPLE_BG, None, radius=0.07)
    c.text(6.98, 4.09, 1.78, 0.69, "target\n|Δrᵖ| + γ‖z̄ᵢ⁺ − z̄ⱼ⁺‖", 11.0, PURPLE, True, "center")
    c.text(4.88, 4.91, 3.88, 0.43, "L_head = ( current distance − target distance )²", 12.3, C["ink"], True, "center")
    c.text(4.88, 5.38, 3.88, 0.24, "update W only · EMA target for successors", 11.1, C["muted"], False, "center")

    # Panel 3: one pure cluster, action-conditioned peers, and final advantage.
    c.rect(9.28, 1.19, 3.58, 2.06, "FAFBFC", TEAL, radius=0.12, width=1.1, dash=True)
    c.text(9.42, 1.29, 3.30, 0.29, "one learned BiGPO cluster", 13.2, TEAL, True, "center")
    for x, num, action, ret in [(9.48,"1","a","0.9"),(10.30,"2","a","0.7"),(11.12,"4","a","0.8"),(12.02,"3","b","0.2")]:
        c.ellipse(x+0.17, 1.94, 0.17, TEAL if action == "a" else GRAY, None, 0)
        c.text(x, 1.77, 0.34, 0.34, num, 12, "FFFFFF", True, "center")
        c.text(x, 2.16, 0.34, 0.25, action, 12.5, C["ink"], True, "center")
        c.text(x-0.09, 2.47, 0.52, 0.24, "R="+ret, 10.5, C["muted"], False, "center")
    c.line(9.43, 2.88, 11.53, 2.88, TEAL, 1.2)
    c.text(9.43, 2.91, 2.10, 0.24, "same-action peers", 10.8, TEAL, True, "center")
    c.rect(9.28, 3.45, 1.70, 0.77, TEAL_BG, None, radius=0.07)
    c.text(9.28, 3.45, 1.70, 0.77, "Q̂(s,a) = 0.80\nmean same-action R", 12.0, TEAL, True, "center")
    c.rect(11.16, 3.45, 1.70, 0.77, BLUE_BG, None, radius=0.07)
    c.text(11.16, 3.45, 1.70, 0.77, "V̂(s) = 0.65\nmean all R", 12.0, NAVY, True, "center")
    c.text(9.28, 4.38, 3.58, 0.38, "A_step = Q̂ − V̂ = +0.15", 16, C["ink"], True, "center")
    c.rect(9.28, 4.94, 3.58, 0.67, PURPLE_BG, None, radius=0.08)
    c.text(9.28, 4.94, 3.58, 0.67, "A = A_episode + ω · A_step", 14.5, PURPLE, True, "center")

    # Cross-panel connectors and invariant footer.
    c.line(4.36, 3.02, 4.54, 3.02, GRAY, 1.4, True)
    c.line(8.76, 3.02, 8.94, 3.02, GRAY, 1.4, True)
    c.rect(0.18, 6.16, 12.98, 0.25, "F3F5F8", None, radius=0.05)
    c.text(0.30, 6.10, 4.05, 0.36, "policy prompt unchanged", 11.2, C["muted"], True, "center")
    c.text(4.62, 6.10, 4.05, 0.36, "PPO reward and objective unchanged", 11.2, C["muted"], True, "center")
    c.text(8.94, 6.10, 4.05, 0.36, "no extra rollouts or learned critic", 11.2, C["muted"], True, "center")

    c.save()
    print(PPT); print(PNG); print(PDF)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()
