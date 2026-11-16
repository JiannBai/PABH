#!/usr/bin/env python3
"""Generate a visual, fully editable PPT method overview for PABH."""

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.font_manager import FontProperties
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

OUT_DIR = Path(__file__).resolve().parent / "figs"
PPT_PATH = OUT_DIR / "fig_pabh_method_editable.pptx"
PNG_PATH = OUT_DIR / "fig_pabh_method_preview.png"
PDF_PATH = OUT_DIR / "fig_pabh_method.pdf"
SLIDE_W, SLIDE_H, SCALE = 13.333, 7.5, 150

C = {
    "ink": "17293E", "muted": "65758A", "line": "D8E0E9",
    "panel": "FFFFFF", "soft": "F5F7FA", "white": "FFFFFF",
    "purple": "7157C8", "purple_dark": "543AA9", "purple_light": "F0ECFB",
    "teal": "178F96", "teal_dark": "0E6F76", "teal_light": "E9F7F6",
    "orange": "D9833B", "orange_light": "FFF4E9",
    "navy": "315C81", "navy_light": "EDF3F8",
    "gray": "8996A6", "gray_light": "F1F4F7",
    "blue_light": "DCEEF7", "yellow_light": "FFF1C7",
    "pink_light": "F9E0E0", "green_light": "E3F1E3",
    "node_purple": "DCD2F7",
}


def rgb(value):
    return RGBColor.from_string(value)


class DualCanvas:
    def __init__(self, ppt_path=PPT_PATH, png_path=PNG_PATH, pdf_path=PDF_PATH,
                 width=SLIDE_W, height=SLIDE_H):
        self.ppt_path, self.png_path, self.pdf_path = ppt_path, png_path, pdf_path
        self.width, self.height = width, height
        self.prs = Presentation()
        self.prs.slide_width = Inches(width)
        self.prs.slide_height = Inches(height)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = rgb(C["white"])
        self.image = Image.new("RGB", (round(width*SCALE), round(height*SCALE)), "#FFFFFF")
        self.draw = ImageDraw.Draw(self.image)
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    @staticmethod
    def _font(size, font="Arial"):
        path = ("/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
                if font == "Cambria Math" else
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
        return ImageFont.truetype(path, size=max(8, round(size*SCALE/72)))

    def rect(self, x, y, w, h, fill, line=None, radius=0.10, width=1.0, dash=False):
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        shape = self.slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(width)
            if dash:
                shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        else:
            shape.line.fill.background()
        self.draw.rounded_rectangle(
            [round(x*SCALE), round(y*SCALE), round((x+w)*SCALE), round((y+h)*SCALE)],
            radius=round(radius*SCALE), fill="#"+fill,
            outline="#"+line if line else None, width=max(1, round(width*SCALE/72)))
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor="#"+fill, edgecolor="#"+line if line else "none",
            linewidth=width, linestyle="--" if dash else "-")
        self.ax.add_patch(patch)
        return shape

    def text(self, x, y, w, h, value, size=10, color=None, bold=False,
             align="left", valign="middle", font="Arial", margin=0.02,
             line_spacing=1.0):
        color = color or C["ink"]
        box = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(margin)
        tf.margin_top = tf.margin_bottom = Inches(margin)
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE,
                              "bottom": MSO_ANCHOR.BOTTOM}[valign]
        for idx, line_value in enumerate(value.split("\n")):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = line_value
            p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER,
                           "right": PP_ALIGN.RIGHT}[align]
            p.space_before = p.space_after = Pt(0)
            p.line_spacing = line_spacing
            for run in p.runs:
                run.font.name = font
                run.font.size = Pt(size)
                run.font.bold = bold
                run.font.color.rgb = rgb(color)
        preview_font = "Cambria Math" if any(ord(ch) > 127 for ch in value) else font
        pil_font = self._font(size, preview_font)
        px = (x*SCALE, y*SCALE, (x+w)*SCALE, (y+h)*SCALE)
        spacing = max(1, round(size*SCALE/72*(line_spacing-1.0)))
        bbox = self.draw.multiline_textbbox((0, 0), value, font=pil_font, spacing=spacing, align=align)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = (px[0]+(px[2]-px[0]-tw)/2 if align == "center" else
              px[2]-tw-margin*SCALE if align == "right" else px[0]+margin*SCALE)
        ty = (px[1]+margin*SCALE if valign == "top" else
              px[3]-th-margin*SCALE if valign == "bottom" else
              px[1]+(px[3]-px[1]-th)/2-bbox[1])
        self.draw.multiline_text((round(tx), round(ty)), value, font=pil_font, fill="#"+color,
                                 spacing=spacing, align=align,
                                 stroke_width=1 if bold and size >= 13 else 0,
                                 stroke_fill="#"+color)
        font_path = ("/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
                     if preview_font == "Cambria Math" else
                     "/usr/share/fonts/truetype/freefont/FreeSans.ttf")
        fp = FontProperties(fname=font_path)
        mx = x+w/2 if align == "center" else x+w-margin if align == "right" else x+margin
        my = y+margin if valign == "top" else y+h-margin if valign == "bottom" else y+h/2
        self.ax.text(mx, my, value, fontproperties=fp, fontsize=size,
                     color="#"+color, fontweight="bold" if bold else "normal",
                     ha=align, va={"top": "top", "middle": "center", "bottom": "bottom"}[valign],
                     linespacing=max(1.0, line_spacing))
        return box

    def line(self, x1, y1, x2, y2, color=None, width=1.4, arrow=False, dash=False):
        color = color or C["muted"]
        conn = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                               Inches(x2), Inches(y2))
        conn.line.color.rgb = rgb(color)
        conn.line.width = Pt(width)
        if dash:
            conn.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        if arrow:
            ln = conn._element.spPr.get_or_add_ln()
            tail = OxmlElement("a:tailEnd")
            tail.set("type", "triangle")
            tail.set("w", "sm")
            tail.set("len", "sm")
            ln.append(tail)
        p1, p2 = (x1*SCALE, y1*SCALE), (x2*SCALE, y2*SCALE)
        self.draw.line([p1, p2], fill="#"+color, width=max(1, round(width*SCALE/72)))
        if arrow:
            angle = math.atan2(y2-y1, x2-x1)
            length, spread = 0.10*SCALE, 0.040*SCALE
            bx, by = p2[0]-length*math.cos(angle), p2[1]-length*math.sin(angle)
            nx, ny = spread*math.sin(angle), -spread*math.cos(angle)
            self.draw.polygon([p2, (bx+nx, by+ny), (bx-nx, by-ny)], fill="#"+color)
        self.ax.plot([x1, x2], [y1, y2], color="#"+color, linewidth=width,
                     linestyle="--" if dash else "-", solid_capstyle="round")
        if arrow:
            angle = math.atan2(y2-y1, x2-x1)
            length, spread = 0.10, 0.040
            bx, by = x2-length*math.cos(angle), y2-length*math.sin(angle)
            nx, ny = spread*math.sin(angle), -spread*math.cos(angle)
            self.ax.add_patch(mpatches.Polygon(
                [(x2, y2), (bx+nx, by+ny), (bx-nx, by-ny)],
                closed=True, facecolor="#"+color, edgecolor="none"))
        return conn

    def ellipse(self, cx, cy, r, fill, line=None, width=1.0):
        shape = self.slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx-r), Inches(cy-r),
                                            Inches(2*r), Inches(2*r))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(width)
        else:
            shape.line.fill.background()
        self.draw.ellipse([round((cx-r)*SCALE), round((cy-r)*SCALE),
                           round((cx+r)*SCALE), round((cy+r)*SCALE)],
                          fill="#"+fill, outline="#"+line if line else None,
                          width=max(1, round(width*SCALE/72)))
        self.ax.add_patch(mpatches.Circle(
            (cx, cy), r, facecolor="#"+fill,
            edgecolor="#"+line if line else "none", linewidth=width))
        return shape

    def save(self):
        self.prs.core_properties.title = "PABH Visual Method Overview"
        self.prs.core_properties.subject = "Editable vector schematic for progress-aware bisimulation"
        self.prs.core_properties.author = "BiPACE research project"
        self.prs.save(self.ppt_path)
        matplotlib.rcParams["pdf.fonttype"] = 42
        self.fig.savefig(self.pdf_path, format="pdf", pad_inches=0)
        self.fig.savefig(self.png_path, format="png", dpi=SCALE, pad_inches=0)
        plt.close(self.fig)


def panel(c, x, y, w, h, accent, title, badge, badge_color):
    c.rect(x, y, w, h, C["panel"], C["line"], radius=0.13, width=0.9)
    c.rect(x, y, w, 0.055, accent, None, radius=0)
    c.text(x+0.18, y+0.13, w-0.98, 0.31, title, 12.3, C["ink"], True)
    c.rect(x+w-0.76, y+0.16, 0.57, 0.23, badge_color, None, radius=0.11)
    c.text(x+w-0.76, y+0.16, 0.57, 0.23, badge, 7.0, C["white"], True, "center")


def progress(c, x, y, filled, color, total=3, seg=0.105, gap=0.025):
    for k in range(total):
        c.rect(x+k*(seg+gap), y, seg, 0.065,
               color if k < filled else C["white"],
               None if k < filled else C["line"], radius=0.018, width=0.55)


def network(c, x, y, color=C["navy"]):
    for yy in (y-0.14, y, y+0.14):
        for yy2 in (y-0.09, y+0.09):
            c.line(x, yy, x+0.38, yy2, C["gray"], 0.55)
    for yy in (y-0.14, y, y+0.14):
        c.ellipse(x, yy, 0.045, C["white"], color, 0.8)
    for yy in (y-0.09, y+0.09):
        c.ellipse(x+0.38, yy, 0.055, color, C["white"], 0.6)


def return_bars(c, xs, baseline, heights, colors, width=0.13):
    for x, h, color in zip(xs, heights, colors):
        c.rect(x-width/2, baseline-h, width, h, color, None, radius=0.025)


def build():
    c = DualCanvas()

    # Header
    c.text(0.29, 0.17, 10.9, 0.40,
           "PABH: Progress-Aware Bisimulation for Step-Level Credit Assignment",
           21.5, C["ink"], True)
    c.text(0.31, 0.62, 10.3, 0.25,
           "Progress-aware states  →  learned behavioral peers  →  action-conditioned advantages",
           10.1, C["muted"])
    c.rect(11.13, 0.27, 1.75, 0.38, C["purple_light"], None, radius=0.12)
    c.text(11.13, 0.27, 1.75, 0.38, "same rollouts as BiPACE", 7.7, C["purple_dark"], True, "center")

    panel(c, 0.25, 1.00, 3.20, 5.84, C["teal"], "1  Rollouts + progress state", "OURS", C["teal"])
    panel(c, 3.57, 1.00, 5.47, 5.84, C["purple"], "2  Train the state metric", "OURS", C["purple"])
    panel(c, 9.16, 1.00, 3.92, 5.84, C["navy"], "3  Cluster + action credit", "BiPACE", C["navy"])

    # --- Panel 1: GiGPO-style rollout records with progress markers ---
    c.rect(0.45, 1.60, 2.72, 0.25, C["navy_light"], None, radius=0.09)
    c.text(0.45, 1.60, 2.72, 0.25, "Task: place two kettles on the table", 7.4, C["navy"], True, "center")

    lane_colors = [C["blue_light"], C["yellow_light"], C["pink_light"], C["green_light"]]
    lane_ys = [1.96, 2.64, 3.32]
    highlights = {(0, 2): (2, C["teal"]), (1, 1): (1, C["orange"])}
    for i, ly in enumerate(lane_ys):
        c.rect(0.43, ly, 2.83, 0.52, C["soft"], None, radius=0.08)
        c.text(0.47, ly+0.12, 0.32, 0.25, f"τ{i+1}", 8.3, C["muted"], True, "center")
        xs = [0.92, 1.44, 1.96, 2.48]
        for k, x in enumerate(xs):
            is_highlight = (i, k) in highlights
            fill = C["node_purple"] if is_highlight else lane_colors[(i+k) % 4]
            c.rect(x, ly+0.07, 0.34, 0.30, fill,
                   C["purple"] if is_highlight else C["white"], radius=0.06, width=0.85)
            c.text(x, ly+0.07, 0.34, 0.30, f"s{k+1}", 7.2,
                   C["purple_dark"] if is_highlight else C["ink"], is_highlight, "center")
            if k < 3:
                c.line(x+0.34, ly+0.22, x+0.50, ly+0.22, C["gray"], 0.75, True)
            if is_highlight:
                filled, color = highlights[(i, k)]
                progress(c, x+0.015, ly+0.41, filled, color)
        c.text(2.86, ly+0.12, 0.18, 0.24, "…", 9.0, C["gray"], False, "center")
        c.text(3.04, ly+0.12, 0.20, 0.24, f"R{i+1}", 7.0, C["navy"], True, "center")

    c.line(2.13, 2.40, 1.61, 2.69, C["purple"], 1.0, False, True)
    c.text(2.16, 2.45, 0.58, 0.17, "same obs", 6.8, C["purple_dark"])
    c.text(0.46, 3.96, 2.76, 0.24, "Same observation, different progress", 9.2, C["ink"], True, "center")

    # Concrete ALFWorld-style example: identical feedback, different completed subgoals.
    for x, filled, color, glabel in [(0.45, 2, C["teal"], "g₂={take(k1),\nput(k2)}"),
                                     (1.96, 1, C["orange"], "g₁={take(k1)}")]:
        c.rect(x, 4.26, 1.28, 0.79, C["white"], color, radius=0.09, width=1.0)
        c.rect(x+0.08, 4.34, 1.12, 0.22, C["node_purple"], C["purple"], radius=0.05, width=0.7)
        c.text(x+0.08, 4.34, 1.12, 0.22, "\"pick up kettle 1\"", 6.5, C["purple_dark"], True, "center")
        progress(c, x+0.15, 4.61, filled, color, seg=0.13, gap=0.04)
        c.text(x+0.08, 4.75, 1.12, 0.25, glabel, 6.5, color, True, "center")
    c.text(1.72, 4.43, 0.25, 0.30, "≠", 15.0, C["muted"], True, "center")
    c.text(0.48, 5.10, 2.74, 0.22, "state = observation + completed subgoals", 8.0, C["teal_dark"], True, "center")

    # Anchor-only composition and actor hidden extraction.
    c.rect(0.45, 5.43, 2.80, 0.93, C["navy_light"], C["line"], radius=0.10, width=0.8)
    c.rect(0.58, 5.65, 0.85, 0.34, C["purple_light"], None, radius=0.07)
    c.text(0.58, 5.65, 0.85, 0.34, "[g_text ; o]", 8.8, C["purple_dark"], True, "center")
    c.text(0.58, 6.04, 0.85, 0.17, "anchor only", 6.8, C["muted"], False, "center")
    c.line(1.47, 5.87, 1.70, 5.87, C["navy"], 1.1, True)
    network(c, 1.86, 5.87)
    c.text(1.66, 6.13, 0.80, 0.16, "late-layer actor", 6.6, C["muted"], False, "center")
    c.line(2.30, 5.87, 2.57, 5.87, C["navy"], 1.1, True)
    c.ellipse(2.79, 5.87, 0.19, C["navy"], C["white"], 0.7)
    c.text(2.60, 5.68, 0.38, 0.38, "h", 11.0, C["white"], True, "center")
    c.text(0.50, 6.46, 2.70, 0.18, "policy prompt remains unchanged", 7.0, C["muted"], False, "center")

    # h and PRM paths into the metric panel.
    c.line(3.25, 5.87, 3.43, 5.87, C["navy"], 1.2)
    c.line(3.43, 5.87, 3.43, 2.80, C["navy"], 1.2)
    c.line(3.43, 2.80, 3.82, 2.80, C["navy"], 1.2, True)
    c.line(3.25, 4.67, 3.50, 4.67, C["orange"], 1.1)
    c.line(3.50, 4.67, 3.50, 5.98, C["orange"], 1.1)
    c.line(3.50, 5.98, 4.02, 5.98, C["orange"], 1.1, True)
    c.text(3.23, 4.43, 0.25, 0.20, "rᵖ", 8.0, C["orange"], True, "center")

    # --- Panel 2: raw versus learned embedding geometry ---
    c.rect(3.82, 1.66, 1.74, 2.39, C["soft"], C["line"], radius=0.10, width=0.8)
    c.text(3.94, 1.76, 1.50, 0.24, "raw hidden  h", 9.5, C["navy"], True, "center")
    c.text(3.94, 2.03, 1.50, 0.20, "progress-confused", 7.4, C["muted"], False, "center")
    c.rect(3.99, 2.39, 1.40, 1.22, C["white"], C["gray"], radius=0.35, width=1.0, dash=True)
    for cx, cy, color in [(4.28, 2.72, C["teal"]), (4.69, 2.65, C["orange"]),
                          (5.04, 2.96, C["teal"]), (4.39, 3.23, C["orange"]),
                          (4.84, 3.34, C["teal"]), (4.62, 2.99, C["orange"])]:
        c.ellipse(cx, cy, 0.095, color, C["white"], 0.7)
    c.text(4.00, 3.70, 1.38, 0.18, "one mixed raw cluster", 7.0, C["muted"], False, "center")

    c.line(5.56, 2.82, 5.78, 2.82, C["purple"], 1.5, True)
    c.rect(5.80, 2.47, 0.84, 0.66, C["purple"], None, radius=0.15)
    c.text(5.80, 2.47, 0.84, 0.66, "W", 18.0, C["white"], True, "center")
    c.text(5.67, 3.19, 1.10, 0.20, "linear head", 7.5, C["purple_dark"], True, "center")
    c.text(5.59, 3.43, 1.26, 0.22, "z = Norm(Wh)", 8.8, C["purple_dark"], False, "center", font="Cambria Math")
    c.rect(5.76, 3.72, 0.93, 0.24, C["purple_light"], None, radius=0.08)
    c.text(5.76, 3.72, 0.93, 0.24, "gradient: W only", 6.7, C["purple_dark"], True, "center")
    c.line(6.64, 2.82, 6.93, 2.82, C["purple"], 1.5, True)

    c.rect(6.95, 1.66, 1.81, 2.39, C["purple_light"], C["purple"], radius=0.10, width=0.9)
    c.text(7.07, 1.76, 1.57, 0.24, "learned metric  z", 9.5, C["purple_dark"], True, "center")
    c.text(7.07, 2.03, 1.57, 0.20, "progress-pure peers", 7.4, C["teal_dark"], False, "center")
    c.rect(7.11, 2.32, 1.48, 0.67, C["teal_light"], C["teal"], radius=0.25, width=0.9, dash=True)
    for cx, cy in [(7.38, 2.65), (7.72, 2.54), (8.03, 2.72)]:
        c.ellipse(cx, cy, 0.095, C["teal"], C["white"], 0.7)
    c.text(8.16, 2.48, 0.34, 0.22, "late", 6.8, C["teal_dark"], True, "center")
    c.rect(7.11, 3.18, 1.48, 0.67, C["orange_light"], C["orange"], radius=0.25, width=0.9, dash=True)
    for cx, cy in [(7.38, 3.51), (7.72, 3.40), (8.03, 3.58)]:
        c.ellipse(cx, cy, 0.095, C["orange"], C["white"], 0.7)
    c.text(8.12, 3.35, 0.42, 0.22, "early", 6.8, C["orange"], True, "center")

    c.ellipse(4.23, 4.20, 0.055, C["teal"], None)
    c.text(4.31, 4.10, 1.13, 0.20, "later progress", 7.0, C["muted"])
    c.ellipse(5.66, 4.20, 0.055, C["orange"], None)
    c.text(5.74, 4.10, 1.27, 0.20, "earlier progress", 7.0, C["muted"])
    c.text(7.22, 4.10, 1.38, 0.20, "dot color = g", 7.0, C["muted"], False, "right")

    # --- MICo as a visual current-versus-target distance fit ---
    c.line(3.83, 4.39, 8.76, 4.39, C["line"], 0.8)
    c.text(3.86, 4.45, 4.86, 0.22, "MICo pair fitting · same rollout batch", 8.8, C["ink"], True, "center")
    c.text(4.03, 4.72, 1.94, 0.19, "current distance", 7.2, C["purple_dark"], True, "center")
    c.ellipse(4.35, 5.08, 0.16, C["teal"], C["white"], 0.7)
    c.text(4.19, 4.92, 0.32, 0.32, "i", 9.0, C["white"], True, "center")
    c.ellipse(5.55, 5.08, 0.16, C["orange"], C["white"], 0.7)
    c.text(5.39, 4.92, 0.32, 0.32, "j", 9.0, C["white"], True, "center")
    c.line(4.51, 5.08, 5.39, 5.08, C["purple"], 2.0)
    c.text(4.00, 5.30, 0.70, 0.20, "rᵢᵖ = +0.5", 7.4, C["orange"], False, "center")
    c.text(5.20, 5.30, 0.70, 0.20, "rⱼᵖ = −0.1", 7.4, C["orange"], False, "center")
    c.line(5.88, 5.08, 6.60, 5.08, C["gray"], 1.0, True)
    c.text(5.93, 4.79, 0.62, 0.18, "next", 6.8, C["muted"], False, "center")

    c.text(6.70, 4.72, 1.90, 0.19, "successors via EMA  W̄", 7.2, C["purple_dark"], True, "center")
    c.ellipse(7.02, 5.08, 0.16, C["white"], C["teal"], 1.1)
    c.text(6.82, 4.92, 0.40, 0.32, "i⁺", 8.6, C["teal_dark"], True, "center")
    c.ellipse(8.20, 5.08, 0.16, C["white"], C["orange"], 1.1)
    c.text(8.00, 4.92, 0.40, 0.32, "j⁺", 8.6, C["orange"], True, "center")
    c.line(7.18, 5.08, 8.04, 5.08, C["purple"], 1.5, False, True)

    c.text(3.98, 5.63, 1.12, 0.18, "target distance", 7.0, C["muted"], True)
    c.rect(4.02, 5.88, 1.54, 0.30, C["orange"], None, radius=0.06)
    c.text(4.02, 5.88, 1.54, 0.30, "|Δrᵖ| = 0.6", 7.4, C["white"], True, "center")
    c.rect(5.56, 5.88, 3.06, 0.30, C["purple"], None, radius=0.06)
    c.text(5.56, 5.88, 3.06, 0.30, "γ × successor distance", 7.0, C["white"], True, "center")
    c.line(4.95, 5.24, 4.95, 5.80, C["purple"], 1.0, True)
    c.text(5.03, 5.47, 0.35, 0.18, "fit", 6.8, C["purple_dark"], True)
    c.text(3.91, 6.27, 4.82, 0.27,
           "L = ( ‖zᵢ−zⱼ‖ − [ |Δrᵖ| + γ‖z̄ᵢ⁺−z̄ⱼ⁺‖ ] )²",
           10.0, C["purple_dark"], True, "center", font="Cambria Math")

    # z-space clusters feed the right-hand credit estimator.
    c.line(8.76, 2.82, 9.40, 2.82, C["purple"], 1.6, True)
    c.text(8.78, 2.52, 0.59, 0.20, "cosine ≤ ε", 6.7, C["purple_dark"], False, "center")

    # --- Panel 3: one purified cluster and action-conditioned return means ---
    c.rect(9.42, 1.67, 3.39, 1.77, C["teal_light"], C["teal"], radius=0.13, width=1.0, dash=True)
    c.text(9.58, 1.77, 3.07, 0.22, "one BiGPO cluster  C", 9.0, C["teal_dark"], True, "center")
    c.text(12.12, 1.80, 0.55, 0.16, "illustrative", 5.5, C["muted"], False, "right")
    node_xs = [9.88, 10.64, 11.40, 12.36]
    node_nums = ["1", "2", "4", "3"]
    actions = ["a", "a", "a", "b"]
    heights = [0.23, 0.34, 0.27, 0.12]
    returns = [0.9, 0.7, 0.8, 0.2]
    for x, num, action, h, ret in zip(node_xs, node_nums, actions, heights, returns):
        c.ellipse(x, 2.31, 0.17, C["teal"], C["white"], 0.7)
        c.text(x-0.17, 2.14, 0.34, 0.34, num, 8.8, C["white"], True, "center")
        c.rect(x-0.15, 2.59, 0.30, 0.23,
               C["navy"] if action == "a" else C["gray"], None, radius=0.06)
        c.text(x-0.15, 2.59, 0.30, 0.23, action, 7.4, C["white"], True, "center")
        c.rect(x-0.065, 3.23-h, 0.13, h, C["navy"] if action == "a" else C["gray"], None, radius=0.025)
        c.text(x-0.18, 3.22, 0.36, 0.13, f"{ret:.1f}", 6.0, C["muted"], False, "center")
    c.text(9.50, 3.15, 0.28, 0.18, "Rₜ", 7.2, C["muted"], False, "center")
    c.line(9.69, 3.35, 11.59, 3.35, C["purple"], 1.0)
    c.line(9.69, 3.30, 9.69, 3.35, C["purple"], 1.0)
    c.line(11.59, 3.30, 11.59, 3.35, C["purple"], 1.0)

    c.line(10.18, 3.44, 10.18, 3.82, C["purple"], 1.2, True)
    c.line(12.00, 3.44, 12.00, 3.82, C["navy"], 1.2, True)
    c.rect(9.43, 3.85, 1.52, 1.36, C["purple_light"], C["purple"], radius=0.10, width=0.9)
    c.text(9.52, 3.94, 1.34, 0.25, "Q̂(s,a) = 0.80", 10.1, C["purple_dark"], True, "center", font="Cambria Math")
    c.text(9.49, 4.20, 1.40, 0.18, "mean{0.9, 0.7, 0.8}", 6.2, C["muted"], False, "center")
    return_bars(c, [9.77, 10.18, 10.59], 4.93, heights[:3],
                [C["purple"]]*3, width=0.15)
    c.line(9.64, 4.64, 10.73, 4.64, C["purple_dark"], 1.0, False, True)

    c.rect(11.19, 3.85, 1.61, 1.36, C["navy_light"], C["navy"], radius=0.10, width=0.9)
    c.text(11.29, 3.94, 1.41, 0.25, "V̂(s) = 0.65", 10.1, C["navy"], True, "center", font="Cambria Math")
    c.text(11.23, 4.20, 1.53, 0.18, "mean{0.9, 0.7, 0.8, 0.2}", 5.9, C["muted"], False, "center")
    return_bars(c, [11.43, 11.78, 12.13, 12.48], 4.93,
                [heights[0], heights[1], heights[2], heights[3]],
                [C["navy"], C["navy"], C["navy"], C["gray"]], width=0.13)
    c.line(11.32, 4.68, 12.66, 4.68, C["navy"], 1.0, False, True)

    c.text(9.47, 5.27, 3.29, 0.48,
           "A_step = Q̂(s,a) − V̂(s)\n= 0.80 − 0.65 = +0.15",
           10.8, C["navy"], True, "center", font="Cambria Math")
    c.rect(9.43, 5.82, 3.37, 0.60, C["soft"], C["line"], radius=0.10, width=0.8)
    c.text(9.57, 5.88, 3.09, 0.28, "A  =  A_episode  +  ω · A_step",
           11.1, C["ink"], True, "center", font="Cambria Math")
    c.text(9.57, 6.18, 3.09, 0.16, "extrinsic discounted returns", 6.9, C["muted"], False, "center")

    # Unchanged PPO loop and the reward boundary.
    c.rect(0.25, 6.97, 12.83, 0.39, C["soft"], C["line"], radius=0.10, width=0.8)
    c.text(0.47, 7.02, 2.15, 0.28, "PPO update  (unchanged)", 9.1, C["ink"], True)
    c.ellipse(3.04, 7.16, 0.13, C["navy"], None)
    c.text(2.91, 7.03, 0.26, 0.26, "A", 8.0, C["white"], True, "center")
    c.line(3.21, 7.16, 3.58, 7.16, C["navy"], 1.0, True)
    network(c, 3.79, 7.16)
    c.text(4.26, 7.03, 0.43, 0.26, "πθ", 9.4, C["navy"], True, "center")
    c.line(4.72, 7.16, 5.10, 7.16, C["gray"], 1.0, True)
    c.text(5.20, 7.03, 1.90, 0.26, "next on-policy rollouts", 8.0, C["muted"])
    c.rect(8.47, 7.02, 4.38, 0.28, C["orange_light"], None, radius=0.08)
    c.text(8.52, 7.02, 4.28, 0.28,
           "PRM shapes the metric only  |  PPO uses extrinsic rewards",
           7.6, C["orange"], True, "center")

    c.save()
    print(PPT_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build()
