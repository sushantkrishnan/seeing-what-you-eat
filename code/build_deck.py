"""Build the COMPSCI 760 Group 1 proposal deck.

  ../.venv/bin/python build_deck.py        -> 01_Proposal_v6.pptx

Run gate_bench.py, gate_probe.py --all --json and figures.py first. This script
refuses to build if the figures are missing.

FRAMING. The predictor is frozen and is not the contribution. The contribution is a
benchmark of the signals an app could use to decide whether to answer at all. Slide 3
states that as a question we picked after reading the literature, not as a gap nobody
has noticed. See CLAUDE.md for the measured results behind every number here.

V6 (17 Aug 2026), a rebuild rather than a patch:
  * figure-led. Nine images across six slides: the plate grid and dataset scatter,
    the crop-vs-blur strip, the reject curve, the backbone comparison, the gate
    benchmark, the degeneracy scatter, and the two damage-ladder strips;
  * motivation and problem are framed as landscape -> question -> contribution;
  * specifics are named. Three backbones with numbers, eight gates with numbers,
    rather than "nine candidate signals";
  * slide 9 is new and carries the result that matters most: model-aware signals do
    work, and a trivial control exposes how a gate can win by refusing big meals;
  * house style is plain. Short declarative sentences. No aphorisms, no "not X but Y"
    inversions, few dashes. Keep it that way when editing.

Speaker names are not printed on slides. SLIDE_PLAN still assigns them because it
drives the timing budget and the notes; per-member roles are on slide 11.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"

FONT = "Helvetica Neue"
INK = RGBColor(0x1B, 0x1E, 0x28)
MUTED = RGBColor(0x6B, 0x71, 0x80)
ACCENT = RGBColor(0xC4, 0x5A, 0x1B)
GREEN = RGBColor(0x0E, 0x7C, 0x5A)
BLUE = RGBColor(0x2F, 0x5F, 0xA8)
BG = RGBColor(0xFA, 0xF9, 0xF7)
TINT = RGBColor(0xFA, 0xEE, 0xE4)
RULE = RGBColor(0xE0, 0xDA, 0xD2)
FAINT = RGBColor(0xC0, 0xBA, 0xB2)

W, H = 13.333, 7.5
L = 0.95
CW = 11.5
R = L + CW

# Speaker, target seconds. Sums to 550s = 9:10, leaving room for four handoffs
# inside the 9-10 minute band the rubric pays for.
SLIDE_PLAN = [
    ("Nisarg",  40, "Clarity: motivation"),
    ("Nisarg",  45, "Clarity: problem vs proposal"),
    ("Nisarg",  45, "Clarity: SMART objectives"),
    ("Sushant", 60, "Literature background"),
    ("Sushant", 45, "Methodology: dataset"),
    ("Keanu",   50, "Methodology: the predictor"),
    ("Keanu",   65, "Preliminary result 1"),
    ("Shreya",  65, "Preliminary result 2"),
    ("Shreya",  55, "Methodology: the experiment"),
    ("Kevin",   45, "Planning: timeline + roles"),
    ("Kevin",   35, "Planning: risks"),
]

REQUIRED_FIGS = ["panel_clean.png", "panel_blur6.png", "panel_crop04.png",
                 "risk_coverage.png", "dataset_scatter.png", "plate_grid.png",
                 "ladder_visible.png", "ladder_invisible.png", "backbones.png",
                 "gates.png", "degeneracy.png"]


# ---------------------------------------------------------------- primitives
def tb(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf


def para(tf, runs, *, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT,
         space_before=0, line=1.25, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.line_spacing = line
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, over in runs:
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = FONT
        f.size = Pt(over.get("size", size))
        f.bold = over.get("bold", bold)
        f.color.rgb = over.get("color", color)
    return p


def rect(slide, x, y, w, h, fill=BG):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.text_frame.text = ""
    return sh


def arrow(slide, x, y, w, h, *, down=False, color=FAINT):
    sh = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW if down else MSO_SHAPE.RIGHT_ARROW,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.text_frame.text = ""
    return sh


def hrule(slide, x, y, w, color=RULE):
    rect(slide, x, y, w, 1 / 72.0, fill=color)


def pic(slide, name, x, y, w):
    return slide.shapes.add_picture(str(FIGS / name), Inches(x), Inches(y), width=Inches(w))


def eyebrow(slide, x, y, text, color=ACCENT, w=6.0):
    t = tb(slide, x, y, w, 0.3)
    para(t, text.upper(), size=11.5, color=color, bold=True, first=True)


def caption(slide, x, y, w, text, size=12, color=MUTED, bold=False):
    t = tb(slide, x, y, w, 0.7)
    para(t, text, size=size, color=color, bold=bold, line=1.32, first=True)


def base(prs, brow, headline, idx):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    eyebrow(slide, L, 0.50, brow, w=9.0)

    lines = headline.split("\n")
    t = tb(slide, L, 0.80, CW, 1.05)
    for i, ln in enumerate(lines):
        para(t, ln, size=32, color=INK, bold=True, line=1.12, first=(i == 0))

    rule_y = 0.80 + 0.52 * len(lines) + 0.16
    rect(slide, L, rule_y, 1.40, 0.037, fill=ACCENT)

    t = tb(slide, R - 1.0, H - 0.55, 1.0, 0.3)
    para(t, str(idx), size=10.5, color=FAINT, align=PP_ALIGN.RIGHT, first=True)

    slide._body_top = rule_y + 0.30
    return slide


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def takeaway(slide, text, y, size=15):
    t = tb(slide, L, y, CW, 0.8)
    para(t, text, size=size, color=INK, bold=True, line=1.3, first=True)


def table_head(slide, y, cols, widths, heads):
    for cx, cwd, htxt in zip(cols, widths, heads):
        t = tb(slide, cx, y, cwd, 0.3)
        para(t, htxt, size=11.5, color=MUTED, bold=True, first=True)
    hrule(slide, L, y + 0.30, CW, color=FAINT)
    return y + 0.42


# ---------------------------------------------------------------- slides
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, fill=BG)
    rect(s, 0, 0, 0.16, H, fill=ACCENT)

    t = tb(s, L + 0.25, 2.30, 11.0, 1.2)
    para(t, "Seeing what you eat", size=52, color=INK, bold=True, first=True)
    t = tb(s, L + 0.25, 3.45, 11.0, 0.8)
    para(t, "When should a nutrition app refuse to answer?", size=23, color=ACCENT,
         first=True)
    rect(s, L + 0.25, 4.60, 1.40, 0.037, fill=FAINT)

    t = tb(s, L + 0.25, 5.00, 11.5, 1.0)
    para(t, "COMPSCI 760  ·  Group 1  ·  Research Project Proposal",
         size=14, color=MUTED, bold=True, first=True)
    para(t, "Sushant Krishnan (sthi106)  ·  Keanu De Cleene (kdec819)  ·  "
            "Nisarg Patel (npat235)", size=14, color=MUTED, space_before=8)
    para(t, "Shreya Tigga (stig804)  ·  Kevin Zou (rzou895)", size=14, color=MUTED,
         space_before=4)
    notes(s, "Title only. Do not speak to this slide. Nisarg opens on slide 2. "
             "Target 9:10, hard limit 10:00.")
    return s


def slide_motivation(prs):
    s = base(prs, "Motivation", '“620 calories.”\nShould you believe it?', 2)
    y = s._body_top

    t = tb(s, L, y, 6.1, 2.6)
    para(t, "Photo-based calorie apps are already used for real health decisions: "
            "diabetes management, clinical diet tracking, weight programmes.",
         size=16, color=MUTED, line=1.45, first=True)
    para(t, "They answer every photo. They never give a range and they never decline.",
         size=16, color=INK, bold=True, space_before=16, line=1.45)
    para(t, "A confident wrong number is worse than no number.",
         size=16, color=ACCENT, bold=True, space_before=16, line=1.45)

    rect(s, 7.45, y, 5.0, 1.15, fill=BG)
    t = tb(s, 7.75, y + 0.36, 4.4, 0.5)
    para(t, "What it says:   620 kcal", size=17, color=MUTED, first=True)

    rect(s, 7.45, y + 1.45, 5.0, 1.95, fill=TINT)
    t = tb(s, 7.75, y + 1.70, 4.5, 1.5)
    para(t, "What it should say:", size=13.5, color=ACCENT, bold=True, first=True)
    para(t, "“480–760, nine times in ten.”\nor\n“I can’t tell from this photo.”",
         size=15, color=INK, bold=True, space_before=8, line=1.32)

    takeaway(s, "To say either of those, an app has to know when it is guessing. "
                "We wanted to find out whether it can.", y=5.95)
    notes(s, "NISARG — 40s.\nOpen with the number, not the method. The point to land is that a "
             "wrong number which looks confident is worse than a refusal, because someone acts "
             "on it.\nThe last line is the whole project. Say it, pause, then move on. Do not "
             "say 'good photo versus bad photo'. Our own results say photo quality is not the "
             "axis, and slide 8 shows why.")
    return s


def slide_problem(prs):
    s = base(prs, "Problem statement", "Everyone is making the number smaller.\n"
                                       "We want to ask if you can trust it.", 3)
    y = s._body_top

    t = tb(s, L, y, CW, 0.6)
    para(t, "We read the food-estimation literature looking for somewhere useful to "
            "contribute. Almost all of it reports one number per photo and competes on how "
            "small that number is. Very little of it asks whether the number can be trusted. "
            "That is the question we picked.",
         size=13.5, color=MUTED, line=1.35, first=True)

    y += 0.76
    bh = 0.74
    rect(s, L, y, 1.55, bh, fill=BG)
    t = tb(s, L, y + 0.22, 1.55, 0.4)
    para(t, "a photo", size=13.5, color=MUTED, align=PP_ALIGN.CENTER, first=True)
    arrow(s, 2.62, y + bh / 2 - 0.09, 0.32, 0.18)

    rect(s, 3.02, y, 3.05, bh, fill=TINT)
    t = tb(s, 3.02, y + 0.13, 3.05, 0.6)
    para(t, "THE GATE", size=12, color=ACCENT, bold=True, align=PP_ALIGN.CENTER, first=True)
    para(t, "answer, or refuse?", size=14, color=INK, bold=True,
         align=PP_ALIGN.CENTER, space_before=3)

    arrow(s, 6.20, y + bh / 2 - 0.09, 0.32, 0.18)
    rect(s, 6.60, y, 2.45, bh, fill=BG)
    t = tb(s, 6.60, y + 0.12, 2.45, 0.6)
    para(t, "the predictor", size=13.5, color=MUTED, align=PP_ALIGN.CENTER, first=True)
    para(t, "frozen", size=11, color=FAINT, align=PP_ALIGN.CENTER, space_before=2)

    arrow(s, 9.18, y + bh / 2 - 0.09, 0.32, 0.18)
    rect(s, 9.58, y, 2.42, bh, fill=BG)
    t = tb(s, 9.58, y + 0.22, 2.42, 0.4)
    para(t, "“620 kcal”", size=14, color=INK, bold=True, align=PP_ALIGN.CENTER, first=True)

    arrow(s, 4.46, y + bh + 0.05, 0.18, 0.26, down=True)
    rect(s, 3.02, y + bh + 0.38, 4.6, 0.56, fill=BG)
    t = tb(s, 3.02, y + bh + 0.51, 4.6, 0.4)
    para(t, "“I can’t tell from this photo.”", size=13.5, color=INK, bold=True,
         align=PP_ALIGN.CENTER, first=True)

    yy = y + bh + 1.12
    for label, body, col in [
        ("THE PROBLEM TODAY",
         "Deployed models answer every photo with a single number, and are tested mostly on "
         "clean lab images. There is no way to decline and no stated confidence.", MUTED),
        ("WHAT WE PROPOSE",
         "Freeze the predictor and study the gate: the part that decides whether to answer. "
         "Eight candidate signals, one protocol, with refusing at random as the floor and a "
         "perfect refuser as the ceiling.", INK),
    ]:
        t = tb(s, L, yy, CW, 0.30)
        para(t, label, size=11.5, color=ACCENT if col is INK else MUTED, bold=True, first=True)
        t = tb(s, L, yy + 0.30, CW, 0.70)
        para(t, body, size=13.5, color=col, bold=(col is INK), line=1.35, first=True)
        yy += 1.02

    notes(s, "NISARG — 45s.\nRead the opening line close to verbatim. It sets the register for "
             "the whole talk: we surveyed a field and picked a question, we are not claiming "
             "nobody has thought about this. If a marker names prior work we have missed, the "
             "right answer is 'thank you, we will read it'.\n"
             "Then walk the diagram once, left to right, and point at the branch. That is the "
             "only time anyone has to explain the word 'gate'.\n"
             "Say the words 'the problem today is' and 'what we propose is'. The rubric wants "
             "them separated. Do not describe method here; Shreya has slides 9 and 10.")
    return s


def slide_objectives(prs):
    s = base(prs, "Research objectives", "Three objectives, each with a number\n"
                                         "attached to it.", 4)
    y = s._body_top
    cols = [L, L + 4.55, L + 8.25, L + 10.55]
    widths = [4.35, 3.55, 2.15, 1.45]
    ry = table_head(s, y, cols, widths,
                    ["Objective", "How we measure it", "Success criterion", "By"])

    rows = [
        ("O1 · Benchmark",
         "Which signals predict when\nthe answer will be wrong?",
         "How well each ranks the errors, on\ndamaged photos and on clean ones",
         "All eight ranked, against\nrandom and perfect", "11 Sep"),
        ("O2 · Ceiling",
         "How much is left on the table,\nand on which axis?",
         "Refusing splits two ways: a bad photo,\nor a hard dish shot well",
         "A gap in kcal for each,\nwith error bars", "20 Sep"),
        ("O3 · Ranges",
         "Does the best signal give\nranges worth having?",
         "Are the ranges honest for good and\nbad photos alike; share within ±100 kcal",
         "Honesty gap ≤ 5 points;\nusable share reported", "11 Oct"),
    ]
    for i, (name, question, measure, target, when) in enumerate(rows):
        rh = 1.16
        if i % 2 == 0:
            rect(s, L - 0.18, ry - 0.10, CW + 0.36, rh, fill=BG)
        t = tb(s, cols[0], ry, widths[0], 1.0)
        para(t, name, size=14.5, color=ACCENT, bold=True, first=True)
        para(t, question, size=13, color=INK, space_before=5, line=1.28)
        t = tb(s, cols[1], ry + 0.04, widths[1], 1.0)
        para(t, measure, size=12.5, color=MUTED, line=1.32, first=True)
        t = tb(s, cols[2], ry + 0.04, widths[2], 1.0)
        para(t, target, size=12.5, color=INK, bold=True, line=1.32, first=True)
        t = tb(s, cols[3], ry + 0.04, widths[3], 1.0)
        para(t, when, size=13, color=MUTED, bold=True, line=1.32, first=True)
        ry += rh

    takeaway(s, "All three are read off one results file. That file already exists.",
             y=ry + 0.12, size=14.5)
    notes(s, "NISARG — 45s.\nDo not read the table. Say the three questions out loud, then "
             "point at the success criterion column and say that is what makes them measurable "
             "rather than aspirational.\nO2 is the one that changed after our first results. "
             "Refusing can fail because the photo is bad or because the dish is hard, and "
             "those turned out to be very different sizes. Hand to Sushant.")
    return s


def slide_literature(prs):
    s = base(prs, "Background", "Two papers we build on,\ntwo protocols we adopt.", 5)
    y = s._body_top
    cols = [L, L + 3.05, L + 7.15]
    widths = [2.85, 3.90, 4.35]
    ry = table_head(s, y, cols, widths, ["Paper", "What it did", "What it leaves open"])

    for i, (paper, did, open_) in enumerate([
        ("[1] Thames et al.\nNutrition5k, CVPR 2021",
         "5,006 lab-weighed dishes. End-to-end\nRGB → calories. 70.6 kcal MAE (RGB),\n"
         "41.3 kcal (best depth model).",
         "One number per photo, on clean lab\nimages, with no statement of confidence\n"
         "and no way to decline."),
        ("[2] Coburn et al.\nIEEE BHI 2025",
         "Benchmarks 8 large multimodal models\non nutrition; metadata and prompting\n"
         "strategies reduce MAE / MAPE.",
         "Reports MAE / MAPE only.\nNo calibration, no ranges,\nno option to abstain."),
    ]):
        if i % 2 == 0:
            rect(s, L - 0.18, ry - 0.11, CW + 0.36, 1.02, fill=BG)
        t = tb(s, cols[0], ry, widths[0], 1.0)
        para(t, paper, size=12.5, color=INK, bold=True, line=1.30, first=True)
        t = tb(s, cols[1], ry, widths[1], 1.0)
        para(t, did, size=12, color=MUTED, line=1.32, first=True)
        t = tb(s, cols[2], ry, widths[2], 1.0)
        para(t, open_, size=12, color=MUTED, line=1.32, first=True)
        ry += 1.02

    ry += 0.10
    eyebrow(s, L, ry, "We take one improvement from each")
    ry += 0.30
    for lbl, txt in [
        ("From [1], its evaluation protocol, used as published: ",
         "we keep the official split, which holds repeated scans of a plate together. A 2026 "
         "paper reporting 28 kcal on this dataset re-split it 60/15/25, so we do not think "
         "that number is comparable to ours."),
        ("From [2], its experimental design, inverted: ",
         "hold the model fixed and vary the photograph, then report whether the answer can be "
         "trusted rather than only how wrong it is."),
    ]:
        t = tb(s, L, ry, CW, 0.52)
        para(t, [(lbl, {"bold": True, "color": INK}), (txt, {"color": MUTED})],
             size=12.5, line=1.30, first=True)
        ry += 0.50

    ry += 0.08
    rect(s, L - 0.18, ry, CW + 0.36, 0.70, fill=TINT)
    t = tb(s, L + 0.14, ry + 0.13, CW - 0.3, 0.55)
    para(t, [("AND TWO PROTOCOLS WE ADOPT RATHER THAN REINVENT   ",
              {"color": ACCENT, "bold": True, "size": 11.5}),
             ("refuse the worst x% and measure what is left [3], the biometrics standard "
              "since 2007  ·  damage families × severities [4], the ImageNet-C ladder design",
              {"color": INK, "size": 12.5})], first=True)

    notes(s, "SUSHANT — 60s.\nWorth 3 marks, and the rubric rewards building on papers rather "
             "than summarising them. Spend the time on the third column and the two 'we take' "
             "lines.\nThe bottom strip matters. We are importing a 19-year-old protocol from "
             "biometrics rather than inventing one. Say that out loud.")
    return s


def slide_dataset(prs):
    s = base(prs, "Dataset", "Nutrition5k: 5,006 dishes, each weighed on a scale.", 6)
    y = s._body_top

    pic(s, "plate_grid.png", L, y + 0.06, 3.40)
    caption(s, L, y + 3.58, 3.40,
            "Overhead RGB, one plate, one fixed rig. Every ingredient weighed beforehand.")

    pic(s, "dataset_scatter.png", 4.85, y + 0.06, 6.80)
    caption(s, 4.90, y + 3.54, 6.75,
            "Mass and calories correlate at r = 0.76. Weighing the food gets you about half "
            "the answer. The rest is knowing what it is.", size=12.5, color=INK, bold=True)

    cy = y + 4.20
    eyebrow(s, L, cy, "The split we use, and what it costs", w=9.0)
    cy += 0.32
    stages = [("4,768", "official split ids"), ("3,490", "have overhead RGB"),
              ("3,262", "usable dishes"), ("2,755 / 507", "train / test"),
              ("1,929 / 826", "fit / calibration")]
    bw, gap = 2.02, 0.32
    for i, (num, lbl) in enumerate(stages):
        x = L + i * (bw + gap)
        rect(s, x, cy, bw, 0.60, fill=TINT if i >= 3 else BG)
        t = tb(s, x, cy + 0.08, bw, 0.5)
        para(t, num, size=14, color=ACCENT if i >= 3 else INK, bold=True,
             align=PP_ALIGN.CENTER, first=True)
        para(t, lbl, size=10.5, color=MUTED, align=PP_ALIGN.CENTER, space_before=2)
        if i < len(stages) - 1:
            arrow(s, x + bw + 0.06, cy + 0.22, 0.20, 0.16)

    notes(s, "SUSHANT — 45s.\nLeft: this is what the data looks like. Overhead, one plate, one "
             "camera rig, cafeteria food. Every dish was weighed ingredient by ingredient "
             "before the photo, which is why we can build a perfect refuser to measure "
             "against.\n"
             "Right is the slide's argument. Each dot is a dish. Point at the orange band: 736 "
             "dishes weigh between 150 and 250 grams, and their calories run from 23 to 792. "
             "Knowing the weight is not enough. That is why this stays a vision problem.\n"
             "Bottom: we lose about 1,500 dishes to missing imagery and report 507 test "
             "dishes, not the 709 in the split file. That is the most likely question on this "
             "slide, so say it before it is asked. Hand to Keanu.")
    return s


def slide_predictor(prs):
    s = base(prs, "Methodology — the predictor",
             "The predictor is built. We are not trying to improve it.", 7)
    y = s._body_top

    eyebrow(s, L, y, "Three frozen backbones, ridge head, no training", w=7.0)
    pic(s, "backbones.png", L - 0.10, y + 0.32, 6.9)
    caption(s, L, y + 3.42, 6.6,
            "507 test dishes, two minutes of compute each. DINOv3 is the only one that "
            "improves portion size. We keep CLIP as the primary model because it matches the "
            "published baseline, and report the others as a robustness check.")

    x2 = 7.75
    eyebrow(s, x2, y, "Why we are not chasing the number", w=4.7)
    t = tb(s, x2, y + 0.32, 4.7, 0.5)
    para(t, "A model that named and weighed every ingredient, then made the mistakes we "
            "actually measure:", size=11.5, color=MUTED, line=1.3, first=True)
    ry = y + 0.90
    hrule(s, x2, ry - 0.06, 4.7, color=FAINT)
    for label, val, hi in [
        ("Everything known exactly", "33 kcal", False),
        ("Mass off by 10%, names perfect", "43 kcal", False),
        ("Mass off by 22%, one name in five wrong", "87 kcal", True),
    ]:
        rh = 0.74 if hi else 0.54
        rect(s, x2 - 0.15, ry + 0.02, 4.85, rh, fill=TINT if hi else BG)
        t = tb(s, x2, ry + 0.13, 3.3, 0.6)
        para(t, label, size=12.5, color=INK, bold=hi, line=1.25, first=True)
        t = tb(s, x2 + 3.40, ry + 0.13, 1.3, 0.4)
        para(t, val, size=13.5, color=INK, bold=hi, first=True)
        ry += rh
    caption(s, x2, ry + 0.18, 4.7,
            "Naming the ingredients separately lands where our one-step model already is. It "
            "buys an explanation, not a better number.")

    takeaway(s, "The predictor is a fixed control. The open question is whether the app can "
                "tell when it is about to be wrong.", y=6.35, size=15)
    notes(s, "KEANU — 50s.\nLeft: three frozen backbones, no training anywhere, two minutes of "
             "compute each. DINOv3 is the only one that improves portion size, 22% down to "
             "19%, which fits what it was built for. We keep CLIP as primary because it lands "
             "on the published baseline and we are not trying to win on accuracy.\n"
             "Right: read the three rows as a budget. Perfect knowledge gives 33. Mass to "
             "within 10% gives 43, roughly the best number in the literature, and that one "
             "needed a depth sensor. With the 22% mass error we actually measure, the two-step "
             "route gives 87, worse than the 73.4 we already have.\n"
             "If asked whether the budget is unfair: yes, deliberately. It is a requirements "
             "spec, not a model, and the point is that it does not promise us headroom.")
    return s


def slide_gate_broken(prs):
    s = base(prs, "Preliminary result 1 — completed",
             "Cheap pixel signals do not predict the error.", 8)
    y = s._body_top

    eyebrow(s, L, y, "One dish, three photographs")
    py = y + 0.32
    pw, gap = 1.80, 0.15
    for i, (fn, label, sharp, err, hi) in enumerate([
            ("panel_clean.png", "clean", "177", "44 g", False),
            ("panel_blur6.png", "blurred", "1.5", "71 g", False),
            ("panel_crop04.png", "cropped", "6.5", "167 g", True)]):
        px = L + i * (pw + gap)
        pic(s, fn, px, py, pw)
        t = tb(s, px, py + pw * 0.75 + 0.10, pw, 0.9)
        para(t, label, size=13, color=ACCENT if hi else INK, bold=True,
             align=PP_ALIGN.CENTER, first=True)
        para(t, [("sharpness  ", {"color": MUTED, "size": 11}),
                 (sharp, {"color": INK, "size": 12, "bold": True})],
             align=PP_ALIGN.CENTER, space_before=4)
        para(t, [("mass error  ", {"color": MUTED, "size": 11}),
                 (err, {"color": ACCENT if hi else INK, "size": 12, "bold": True})],
             align=PP_ALIGN.CENTER, space_before=2)

    caption(s, L, py + pw * 0.75 + 0.98, 6.0,
            "Sharpness calls the cropped photo four times better than the blurred one. It "
            "carries two and a half times the error. Cropping removes the plate rim, the only "
            "clue to how big the food is.", size=12.5, color=INK, bold=True)

    cx = 7.15
    eyebrow(s, cx, y, "Refuse the worst photos. Does the error fall?")
    pic(s, "risk_coverage.png", cx, y + 0.28, 5.15)

    sy = 5.84
    rect(s, L - 0.18, sy, CW + 0.36, 1.06, fill=TINT)
    t = tb(s, L + 0.14, sy + 0.14, CW - 0.3, 0.92)
    para(t, [("WE SPOT THE DAMAGE ALMOST PERFECTLY   ",
              {"color": ACCENT, "bold": True, "size": 11.5}),
             ("sharpness separates blurred, cropped, shrunk and noisy photos from clean ones "
              "almost every time (AUC ≈ 1.00). Seeing damage is not the problem. Damage "
              "explains only 7% of the error, and which dish it is explains 56%.",
              {"color": INK, "size": 12.5})], line=1.32, first=True)
    para(t, "The error is about the dish, not the photograph.",
         size=14, color=INK, bold=True, space_before=7)

    notes(s, "KEANU — 65s. This is the slide the project rests on, so slow down.\n"
             "Left: same dish, three photographs. Point at the cropped one. It is sharp, it is "
             "clean, and it is the worst of the three, because cropping removed the plate rim "
             "and with it any sense of scale. A sharpness score ranks it four times better "
             "than the blurred photo, and it is two and a half times more wrong.\n"
             "Right: refuse the photos each signal likes least and watch the error. Sharpness "
             "in blue and refusing at random in dashed grey sit on top of each other. A "
             "perfect refuser in green drops the error from 106 to 40.\n"
             "The orange strip is the honest reading and it surprised us. We are not failing "
             "to detect damage, we detect it almost perfectly. Damage is simply 7% of the "
             "error while which dish it is accounts for 56%.\n"
             "If asked why we degrade photos at all: that is how we know, and the answer "
             "turned out to be that it matters less than we assumed. Hand to Shreya.")
    return s


def slide_gate_works(prs):
    s = base(prs, "Preliminary result 2 — completed",
             "Signals that read the model do work. One cheats.", 9)
    y = s._body_top

    eyebrow(s, L, y, "Refuse the worst half, clean photos only", w=6.0)
    pic(s, "gates.png", L - 0.10, y + 0.30, 6.0)

    cx = 6.60
    eyebrow(s, cx, y, "Then check what each one refuses", w=5.8)
    pic(s, "degeneracy.png", cx, y + 0.26, 5.80)

    sy = 5.58
    rect(s, L - 0.18, sy, CW + 0.36, 1.20, fill=TINT)
    t = tb(s, L + 0.14, sy + 0.14, CW - 0.3, 1.04)
    para(t, [("THE CONTROL MATTERS   ", {"color": ACCENT, "bold": True, "size": 11.5}),
             ("scoring risk by the predicted calorie count alone reaches 47.6 kcal, which "
              "looks like one of the best gates. It gets there by refusing 98% of dishes over "
              "400 kcal and answering only for small meals. Error grows with portion size, so "
              "any gate can win on error alone by declining big plates.",
              {"color": INK, "size": 12.5})], line=1.32, first=True)
    para(t, "Ensemble disagreement refuses large meals at the same rate as a perfect refuser, "
            "71% against 72%, and still cuts the error from 73.4 to 50.6.",
         size=13, color=INK, bold=True, space_before=6, line=1.3)

    notes(s, "SHREYA — 65s.\nLeft: every signal scored the same way. Refuse the worst half of "
             "the photos and see what error is left on the ones you still answer. Answering "
             "everything is 73.4. A perfect refuser is 25.6. The cheap pixel signals from the "
             "last slide are not even on this chart, they sit at random.\n"
             "Right is the check that matters and it is why this slide exists. Plot each gate "
             "by the average size of the meals it still answers. A perfect refuser keeps meals "
             "averaging 185 kcal. The blue control keeps meals averaging 101. It is not finding "
             "hard dishes, it is refusing big ones.\n"
             "Orange is the honest winner. It sits almost on top of the perfect refuser's "
             "selectivity and still cuts the error by a third.\n"
             "If asked why this matters for deployment: an app that refuses every large meal "
             "is useless, because large meals are the ones a health decision turns on.")
    return s


def slide_experiment(prs):
    s = base(prs, "Proposed methodology",
             "Damage the photos, then score every signal.", 10)
    y = s._body_top
    sw = 7.05

    eyebrow(s, L, y, "Damage a sharpness metric can see", color=BLUE, w=7.0)
    pic(s, "ladder_visible.png", L, y + 0.30, sw)
    caption(s, L, y + 2.10, sw, "blur  ·  camera shake  ·  shrinking  ·  noise", size=11.5)

    y2 = y + 2.46
    eyebrow(s, L, y2, "Damage it cannot", w=7.0)
    pic(s, "ladder_invisible.png", L, y2 + 0.30, sw)
    caption(s, L, y2 + 2.10, sw,
            "cropping  ·  brightness  ·  colour  ·  compression, plus a phone-camera rung",
            size=11.5)

    x2 = 8.45
    sy = y + 0.02
    for num, title, body in [
        ("1", "The grouping is the test",
         "The cheap metrics move on the first group and are blind to the second. The error is "
         "driven by the second."),
        ("2", "One results file, written once",
         "Every dish, every damage level, every gate score, in a single CSV. Every question "
         "after that is a lookup."),
        ("3", "Three measurements per gate",
         "How well it ranks the errors, the refuse-and-measure curve [3], and what it does to "
         "the width of the confidence range."),
    ]:
        t = tb(s, x2, sy, 0.4, 0.4)
        para(t, num, size=16, color=ACCENT, bold=True, first=True)
        t = tb(s, x2 + 0.42, sy - 0.02, 3.55, 1.5)
        para(t, title, size=13, color=INK, bold=True, line=1.25, first=True)
        para(t, body, size=11.5, color=MUTED, space_before=4, line=1.3)
        sy += 1.58

    rect(s, L - 0.18, 6.58, CW + 0.36, 0.62, fill=TINT)
    t = tb(s, L + 0.14, 6.71, CW - 0.3, 0.4)
    para(t, [("Headline question:  ", {"color": ACCENT, "bold": True}),
             ("which signals let an app know it is guessing, and can any of them run on the "
              "phone?", {"color": INK, "bold": True})], size=14, first=True)

    notes(s, "SHREYA — 55s.\nThe two strips are the experiment. Top row is damage a sharpness "
             "metric detects easily. Bottom row is damage it misses. Our claim is that the "
             "bottom row drives the error, and slide 8 already showed that for cropping.\n"
             "Then the three steps on the right, quickly, and finish on the headline "
             "question.\n"
             "If asked why conformal prediction: distribution-free, finite-sample, and it "
             "works on top of a frozen predictor. Training an ensemble of full models is what "
             "our hardware rules out. If asked about exchangeability: one damage level per "
             "dish, drawn from a stated deployment mix, and we report sensitivity over three "
             "mixes. Hand to Kevin.")
    return s


def slide_timeline(prs):
    s = base(prs, "Timeline & team roles", "Nine weeks, split so nobody blocks anybody.", 11)
    y = s._body_top
    ry = table_head(s, y, [L, L + 2.5, L + 9.1], [2.4, 6.4, 2.3], ["When", "Milestone", "Lead"])

    for i, (when, what, lead, done) in enumerate([
        ("done — 17 Aug", "Data, frozen pipeline, three backbones, first gate results", "all", True),
        ("18 – 25 Aug", "More damage levels: phone-camera and hidden food", "Shreya", False),
        ("26 Aug – 11 Sep", "All eight gates scored; the results file", "Nisarg", False),
        ("20 Sep", "Literature review due (10%)", "Sushant", False),
        ("21 – 28 Sep", "Update presentation", "Keanu", False),
        ("29 Sep – 11 Oct", "Confidence ranges: honesty, width, usable share", "Kevin", False),
        ("25 Oct", "Research report due (22%)", "all", False),
    ]):
        rh = 0.42
        if done:
            rect(s, L - 0.18, ry - 0.07, CW + 0.36, rh, fill=TINT)
        elif i % 2 == 1:
            rect(s, L - 0.18, ry - 0.07, CW + 0.36, rh, fill=BG)
        t = tb(s, L, ry, 2.4, 0.32)
        para(t, when, size=13, color=ACCENT if done else MUTED, bold=done, first=True)
        t = tb(s, L + 2.5, ry, 6.4, 0.32)
        para(t, what, size=13, color=INK, bold=done, first=True)
        t = tb(s, L + 9.1, ry, 2.3, 0.32)
        para(t, lead, size=13, color=INK, bold=True, first=True)
        ry += rh

    ry += 0.20
    eyebrow(s, L, ry, "Who owns what")
    ry += 0.32
    for i, (who, what) in enumerate([
            ("Nisarg", "pipeline + results file"), ("Shreya", "damage levels + pixel gates"),
            ("Keanu", "model-aware gates + curves"), ("Sushant", "literature + writing"),
            ("Kevin", "confidence ranges")]):
        t = tb(s, L + i * 2.32, ry, 2.25, 0.5)
        para(t, who, size=13, color=INK, bold=True, first=True)
        para(t, what, size=11, color=MUTED, space_before=2, line=1.25)

    takeaway(s, "One results file is the only shared dependency. A member stalling costs one "
                "column, not the project.", y=ry + 0.60, size=14)
    notes(s, "KEVIN — 45s.\nLead with the top row. The first milestone is done and so are the "
             "first two results, which is why the rest of the timeline is credible. Then name "
             "all five people from the roles strip. Close on the shared-dependency line.")
    return s


def slide_risks(prs):
    s = base(prs, "Risks & Plan B", "Every outcome still gives us a result.", 12)
    y = s._body_top
    ry = table_head(s, y, [L, L + 4.8], [4.6, 6.7], ["Risk", "Plan B"])

    for i, (risk, plan) in enumerate([
        ("No gate beats refusing at random",
         "That is the result, and it is the useful one. It bounds what abstention on a phone "
         "can achieve, measured against a ceiling we can compute exactly."),
        ("The winning gate turns out to be degenerate",
         "We already found one that is, and caught it with a trivial control. Every gate is "
         "reported with what it refuses, not only with its error."),
        ("Only the expensive gates work",
         "The cost column is part of the finding. Reliability costing K extra runs is a "
         "deployment result, not a failed experiment."),
        ("Nutrition5k is one cafeteria, one fixed rig",
         "There is no phone camera in this data, so our phone rung is an imitation and we say "
         "so. Claims stay within-distribution, sliced by ingredient rarity rather than cuisine."),
    ]):
        rh = 0.86
        if i % 2 == 0:
            rect(s, L - 0.18, ry - 0.09, CW + 0.36, rh, fill=BG)
        t = tb(s, L, ry, 4.6, 0.70)
        para(t, risk, size=13.5, color=INK, bold=True, line=1.28, first=True)
        t = tb(s, L + 4.8, ry, 6.7, 0.70)
        para(t, plan, size=12.5, color=MUTED, line=1.32, first=True)
        ry += rh

    takeaway(s, "The question is which signals work. “None of the cheap ones do” is an answer.",
             y=ry + 0.24, size=15)
    notes(s, "KEVIN — 35s.\nDo not read all four. Say the first and the second. A null result "
             "still bounds what is achievable, and the degeneracy risk is one we have already "
             "hit and already caught. Note the others are quantified rather than hoped. Close "
             "on the bold line and hand back for questions.")
    return s


def slide_questions(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, fill=BG)
    rect(s, 0, 0, 0.16, H, fill=ACCENT)

    t = tb(s, L + 0.25, 1.35, 11.0, 1.0)
    para(t, "Questions", size=46, color=INK, bold=True, first=True)
    t = tb(s, L + 0.25, 2.45, 10.5, 0.9)
    para(t, "The error is about the dish, not the photograph.\n"
            "We can measure that, and we can act on it.", size=19, color=ACCENT,
         line=1.35, first=True)
    rect(s, L + 0.25, 3.70, 1.40, 0.037, fill=FAINT)

    eyebrow(s, L + 0.25, 4.05, "References", color=MUTED)
    t = tb(s, L + 0.25, 4.40, 11.2, 2.3)
    for i, ref in enumerate([
        "[1]  Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, "
        "“Nutrition5k: Towards automatic nutritional understanding of generic food,” in Proc. "
        "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 8903–8911.",
        "[2]  B. Coburn, J. He, M. E. Rollo, S. S. Dhaliwal, D. A. Kerr, and F. Zhu, "
        "“Comprehensive evaluation of large multimodal models for nutrition analysis: A new "
        "benchmark enriched with contextual metadata,” in Proc. IEEE Int. Conf. Biomed. "
        "Health Informat. (BHI), 2025.",
        "[3]  P. Grother and E. Tabassi, “Performance of biometric quality measures,” IEEE "
        "Trans. Pattern Anal. Mach. Intell., vol. 29, no. 4, pp. 531–543, Apr. 2007.",
        "[4]  D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to "
        "common corruptions and perturbations,” in Proc. Int. Conf. Learn. Representations "
        "(ICLR), 2019.",
    ]):
        para(t, ref, size=11.5, color=MUTED, line=1.4, space_before=0 if i == 0 else 6,
             first=(i == 0))
    para(t, "All figures, tables and numbers in this deck are our own computation on the "
            "Nutrition5k dataset [1] unless cited otherwise.",
         size=11, color=FAINT, space_before=9, line=1.35)

    notes(s, "ALL — Q&A. The rubric rewards more than one member answering, so route to the "
             "owner: gate results → Keanu, damage levels → Shreya, confidence ranges → Kevin, "
             "literature → Sushant, framing and data → Nisarg. Answer format: direct answer "
             "first, one sentence of reasoning, stop.")
    return s


def main() -> None:
    missing = [f for f in REQUIRED_FIGS if not (FIGS / f).exists()]
    if missing:
        raise SystemExit(f"missing figures {missing}\n"
                         "run: gate_bench.py, gate_probe.py --all --json, figures.py")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(int(W * 914400)), Emu(int(H * 914400))
    for fn in (slide_title, slide_motivation, slide_problem, slide_objectives,
               slide_literature, slide_dataset, slide_predictor, slide_gate_broken,
               slide_gate_works, slide_experiment, slide_timeline, slide_risks,
               slide_questions):
        fn(prs)

    out = ROOT / "01_Proposal_v6.pptx"
    prs.save(out)
    total = sum(s[1] for s in SLIDE_PLAN)
    print(f"wrote {out}")
    print(f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides | "
          f"target {total}s = {total // 60}:{total % 60:02d} spoken")
    by = {}
    for speaker, secs, _ in SLIDE_PLAN:
        by[speaker] = by.get(speaker, 0) + secs
    print("speaking split:", ", ".join(f"{k} {v}s" for k, v in by.items()))


if __name__ == "__main__":
    main()
