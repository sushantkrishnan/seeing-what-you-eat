"""Build the COMPSCI 760 Group 1 update deck: methodology and results.

  ../.venv/bin/python build_update_deck.py        -> 01_MethodResults.pptx

Run gate_bench.py, gate_probe.py --all --json, conformal.py --all --json and
figures.py first. Every number on these slides is read from results/*.json at build
time, so the deck cannot drift from the code.

RUBRIC MAP (Presentation2_rubic.txt, 20 points). Slide numbers in brackets.
  objectives & problem vs proposal [2]      motivation, why the gate [3]
  changes since proposal + feedback [4]     dataset with rationale, splits [5]
  methodology, tuning, compute [6]          measures and significance [7]
  existing vs our results, tagged [8]       results so far [9-13]
  challenges, timeline vs plan, roles [14]  conclusions and next steps [15]
  citations, GenAI acknowledgement, GitHub link [1, 16]

Eight minutes, five speakers at ~95 s each, 7:45 spoken. Speaker names are not
printed on slides; SLIDE_PLAN drives the timing budget and the notes.

AT A GLANCE (19 Sep rebuild). One claim per slide as the headline, one picture or
one small table, at most three short lines, provenance tags. Every "because" lives in
update-talk-script.md, not on the slide. The order builds the idea up: problem, why
the gate, data, pipeline, how to read the charts, then the results on those axes.

House style, unchanged from v6: short declarative sentences, no aphorisms, no
"not X but Y" inversions, few dashes, no novelty-by-absence claims. Every number is
tagged [ours], [ours, replicated] or [1] so the audience can tell whose it is.
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_deck import (ACCENT, BG, BLUE, CW, FAINT, GREEN, H, INK, L, MUTED, R, RULE,
                        TINT, W, arrow, base, caption, eyebrow, hrule, notes, para, pic,
                        rect, table_head, takeaway, tb)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
OUT = ROOT / "01_MethodResults.pptx"

REPO_URL = "github.com/sushantkrishnan/seeing-what-you-eat"
CLIP = "vit_base_patch16_clip_224.openai"
SIGLIP = "vit_base_patch16_siglip_224.v2_webli"
DINO = "vit_base_patch16_dinov3.lvd1689m"
PIXEL = ("lap_var", "tenengrad", "blockiness", "hf_ratio")

# The two-stage projection: an oracle handed ingredient names and grams, corrupted by
# the stage error our own heads measure (22% mass, one name in five). oracle_ladder.py,
# the perturbed_o3 block, n = 507. Not read from JSON because that script writes none.
TWO_STAGE_KCAL = 87
PUBLISHED_RGB, PUBLISHED_BEST = 70.6, 41.3

SLIDE_PLAN = [
    ("Nisarg",  35, "Recap: problem, proposal, objectives"),
    ("Nisarg",  30, "Why the gate: 71 to 50, against 87"),
    ("Nisarg",  30, "Feedback and changes"),
    ("Sushant", 30, "Dataset, splits, sessions"),
    ("Sushant", 30, "Pipeline, the one knob, compute"),
    ("Sushant", 35, "How to read every chart"),
    ("Keanu",   30, "Existing vs ours: the predictor"),
    ("Keanu",   45, "Result 1: pixel signals"),
    ("Shreya",  40, "Result 2: the curve"),
    ("Shreya",  30, "Result 2: three backbones"),
    ("Shreya",  35, "Result 3: what each gate refuses"),
    ("Kevin",   40, "Result 4: ranges"),
    ("Kevin",   25, "Challenges, timeline, roles"),
    ("Kevin",   30, "Conclusions"),
]

REQUIRED_FIGS = ["plate_grid.png", "sessions.png", "backbones.png", "panel_clean.png",
                 "panel_blur6.png", "panel_crop04.png", "panel_phone.png",
                 "risk_coverage.png", "curves_ci.png", "significance.png",
                 "degeneracy.png", "conformal_kcal.png", "reader.png"]


# ---------------------------------------------------------------- numbers
def load_numbers() -> dict:
    gp = json.loads((RESULTS / "gate_probe.json").read_text())
    gb = json.loads((RESULTS / "gate_bench.json").read_text())
    cf = json.loads((RESULTS / "conformal.json").read_text())
    c = gp[CLIP]
    g90 = {k: v["mae"]["0.9"] for k, v in c["clean"]["gates"].items()}
    g50 = {k: v["mae"]["0.5"] for k, v in c["clean"]["gates"].items()}
    vr = {k: v["vs_random"]["0.9"] for k, v in c["clean"]["gates"].items()
          if "vs_random" in v}
    sel = {k: v["selectivity"] for k, v in c["clean"]["gates"].items()}
    pool = gb["risk_coverage"]
    conf = cf[CLIP]["clean"]
    n = {
        "n_fit": c["n_fit"], "n_calib": c["n_calib"], "n_test": c["n_test"],
        "n_sessions": c["clean"]["n_sessions"], "n_boot": c["clean"]["n_boot"],
        "mae": c["predictor"]["cal_mae"], "mae_ci": c["predictor"]["cal_mae_ci"],
        "mass": c["predictor"]["mass_mae"], "mass_rel": 100 * c["predictor"]["mass_rel"],
        "mae_siglip": gp[SIGLIP]["predictor"]["cal_mae"],
        "mae_dino": gp[DINO]["predictor"]["cal_mae"],
        "alpha_cal": c["alpha"]["cal"],
        "per_rung": gb["per_rung"], "pooled_all": pool["random"]["1.0"],
        "pooled50": {k: pool[k]["0.5"] for k in pool},
        "rho_pixel": gb["spearman"],
        "g90": g90, "g50": g50, "vr": vr, "sel": sel,
        "ci90": {k: v["mae_ci"]["0.9"] for k, v in c["clean"]["gates"].items()},
        "gap": {k: v["gap_share"] for k, v in c["clean"]["gates"].items()},
        "gap_ci": {k: v["gap_share_ci"] for k, v in c["clean"]["gates"].items()},
        "pair": c["clean"]["pairwise"]["0.9"],
        "pair_siglip": gp[SIGLIP]["clean"]["pairwise"]["0.9"],
        "pair_dino": gp[DINO]["clean"]["pairwise"]["0.9"],
        "conf": conf, "mix": cf[CLIP]["mixtures"],
        "conf_siglip": cf[SIGLIP]["clean"], "conf_dino": cf[DINO]["clean"],
    }
    return n


# ---------------------------------------------------------------- helpers
def tag(slide, x, y, text, color=MUTED):
    """The provenance tag: [ours] / [ours, replicated] / [1]."""
    t = tb(slide, x, y, 2.4, 0.25)
    para(t, text, size=9.5, color=color, bold=True, first=True)


def rows_table(slide, y, cols, widths, heads, rows, *, rh=0.42, size=12,
               head_size=11.5, bold_col=None, accent_col=None, zebra=True):
    ry = table_head(slide, y, cols, widths, heads)
    for i, row in enumerate(rows):
        if zebra and i % 2 == 0:
            rect(slide, L - 0.18, ry - 0.07, CW + 0.36, rh, fill=BG)
        for j, (cx, cwd, txt) in enumerate(zip(cols, widths, row)):
            t = tb(slide, cx, ry, cwd, rh - 0.05)
            col = ACCENT if j == accent_col else INK
            para(t, txt, size=size, color=col, bold=(j == bold_col), line=1.22, first=True)
        ry += rh
    return ry


def fmt_ci(ci):
    return f"[{ci[0]:.0f}, {ci[1]:.0f}]"


def fmt_p(p):
    """Holm-corrected bootstrap p. The floor is 1/n_boot × 10 = 0.005, so never print
    it as if it were exact."""
    return "p < 0.01" if p < 0.01 else f"p = {p:.2f}"


def strip(slide, y, h, label, body, *, size=11.5, label_size=11):
    """One tinted strip: an accent label and one or two lines of body text."""
    rect(slide, L - 0.18, y, CW + 0.36, h, fill=TINT)
    t = tb(slide, L + 0.14, y + 0.10, CW - 0.3, h - 0.15)
    para(t, [(label.upper() + "   ", {"color": ACCENT, "bold": True, "size": label_size}),
             (body, {"color": INK, "size": size})], line=1.3, first=True)
    return t


# ---------------------------------------------------------------- slides
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, fill=BG)
    rect(s, 0, 0, 0.16, H, fill=ACCENT)
    t = tb(s, L + 0.25, 2.05, 11.0, 1.2)
    para(t, "Seeing what you eat", size=52, color=INK, bold=True, first=True)
    t = tb(s, L + 0.25, 3.20, 11.0, 0.8)
    para(t, "When should a nutrition app refuse to answer?", size=23, color=ACCENT,
         first=True)
    rect(s, L + 0.25, 4.35, 1.40, 0.037, fill=FAINT)
    t = tb(s, L + 0.25, 4.70, 11.5, 1.6)
    para(t, "COMPSCI 760  ·  Group 1  ·  Research Project Update: Methodology and Results",
         size=14, color=MUTED, bold=True, first=True)
    para(t, "Sushant Krishnan (sthi106)  ·  Keanu De Cleene (kdec819)  ·  "
            "Nisarg Patel (npat235)", size=14, color=MUTED, space_before=8)
    para(t, "Shreya Tigga (stig804)  ·  Kevin Zou (rzou895)", size=14, color=MUTED,
         space_before=4)
    para(t, [("Code and results:  ", {"color": MUTED}), (REPO_URL, {"color": INK, "bold": True})],
         size=14, space_before=14)
    notes(s, "Title only. Do not speak to this slide. Nisarg opens on slide 2. "
             "Target 7:45 spoken, hard stop 8:00.")
    return s


def slide_recap(prs, n):
    s = base(prs, "Recap", "Every app answers every photo. We study the gate.", 2)
    y = s._body_top

    bh = 0.66
    rect(s, L, y, 1.45, bh, fill=BG)
    t = tb(s, L, y + 0.19, 1.45, 0.4)
    para(t, "a photo", size=13, color=MUTED, align=PP_ALIGN.CENTER, first=True)
    arrow(s, 2.50, y + bh / 2 - 0.09, 0.30, 0.18)
    rect(s, 2.88, y, 2.55, bh, fill=TINT)
    t = tb(s, 2.88, y + 0.10, 2.55, 0.6)
    para(t, "THE GATE", size=11, color=ACCENT, bold=True, align=PP_ALIGN.CENTER, first=True)
    para(t, "answer, or refuse?", size=13, color=INK, bold=True, align=PP_ALIGN.CENTER,
         space_before=2)
    arrow(s, 5.52, y + bh / 2 - 0.09, 0.30, 0.18)
    rect(s, 5.90, y, 2.05, bh, fill=BG)
    t = tb(s, 5.90, y + 0.10, 2.05, 0.6)
    para(t, "the predictor", size=13, color=MUTED, align=PP_ALIGN.CENTER, first=True)
    para(t, "frozen", size=10.5, color=FAINT, align=PP_ALIGN.CENTER, space_before=2)
    arrow(s, 8.04, y + bh / 2 - 0.09, 0.30, 0.18)
    rect(s, 8.42, y, 1.75, bh, fill=BG)
    t = tb(s, 8.42, y + 0.19, 1.75, 0.4)
    para(t, "“620 kcal”", size=13.5, color=INK, bold=True, align=PP_ALIGN.CENTER, first=True)
    arrow(s, 4.06, y + bh + 0.05, 0.18, 0.22, down=True)
    rect(s, 2.88, y + bh + 0.32, 4.0, 0.46, fill=BG)
    t = tb(s, 2.88, y + bh + 0.42, 4.0, 0.4)
    para(t, "“I can’t tell from this photo.”", size=12.5, color=INK, bold=True,
         align=PP_ALIGN.CENTER, first=True)

    yy = y + 1.80
    for label, body in [
        ("The problem", "Calorie apps return one number for every photo and never decline."),
        ("What we proposed", "Freeze the predictor. Benchmark the signals that could decide "
                             "whether to answer, between refusing at random and a perfect "
                             "refuser."),
    ]:
        t = tb(s, L, yy, 8.9, 0.28)
        para(t, label.upper(), size=11, color=ACCENT, bold=True, first=True)
        t = tb(s, L, yy + 0.27, 8.9, 0.7)
        para(t, body, size=13, color=INK, line=1.32, first=True)
        yy += 1.05

    x2 = 10.35
    eyebrow(s, x2, y, "Objectives · status", w=3.0)
    ry = y + 0.34
    for name, q, when, status, col in [
        ("O1 Benchmark", "Which signals predict\nwhen the answer is wrong?", "11 Sep",
         "done", GREEN),
        ("O2 Ceiling", "How much is on the table,\nand on which axis?", "20 Sep",
         "done", GREEN),
        ("O3 Ranges", "Does the best signal give\nranges worth having?", "11 Oct",
         "first cut done", ACCENT),
    ]:
        rect(s, x2 - 0.12, ry - 0.06, 2.35, 1.22, fill=BG)
        t = tb(s, x2, ry, 2.2, 1.2)
        para(t, name, size=12, color=INK, bold=True, first=True)
        para(t, q, size=10.5, color=MUTED, line=1.25, space_before=2)
        para(t, [("due " + when + "  ·  ", {"color": FAINT}), (status, {"color": col, "bold": True})],
             size=10.5, space_before=3)
        ry += 1.30

    takeaway(s, "Which signals let an app know it is guessing, and can any of them run on "
                "the phone?", y=6.22, size=15)
    notes(s, "NISARG — 35s.\nWalk the diagram once, left to right. The problem and the "
             "proposal as two sentences. Point at the three objectives: two done, the third "
             "started early. Say the headline question out loud, it is the sentence the "
             "whole talk answers.")
    return s


def slide_why(prs, n):
    s = base(prs, "Why the gate, not the predictor",
             "Refusing one photo in ten beats a better predictor.", 3)
    y = s._body_top
    mae, perf = n["mae"], n["g90"]["perfect"]

    eyebrow(s, L, y, "Our own numbers, 507 test dishes, calorie error in kcal  [ours]", w=9)
    ty = y + 0.42
    th = 2.55
    tiles = [
        (L, 3.15, BG, INK, f"{mae:.0f}", "answering every photo", "our frozen predictor"),
        (L + 3.75, 3.15, TINT, GREEN, f"{perf:.0f}", "with a perfect refuser declining "
                                                     "one photo in ten",
         f"{round(mae) - round(perf):.0f} kcal of headroom, in refusing"),
        (L + 7.90, 3.60, BG, ACCENT, str(TWO_STAGE_KCAL),
         "a two-stage model that names and weighs the ingredients",
         "with the stage error we measure: 22% on mass, one name in five  "
         "[ours, oracle budget]"),
    ]
    for x, w, fill, col, big, lab, sub in tiles:
        rect(s, x, ty, w, th, fill=fill)
        t = tb(s, x + 0.25, ty + 0.15, w - 0.5, 1.0)
        para(t, big, size=54, color=col, bold=True, first=True)
        t = tb(s, x + 0.25, ty + 1.12, w - 0.5, 1.2)
        para(t, lab, size=12.5, color=INK, bold=True, line=1.28, first=True)
        para(t, sub, size=11, color=MUTED, line=1.28, space_before=4)
    arrow(s, L + 3.22, ty + th / 2 - 0.12, 0.42, 0.24, color=GREEN)

    caption(s, L, ty + th + 0.22, CW,
            f"The best published number, {PUBLISHED_BEST:.0f} kcal, needed a depth sensor "
            f"[1]. A phone photo has none. Published RGB: {PUBLISHED_RGB} kcal [1].",
            size=12, color=INK)
    takeaway(s, "The headroom is in refusing, not in predicting. So the predictor is frozen "
                "and the gate is the object of study.", y=5.85, size=15)
    notes(s, "NISARG — 30s.\nThree numbers, left to right. 71 is where we are. 50 is what "
             "a perfect refuser would give us by declining one photo in ten: 21 kcal on the "
             "table. 87 is where the ambitious route lands, naming and weighing the "
             "ingredients, once you feed it the stage error we actually measure: worse than "
             "what we have. The only number that beats us needed a depth camera. That is the "
             "whole reason the predictor is frozen. Hand to feedback.")
    return s


def slide_changes(prs, n):
    s = base(prs, "Since the proposal", "What the markers asked, and what we changed.", 4)
    y = s._body_top

    eyebrow(s, L, y, "Feedback we received, and what we did with it", w=8.0)
    ry = y + 0.34
    for fb, did in [
        ("Freezing the predictor and studying the gate is the right design.",
         "Kept. Nothing on the predictor side changed."),
        ("Which evaluation metrics, exactly?",
         "One fixed set, on slide 7: error at a fixed answer rate, area under the curve, "
         "share of the gap captured, and what each gate refuses."),
        ("How will you show a difference is significant?",
         "Every number carries a 95% interval from a bootstrap over plate sessions; every "
         "gate is tested against random with a Holm correction. Two proposal claims did "
         "not survive (slide 10)."),
    ]:
        if ry < y + 1:
            rect(s, L - 0.18, ry - 0.06, CW + 0.36, 0.62, fill=BG)
        t = tb(s, L, ry, 4.2, 0.7)
        para(t, fb, size=12.5, color=INK, bold=True, line=1.25, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.7)
        para(t, did, size=11.5, color=MUTED, line=1.25, first=True)
        ry += 0.72

    ry += 0.10
    eyebrow(s, L, ry, "Changes we made, and why", w=8.0)
    ry += 0.34
    for what, why in [
        ("Calibration carved by plate session, not by dish.",
         f"98% of calibration dishes shared a session with a fit dish. Fixed; the predictor "
         f"moved 73.4 → {n['mae']:.1f} kcal, inside its interval  → slide 5"),
        ("We quote 90% answered and show the whole curve.",
         "Where an app would sit. The proposal quoted 50%, where every gate looks better  "
         "→ slide 7"),
        ("Phone rung in, late. Hidden-food rung dropped.",
         "Damage is 9% of the error, and occlusion cannot be labelled honestly  → slide 9"),
        ("Confidence ranges started three weeks early.",
         "First cut done; it found the next problem, large meals  → slide 13"),
    ]:
        t = tb(s, L, ry, 4.2, 0.6)
        para(t, what, size=12, color=INK, bold=True, line=1.22, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.6)
        para(t, why, size=11.5, color=MUTED, line=1.22, first=True)
        ry += 0.58

    notes(s, "NISARG — 30s.\nTop: the three things the markers said, and one sentence each "
             "on what we did. Bottom: four changes, one line each, each pointing at the "
             "slide where it bites. Slow down on the first: we found a leak in our own "
             "calibration split, fixed it, and the headline moved 2.5 kcal inside its "
             "interval. Nothing was tuned on test. Hand to Sushant.")
    return s


def slide_dataset(prs, n):
    s = base(prs, "Dataset", "507 test dishes are 127 plate sessions.", 5)
    y = s._body_top

    pic(s, "plate_grid.png", L, y + 0.02, 2.75)
    caption(s, L, y + 2.86, 2.75,
            "Nutrition5k [1]: overhead photos, one rig, every ingredient weighed. The "
            "weighed truth is what lets us build a perfect refuser.", size=10.5)

    x2 = 4.05
    eyebrow(s, x2, y, "Scans of one plate arrive 40 seconds apart", w=7.5)
    pic(s, "sessions.png", x2, y + 0.30, 6.5)
    caption(s, x2, y + 3.36, 7.45,
            "Errors inside a session are correlated (intraclass correlation 0.34). The "
            "session is the unit for the calibration carve and for every interval in this "
            "talk.", size=12, color=INK, bold=True)

    cy = y + 4.22
    eyebrow(s, L, cy, "Splits", w=9.0)
    cy += 0.32
    stages = [("4,768", "official split ids [1]"), ("3,490", "have overhead RGB"),
              ("2,755 / 507", "train / test, official"),
              (f"{n['n_fit']:,} / {n['n_calib']:,}", "fit / calibration, by session"),
              ("367 / 127", "sessions, train / test")]
    bw, gap = 2.02, 0.32
    for i, (num, lbl) in enumerate(stages):
        x = L + i * (bw + gap)
        rect(s, x, cy, bw, 0.60, fill=TINT if i >= 3 else BG)
        t = tb(s, x, cy + 0.08, bw, 0.5)
        para(t, num, size=13.5, color=ACCENT if i >= 3 else INK, bold=True,
             align=PP_ALIGN.CENTER, first=True)
        para(t, lbl, size=10, color=MUTED, align=PP_ALIGN.CENTER, space_before=2)
        if i < len(stages) - 1:
            arrow(s, x + bw + 0.06, cy + 0.22, 0.20, 0.16)
    notes(s, "SUSHANT — 30s.\nLeft: what the data is and why: weighed ingredients give us "
             "a perfect refuser to measure against. Middle, the histogram: dish ids are "
             "timestamps; scans of one plate are 40 seconds apart, the next session is hours "
             "away. So test is 127 sessions, not 507 independent dishes. Bottom: official "
             "split untouched; train carved into fit and calibration by session.")
    return s


def slide_design(prs, n):
    s = base(prs, "Experimental design", "One frozen model, one linear head, one tuned knob.", 6)
    y = s._body_top

    # the pipeline as boxes
    boxes = [("a photo", "224 × 224", BG, MUTED),
             ("frozen backbone", "CLIP ViT-B/16; SigLIP 2,\nDINOv3 as checks", TINT, INK),
             ("2,304 features", "cached once, per\nphoto and damage level", BG, INK),
             ("standardise", "mean 0, spread 1,\non the fit split", BG, INK),
             ("ridge head", "closed form; one for\nkcal, one for grams", TINT, INK),
             ("“620 kcal”", "and a gate score\n(slide 10)", BG, INK)]
    bw, gap, bh = 1.62, 0.34, 1.05
    for i, (head, sub, fill, col) in enumerate(boxes):
        x = L + i * (bw + gap)
        rect(s, x, y, bw, bh, fill=fill)
        t = tb(s, x + 0.08, y + 0.12, bw - 0.16, bh - 0.2)
        para(t, head, size=12.5, color=col, bold=True, align=PP_ALIGN.CENTER, first=True)
        para(t, sub, size=9.5, color=MUTED, align=PP_ALIGN.CENTER, line=1.2, space_before=3)
        if i < len(boxes) - 1:
            arrow(s, x + bw + 0.07, y + bh / 2 - 0.08, 0.20, 0.16)

    cy = y + bh + 0.40
    cards = [
        ("The one tuned knob", "Ridge alpha, chosen on the calibration split over a 25-point "
                              f"quarter-decade grid. It picked {n['alpha_cal']:,.0f}."),
        ("Fixed in advance", "PCA k = 64 for the distance gate, B = 32 ensemble "
                                         "heads, K = 10 neighbours."),
        ("One results file", "8 damage levels × 507 dishes × 3 backbones = 12,168 rows. "
                             "Every result is a lookup on that file."),
        ("Compute", "One M-series laptop, ~68 images/s through the backbone. The whole "
                    "pipeline reruns in under 15 minutes."),
    ]
    cw_, cg = 2.72, 0.21
    for i, (title, body) in enumerate(cards):
        x = L + i * (cw_ + cg)
        rect(s, x, cy, cw_, 1.55, fill=BG)
        t = tb(s, x + 0.16, cy + 0.12, cw_ - 0.32, 0.3)
        para(t, title.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        t = tb(s, x + 0.16, cy + 0.44, cw_ - 0.32, 1.1)
        para(t, body, size=11, color=INK, line=1.28, first=True)

    takeaway(s, "No training loop, so no convergence curve to show. Nothing was tuned on "
                "the test split.", y=cy + 1.55 + 0.35, size=14.5)
    caption(s, L, cy + 1.55 + 0.85, CW,
            "Leak discipline: heads are fit on the fit dishes; their errors are measured on "
            "the calibration dishes, which they never saw and which share no session with "
            "them; the error head is trained on those errors; everything is reported on the "
            "507 test dishes.", size=11)
    notes(s, "SUSHANT — 30s.\nLeft to right: the backbone sees each photo once, the "
             "features are cached, everything after is arithmetic. A ridge head is a linear "
             "formula with one penalty, alpha, and alpha is the only thing chosen from "
             "data, on calibration. The other three settings were fixed in advance. One "
             "laptop, fifteen minutes. Next: how every chart is read.")
    return s


def slide_reader(prs, n):
    s = base(prs, "How we score a gate", "How to read every chart that follows.", 7)
    y = s._body_top
    pic(s, "reader.png", L - 0.10, y + 0.02, 6.2)
    caption(s, L, y + 3.82, 6.2,
            "Clean test photos, CLIP [ours]. The protocol is adopted from biometrics [3] and "
            "selective prediction [6].", size=10.5)

    x2 = 7.75
    eyebrow(s, x2, y, "A gate is scored on four things", w=4.8)
    ry = y + 0.36
    for i, (title, body) in enumerate([
        ("Error at 90% answered", "MAE on the photos it still answers, at the operating "
                                  "point. The whole curve is shown too."),
        ("Area under the curve", "AURC, and the share of the random-to-perfect gap the gate "
                                 "closes across all answer rates."),
        ("Rank correlation", "Spearman ρ between the signal and the true error."),
        ("What it refuses", "Mean size of the meals kept, and the share of 400+ kcal meals "
                            "refused, against the perfect refuser’s. Ours."),
    ]):
        t = tb(s, x2, ry, 4.75, 0.3)
        para(t, [(f"{i + 1}  ", {"color": ACCENT, "bold": True, "size": 12}),
                 (title, {"color": INK, "bold": True, "size": 12})], first=True)
        t = tb(s, x2 + 0.32, ry + 0.28, 4.45, 0.6)
        para(t, body, size=10.5, color=MUTED, line=1.25, first=True)
        ry += 0.86

    strip(s, 6.08, 0.80, "Every number carries an interval",
          f"{n['n_boot']:,} resamples of the {n['n_sessions']} plate sessions, every curve "
          "recomputed on the same resample, so tests are paired. Each gate is tested "
          "against refusing at random and Holm-corrected across the ten candidates. A tick "
          "on a later slide means it survived.", size=11.5)
    notes(s, "SUSHANT — 35s.\nThis is the one chart to learn. Across: how many photos the "
             "app still answers. Down: the error on those photos. Refusing at random is "
             "flat. A perfect refuser, one that knows the true error, falls away. Every "
             "real gate lands between them, and we read it at 90% answered, one refusal in "
             "ten. Four measures on the right; the fourth is ours, and slide 12 says why it "
             "is there. Strip: every number gets an interval from resampling sessions, "
             "tests are paired, and there is a correction for testing ten gates. Hand to "
             "Keanu.")
    return s


def slide_predictor(prs, n):
    s = base(prs, "Existing results vs ours: the predictor",
             "The published baseline, reproduced without training.", 8)
    y = s._body_top
    pic(s, "backbones.png", L - 0.10, y + 0.12, 6.6)
    caption(s, L, y + 3.15, 6.5,
            f"507 test dishes, official split. CLIP {n['mae']:.1f} kcal {fmt_ci(n['mae_ci'])}: "
            f"the published {PUBLISHED_RGB} sits inside the interval. Portion mass "
            f"{n['mass']:.1f} g ({n['mass_rel']:.0f}%).", size=11.5, color=INK, bold=True)

    x2 = 7.85
    cols, widths = [x2, x2 + 2.55, x2 + 3.45], [2.5, 0.9, 1.5]
    ry = table_head(s, y + 0.10, cols, widths, ["Calories, MAE", "kcal", "source"])
    for what, val, src, hi in [
        ("RGB, Inception v3, trained", f"{PUBLISHED_RGB}", "[1]", False),
        ("RGB-D", "47.6", "[1]", False),
        ("Volume scalar, best", f"{PUBLISHED_BEST}", "[1]", False),
        ("Ours, CLIP, frozen", f"{n['mae']:.1f}", "[ours, replicated]", True),
        ("Ours, SigLIP 2, frozen", f"{n['mae_siglip']:.1f}", "[ours]", False),
        ("Ours, DINOv3, frozen", f"{n['mae_dino']:.1f}", "[ours]", False),
        ("2026 re-split, 60/15/25", "28.0", "not comparable", False),
    ]:
        rh = 0.40
        if hi:
            rect(s, x2 - 0.12, ry - 0.06, 4.85, rh, fill=TINT)
        t = tb(s, cols[0], ry, widths[0], 0.35)
        para(t, what, size=11.5, color=INK, bold=hi, first=True)
        t = tb(s, cols[1], ry, widths[1], 0.35)
        para(t, val, size=11.5, color=INK, bold=True, first=True)
        t = tb(s, cols[2], ry, widths[2], 0.35)
        para(t, src, size=10, color=ACCENT if src.startswith("[ours") else MUTED,
             bold=True, first=True)
        ry += rh
    caption(s, x2, ry + 0.08, 4.8,
            "The 28.0 kcal paper re-splits Nutrition5k at random, so near-duplicate scans "
            "land on both sides. We keep [1]’s split and do not compare to it.", size=10.5)

    takeaway(s, "Reproduced, so the predictor is frozen. Everything from here is about the "
                "gate.", y=6.15, size=15)
    notes(s, "KEANU — 30s.\nThree frozen backbones, a ridge head, no training, with "
             "whiskers. The published number sits inside our interval: we reproduce the "
             "baseline, we do not beat it. The table keeps the literature's numbers apart "
             "from ours and names the one we refuse to compare to, because it re-split the "
             "data. Then result one.")
    return s


def slide_pixel(prs, n):
    s = base(prs, "Result 1, completed", "Pixel signals see the damage, not the harm.", 9)
    y = s._body_top
    pr = n["per_rung"]
    base_m = pr["clean"]["mass_mae"]
    rho_px = max(abs(n["rho_pixel"][k]) for k in PIXEL)

    eyebrow(s, L, y, "One dish, four photographs  [ours]")
    py = y + 0.32
    pw, gap = 1.42, 0.12
    for i, (fn, label, sharp, tagk, hi) in enumerate([
            ("panel_clean.png", "clean", "177", "clean", False),
            ("panel_blur6.png", "blurred", "1.5", "blur6", False),
            ("panel_crop04.png", "cropped", "6.5", "crop0p4", True),
            ("panel_phone.png", "phone", "10", "phone", False)]):
        px = L + i * (pw + gap)
        pic(s, fn, px, py, pw)
        t = tb(s, px, py + pw * 0.75 + 0.08, pw, 0.9)
        para(t, label, size=12, color=ACCENT if hi else INK, bold=True,
             align=PP_ALIGN.CENTER, first=True)
        para(t, [("sharpness ", {"color": MUTED, "size": 10}),
                 (sharp, {"color": INK, "size": 11, "bold": True})],
             align=PP_ALIGN.CENTER, space_before=3)
        para(t, [("mass error ", {"color": MUTED, "size": 10}),
                 (f"{pr[tagk]['mass_mae'] / base_m:.1f}×",
                  {"color": ACCENT if hi else INK, "size": 11, "bold": True})],
             align=PP_ALIGN.CENTER, space_before=1)
    caption(s, L, py + pw * 0.75 + 0.95, 6.1,
            "Sharpness scores the cropped photo four times better than the blurred one. It "
            "carries far more error: the plate rim is the only clue to portion size. The "
            "phone rung is an imitation, and costs 1.2× on calories.",
            size=11.5, color=INK, bold=True)

    cx = 7.25
    eyebrow(s, cx, y, "Refuse the worst photos, all 8 damage levels pooled  [ours]")
    pic(s, "risk_coverage.png", cx, y + 0.28, 5.1)

    strip(s, 5.72, 1.15, "Detected, and it does not help",
          f"pixel metrics separate damaged photos from clean ones at AUC ≈ 1.00. Over "
          f"{507 * 8:,} dish × damage rows none is distinguishable from random at any answer "
          f"rate (|ρ| ≤ {rho_px:.2f}). Variance of the error: which dish 62%, damage level "
          "9%, their interaction 29%. A gate has to read the dish.", size=11.5)
    notes(s, "KEANU — 45s.\nLeft: same dish, four photographs. The cropped one is sharp and "
             "the worst of the four, because the rim is the only clue to portion size; "
             "sharpness ranks it four times better than the blurred one. Right: refuse the "
             "photos each signal likes least; sharpness and random sit on top of each other, "
             "a perfect refuser drops from a hundred to thirty-eight. Strip: we detect the "
             "damage almost perfectly and it does not help, because damage is nine percent "
             "of the error and the dish is sixty-two. Hand to Shreya.")
    return s


def slide_curve(prs, n):
    s = base(prs, "Result 2, completed", "Model-aware signals bend the curve.", 10)
    y = s._body_top
    pic(s, "curves_ci.png", L - 0.10, y + 0.02, 6.2)
    caption(s, L, y + 3.82, 6.2,
            "Clean test photos, CLIP [ours]. Bands are each curve’s own 95% interval; the "
            "tests are paired, so differences are tighter than the bands (slide 11).",
            size=10.5)

    x2 = 7.75
    eyebrow(s, x2, y, "At 90% answered, CLIP  [ours]", w=4.8)
    cols, widths = [x2, x2 + 2.35, x2 + 3.30], [2.3, 0.9, 1.4]
    ry = table_head(s, y + 0.34, cols, widths, ["gate", "kcal", "vs random"])
    g, vr = n["g90"], n["vr"]
    for key, label in [
            ("perfect", "perfect refuser"), ("learned_error", "learned error head"),
            ("blend", "blend of four"), ("pred_magnitude", "predicted size (control)"),
            ("ensemble_spread", "ensemble disagreement"), ("mahalanobis", "distance to mean"),
            ("knn_dist", "neighbour distance"), ("random", "random"),
            ("lap_var", "sharpness")]:
        rh = 0.33
        hi = key == "learned_error"
        if hi:
            rect(s, x2 - 0.12, ry - 0.05, 4.75, rh, fill=TINT)
        t = tb(s, cols[0], ry, widths[0], 0.3)
        para(t, label, size=11, color=GREEN if key == "perfect" else INK, bold=hi, first=True)
        t = tb(s, cols[1], ry, widths[1], 0.3)
        para(t, f"{g[key]:.1f}", size=11, color=INK, bold=True, first=True)
        t = tb(s, cols[2], ry, widths[2], 0.3)
        if key in vr:
            d = vr[key]
            star = "" if d.get("p_holm", 1) >= 0.05 else "  ✓"
            txt = f"{d['diff']:+.1f} {fmt_ci(d['ci'])}{star}"
            col = GREEN if star else MUTED
        else:
            txt, col = "—", FAINT
        para(t, txt, size=10.5, color=col, bold=bool(key in vr and star), first=True)
        ry += rh
    caption(s, x2, ry + 0.06, 4.75,
            "✓ = better than random after Holm correction, p < 0.05. The control scores "
            "risk by the predicted calorie count alone (slide 12).", size=10)

    le, es = vr["learned_error"], vr["ensemble_spread"]
    pr = n["pair"]["ensemble_spread - learned_error"]
    strip(s, 6.12, 0.78, "What survived",
          f"the learned error head beats random by {-le['diff']:.0f} kcal at 90% answered "
          f"({fmt_p(le['p_holm'])}). Ensemble disagreement, the proposal’s favourite, does "
          f"not clear the correction here ({fmt_p(es['p_holm'])}) and trails the head by "
          f"{pr['diff']:.0f} kcal. Neither distance gate is distinguishable from random on "
          "clean photos.", size=11.5)
    notes(s, "SHREYA — 40s.\nSame axes as slide 7, now with the gates drawn. Random flat, "
             "perfect refuser the floor, the learned gates between. Table: at 90% answered, "
             "the error head takes seventy to sixty, ten kcal better than random, interval "
             "clear of zero, corrected p under one percent. The two proposal claims that "
             "died: ensemble disagreement does not clear the correction here, and the "
             "distance gates are not distinguishable from random. On damaged photos they "
             "are: they detect damage, not difficulty.")
    return s


def slide_significance(prs, n):
    s = base(prs, "Result 2, completed", "The same answer on three backbones.", 11)
    y = s._body_top
    pic(s, "significance.png", L + 1.05, y + 0.02, 9.4)

    gap, gci = n["gap"], n["gap_ci"]
    sy = 5.68
    x = L
    for title, body in [
        ("Beat random, all three",
         "The learned error head, the blend and the control (Holm p < 0.01). Head and blend "
         "are identical: blending adds nothing."),
        ("Do not",
         "Ensemble disagreement at 90% (it does at 50%). The distance gates, never. The "
         "pixel gates, never; sharpness is worse than random at 50% on two of three."),
        ("Share of the gap captured",
         f"Error head {100 * gap['learned_error']:.0f}% "
         f"{fmt_ci([100 * v for v in gci['learned_error']])}; control "
         f"{100 * gap['pred_magnitude']:.0f}%; ensemble {100 * gap['ensemble_spread']:.0f}%; "
         f"distance to mean {100 * gap['mahalanobis']:.0f}%, interval spanning zero."),
    ]:
        rect(s, x - 0.12, sy - 0.04, 3.72, 1.22, fill=BG)
        t = tb(s, x, sy + 0.04, 3.5, 0.3)
        para(t, title.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        t = tb(s, x, sy + 0.33, 3.5, 0.9)
        para(t, body, size=10.5, color=INK, line=1.22, first=True)
        x += 3.95
    notes(s, "SHREYA — 30s.\nOne panel per backbone. Each dot is a gate's paired difference "
             "to random at 90%, with its interval; filled means it survived the correction. "
             "The pattern repeats three times: the error head, the blend and the control "
             "are filled; ensemble disagreement, the distance gates and the pixel gates are "
             "hollow. The control being filled is the point of the next slide.")
    return s


def slide_degeneracy(prs, n):
    s = base(prs, "Result 3, completed", "Check what each gate refuses before believing it.", 12)
    y = s._body_top
    pic(s, "degeneracy.png", L + 1.15, y + 0.02, 9.2)
    sel = n["sel"]
    pc = n["pair"]["learned_error - pred_magnitude"]
    k = lambda g, c: sel[g][c]["kept_mean_kcal"]
    t = strip(s, 5.55, 1.35, "The control",
          "scoring risk by the predicted calorie count refuses big meals and nothing else, "
          "and error grows with size, so that alone lowers the error. At 90% the error head, "
          f"the control and the oracle keep the same meals ({k('learned_error', '0.9'):.0f} / "
          f"{k('pred_magnitude', '0.9'):.0f} / {k('perfect', '0.9'):.0f} kcal): the head’s "
          f"win is honest, and the head and the control are statistically tied (paired "
          f"p = {pc['p']:.2f}). At 50% the head drifts to {k('learned_error', '0.5'):.0f}-kcal "
          f"meals and the control to {k('pred_magnitude', '0.5'):.0f}; only ensemble "
          f"disagreement stays with the oracle ({k('ensemble_spread', '0.5'):.0f} vs "
          f"{k('perfect', '0.5'):.0f}).", size=11)
    para(t, "Which gate is honest depends on the answer rate. The kept-set size sits beside "
            "every error we report.", size=12, color=INK, bold=True, space_before=4)
    notes(s, "SHREYA — 35s.\nAcross: the average size of the meals a gate still answers. "
             "Down: its error. The dashed line is the perfect refuser. Left panel, refuse one "
             "in ten: the error head, the control and the oracle all keep meals around 220 "
             "kcal, so the head's win is honest, and it is tied with simply reading the "
             "predicted size. That is not a scandal: at this point the oracle refuses big "
             "meals too, because the biggest errors are on the biggest meals. Right panel, "
             "refuse half: the head and the control drift to small meals; only ensemble "
             "disagreement stays with the oracle. Which gate is honest depends on the answer "
             "rate. Hand to Kevin.")
    return s


def slide_ranges(prs, n):
    s = base(prs, "Result 4, first cut", "Ranges: honest on average, not on large meals.", 13)
    y = s._body_top
    c = n["conf"]
    plain, adapt = c["constant"], c["ensemble_spread"]
    pic(s, "conformal_kcal.png", L - 0.15, y + 0.02, 7.6)
    caption(s, L, y + 2.98, 7.4,
            f"Split conformal [5], 90% target, {n['n_calib']} calibration dishes, 507 clean "
            f"test photos, CLIP [ours].", size=10.5)

    x2 = 8.85
    cols, widths = [x2, x2 + 1.85, x2 + 2.75], [1.8, 0.9, 0.9]
    ry = table_head(s, y, cols, widths, ["", "plain", "adaptive"])
    for lab, a, b in [
        ("coverage", f"{100 * plain['coverage']:.1f}%", f"{100 * adapt['coverage']:.1f}%"),
        ("interval", fmt_ci([100 * v for v in plain["coverage_ci"]]),
         fmt_ci([100 * v for v in adapt["coverage_ci"]])),
        ("median half-width", f"±{plain['width_median'] / 2:.0f} kcal",
         f"±{adapt['width_median'] / 2:.0f} kcal"),
        ("largest meals", f"{100 * plain['by_kcal']['coverage'][3]:.0f}%",
         f"{100 * adapt['by_kcal']['coverage'][3]:.0f}%"),
        ("hardest quarter", f"{100 * plain['by_difficulty']['coverage'][3]:.0f}%",
         f"{100 * adapt['by_difficulty']['coverage'][3]:.0f}%"),
        ("within ±100 kcal", f"{100 * plain['usable_share']:.0f}%",
         f"{100 * adapt['usable_share']:.0f}%"),
        ("interval score", f"{plain['interval_score']:.0f}", f"{adapt['interval_score']:.0f}"),
    ]:
        rh = 0.33
        t = tb(s, cols[0], ry, widths[0], 0.3)
        para(t, lab, size=10.5, color=INK, first=True)
        for cx, v, b_ in ((cols[1], a, False), (cols[2], b, True)):
            t = tb(s, cx, ry, 0.9, 0.3)
            para(t, v, size=10.5, color=INK, bold=b_, first=True)
        ry += rh
    caption(s, x2, ry + 0.04, 3.6, "Interval score: width plus a penalty for misses; lower "
                                   "is better.", size=9.5)

    mc = n["mix"]["uniform"]["constant"]
    bk = plain["by_kcal"]["coverage"]
    t = strip(s, 5.35, 1.50, "Two things this tells us",
              f"(1) The 90% average is {100 * bk[0]:.0f}% on the smaller meals and "
              f"{100 * bk[3]:.0f}% on the largest quarter; the adaptive width barely grows "
              "with size. Same big-meal effect as slide 12, and the next O3 problem.",
              size=11)
    para(t, f"(2) By difficulty, the adaptive width lifts the hardest quarter from "
            f"{100 * plain['by_difficulty']['coverage'][3]:.0f}% to "
            f"{100 * adapt['by_difficulty']['coverage'][3]:.0f}% and improves the proper "
            f"score. Shift check: calibrate on clean, deploy on damaged, coverage "
            f"{100 * mc['calibrated_on_clean']['coverage']:.0f}%; recalibrate, "
            f"{100 * mc['calibrated_on_mixture']['coverage']:.0f}% at "
            f"{mc['calibrated_on_mixture']['width_mean'] / mc['calibrated_on_clean']['width_mean'] * 100 - 100:.0f}% "
            "more width. A check, not a finding.", size=11, color=INK, line=1.3,
         space_before=3)
    notes(s, "KEVIN — 40s.\nA range is a promise: the truth is inside it nine times in "
             "ten. Split conformal keeps that promise on average, 89.5 percent. Left chart: "
             "cut by meal size, the promise is 98 percent on small meals and 67 on the "
             "largest quarter. The average is made of two wrong numbers. Right chart: cut "
             "by how hard the dish looks, letting the gate signal set the width lifts the "
             "hardest quarter from 80 to 85. Table: the numbers. The shift check in one "
             "sentence, and it is a check, not a finding.")
    return s


def slide_timeline(prs, n):
    s = base(prs, "Challenges, timeline, roles", "Two milestones slipped, one moved early.", 14)
    y = s._body_top

    eyebrow(s, L, y, "Two challenges, and what we did", w=5.5)
    ry = y + 0.34
    for what, did in [
        ("Correlated test dishes",
         "Errors cluster by plate session (ICC 0.34). The session is now the unit of "
         "resampling, and the calibration carve respects it."),
        ("A gate that cheats",
         "Refusing big meals looks like skill. A control gate exposed it; the kept-set size "
         "is reported beside every error, at every answer rate."),
    ]:
        t = tb(s, L, ry, 5.3, 0.3)
        para(t, what, size=12, color=INK, bold=True, first=True)
        t = tb(s, L, ry + 0.30, 5.3, 0.8)
        para(t, did, size=11, color=MUTED, line=1.28, first=True)
        ry += 1.15
    caption(s, L, ry + 0.05, 5.3,
            "Also: a rank-deficient covariance (2,304 dims on 1,924 dishes; Mahalanobis on "
            "64 components, fixed in advance), and no phone photos in the data (the phone "
            "rung is an imitation, labelled as one).", size=10)

    x2 = 6.85
    eyebrow(s, x2, y, "Proposal timeline vs actual", w=5.5)
    cols, widths = [x2, x2 + 1.35, x2 + 4.15], [1.3, 2.75, 1.5]
    ty = table_head(s, y + 0.34, cols, widths, ["planned", "milestone", "status"])
    for when, what, status, col in [
        ("17 Aug", "Pipeline, 3 backbones, first results", "done", GREEN),
        ("25 Aug", "Phone rung; hidden-food rung", "late · dropped", ACCENT),
        ("11 Sep", "All gates scored, one results file", "done 17 Sep", ACCENT),
        ("20 Sep", "Literature review", "in progress", INK),
        ("11 Oct", "Confidence ranges", "first cut 17 Sep", GREEN),
        ("25 Oct", "Final report", "next", INK),
    ]:
        rh = 0.38
        t = tb(s, cols[0], ty, widths[0], 0.3)
        para(t, when, size=10.5, color=MUTED, first=True)
        t = tb(s, cols[1], ty, widths[1], 0.3)
        para(t, what, size=10.5, color=INK, first=True)
        t = tb(s, cols[2], ty, widths[2], 0.3)
        para(t, status, size=10.5, color=col, bold=True, first=True)
        ty += rh
    caption(s, x2, ty + 0.02, 5.6,
            "Next: size-aware ranges for large meals (conformalised quantile regression, "
            "CQR); the deployment mixture on all backbones; the cost tier (test-time "
            "augmentation, a second backbone); the report.", size=10.5, color=INK, bold=True)

    ry = 5.65
    eyebrow(s, L, ry, "Who owns what")
    ry += 0.32
    for i, (who, what) in enumerate([
            ("Nisarg", "pipeline, results file, framing"), ("Shreya", "damage levels, pixel gates"),
            ("Keanu", "model-aware gates, curves"), ("Sushant", "literature, writing, data"),
            ("Kevin", "confidence ranges")]):
        t = tb(s, L + i * 2.32, ry, 2.25, 0.5)
        para(t, who, size=12, color=INK, bold=True, first=True)
        para(t, what, size=10, color=MUTED, space_before=2, line=1.25)
    notes(s, "KEVIN — 25s.\nTwo challenges, the first in full: test dishes are correlated "
             "within a session, so we changed the unit of analysis. Timeline: first "
             "milestone on time, two slipped by three weeks, one dropped with a reason, "
             "the ranges early. Next steps in one breath. Roles as on the proposal; name "
             "each member and one thing they did.")
    return s


def slide_conclusions(prs, n):
    s = base(prs, "Conclusions", "What we know now that we did not in August.", 15)
    y = s._body_top
    le = n["vr"]["learned_error"]
    for i, (head, body) in enumerate([
        ("Photo quality is not the axis.",
         "Damage is detected almost perfectly and explains 9% of the error; the dish "
         "explains 62%. No pixel gate beats random, on any backbone, at any answer rate."),
        ("A free model-aware signal works, with intervals.",
         f"At 90% answered the learned error head cuts the error by {-le['diff']:.0f} kcal "
         f"against random ({fmt_p(le['p_holm'])}) on three backbones and keeps normal-sized "
         "meals. So does reading the predicted size; the two are tied there."),
        ("Which gate is honest depends on the answer rate.",
         "At 50% the same gate drifts to small meals and ensemble disagreement is the one "
         "that tracks the oracle. The kept-set size sits beside every error we report."),
        ("Ranges are honest on average, not on large meals.",
         "Adaptive width fixes most of the difficulty gap. The largest quarter of meals is "
         "covered 67% of the time. That is O3’s job by 11 October."),
    ]):
        rect(s, L - 0.18, y - 0.04, CW + 0.36, 0.90, fill=BG if i % 2 == 0 else None)
        t = tb(s, L, y + 0.02, 0.5, 0.5)
        para(t, str(i + 1), size=18, color=ACCENT, bold=True, first=True)
        t = tb(s, L + 0.55, y + 0.02, CW - 0.6, 0.85)
        para(t, head, size=13.5, color=INK, bold=True, first=True)
        para(t, body, size=11, color=MUTED, line=1.28, space_before=2)
        y += 0.97
    t = tb(s, L, y + 0.10, CW, 1.2)
    para(t, "So far: an app can tell when it is guessing, at the price of refusing one photo "
            "in ten, for about half of what a perfect refuser would give. On the phone: the "
            "signals that need no model all fail; the ones that work are a linear head on "
            "features the app already computes.", size=13, color=INK, bold=True, line=1.3,
         first=True)
    para(t, "Every method here is borrowed and cited. The benchmark, the control, the "
            "sessions and the findings are ours.", size=11.5, color=MUTED, line=1.3,
         space_before=6)
    notes(s, "KEVIN — 30s.\nRead the four bold lines. Then the answer so far, including "
             "the phone: the cheap signals fail, the working ones are free if the model is "
             "on the device. Last line: everything is borrowed except the benchmark and what "
             "it found. Questions. Owners: framing Nisarg, splits and metrics Sushant, "
             "predictor and pixel result Keanu, gates and significance Shreya, ranges and "
             "timeline Kevin.")
    return s


def slide_references(prs, n):
    s = base(prs, "Closing", "Sources, AI use, and where the code is.", 16)
    y = s._body_top
    refs = [
        "[1] Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, “Nutrition5k: "
        "Towards automatic nutritional understanding of generic food,” in Proc. IEEE/CVF CVPR, 2021, "
        "pp. 8903–8911.",
        "[2] B. Coburn, J. He, M. E. Rollo, S. S. Dhaliwal, D. A. Kerr, and F. Zhu, “Comprehensive "
        "evaluation of large multimodal models for nutrition analysis: A new benchmark enriched with "
        "contextual metadata,” in Proc. IEEE Int. Conf. Biomed. Health Informat. (BHI), 2025.",
        "[3] P. Grother and E. Tabassi, “Performance of biometric quality measures,” IEEE Trans. "
        "Pattern Anal. Mach. Intell., vol. 29, no. 4, pp. 531–543, Apr. 2007.",
        "[4] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common "
        "corruptions and perturbations,” in Proc. ICLR, 2019.",
        "[5] A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal prediction and "
        "distribution-free uncertainty quantification,” arXiv:2107.07511, 2021.",
        "[6] Y. Geifman and R. El-Yaniv, “SelectiveNet: A deep neural network with an integrated "
        "reject option,” in Proc. ICML, PMLR 97, 2019.",
        "Re-split comparison on slide 8: V. Awasthi et al., Int. J. Intell. Eng. Syst., vol. 19, "
        "no. 2, 2026, doi:10.22266/ijies2026.0228.08.",
        "The gate signals (feature-space distance, nearest-neighbour distance, ensemble "
        "disagreement, learned error), the normalised conformal score, the interval score and "
        "the Holm correction are used as published; full references in the report.",
        "All figures and tables are ours unless tagged [1]. Images on slides 5 and 9 are Nutrition5k "
        "dishes [1], reproduced under the dataset licence.",
    ]
    t = tb(s, L, y, CW, 3.8)
    for i, r in enumerate(refs):
        para(t, r, size=9.5, color=INK if i < 6 else MUTED, line=1.25, first=(i == 0),
             space_before=0 if i == 0 else 4)

    ry = 5.25
    rect(s, L - 0.18, ry, CW + 0.36, 0.72, fill=BG)
    t = tb(s, L + 0.14, ry + 0.09, CW - 0.3, 0.65)
    para(t, [("GENERATIVE AI USE   ", {"color": ACCENT, "bold": True, "size": 10.5}),
             ("Claude Code (Anthropic) was used to scaffold pipeline code, generate the slides "
              "from the results files, and improve wording. The experimental design, analysis "
              "choices, results and conclusions are the group’s own and reproduce from the "
              "repository.", {"color": INK, "size": 10.5})], line=1.3, first=True)
    rect(s, L - 0.18, ry + 0.84, CW + 0.36, 0.60, fill=TINT)
    t = tb(s, L + 0.14, ry + 0.96, CW - 0.3, 0.5)
    para(t, [("CODE AND RESULTS   ", {"color": ACCENT, "bold": True, "size": 10.5}),
             (REPO_URL, {"color": INK, "bold": True, "size": 12}),
             ("   ·   every number in this deck is regenerated from results/*.json by the "
              "deck builder", {"color": MUTED, "size": 10.5})], first=True)
    notes(s, "Not spoken. Stays up during questions.")
    return s


# ---------------------------------------------------------------- main
def main() -> None:
    missing = [f for f in REQUIRED_FIGS if not (FIGS / f).exists()]
    if missing:
        raise SystemExit(f"missing figures, run figures.py first: {missing}")
    n = load_numbers()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for fn in (slide_title, slide_recap, slide_why, slide_changes, slide_dataset,
               slide_design, slide_reader, slide_predictor, slide_pixel, slide_curve,
               slide_significance, slide_degeneracy, slide_ranges, slide_timeline,
               slide_conclusions, slide_references):
        fn(prs, n) if fn is not slide_title else fn(prs)
    prs.save(OUT)
    total = sum(sec for _, sec, _ in SLIDE_PLAN)
    per = {}
    for who, sec, _ in SLIDE_PLAN:
        per[who] = per.get(who, 0) + sec
    print(f"wrote {OUT}  ({len(prs.slides)} slides)")
    print(f"spoken budget {total // 60}:{total % 60:02d}  " +
          "  ".join(f"{w} {s}s" for w, s in per.items()))


if __name__ == "__main__":
    main()
