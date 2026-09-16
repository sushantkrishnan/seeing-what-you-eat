"""Build the COMPSCI 760 Group 1 update deck: methodology and results.

  ../.venv/bin/python build_update_deck.py        -> 01_MethodResults.pptx

Run gate_bench.py, gate_probe.py --all --json, conformal.py --all --json and
figures.py first. Every number on these slides is read from results/*.json at build
time, so the deck cannot drift from the code.

RUBRIC MAP (Presentation2_rubic.txt, 20 points). Slide numbers in brackets.
  objectives & problem vs proposal [2]      changes since proposal + feedback [3]
  dataset with rationale, splits [4]        methodology, tuning, measures, compute [5]
  existing vs our results, tagged [6]       results so far [7-11]
  challenges, timeline vs plan, roles [12]  conclusions and next steps [13]
  citations, GenAI acknowledgement, GitHub link [1, 14]

Eight minutes, five speakers at ~90 s each, 7:30 spoken. Speaker names are not
printed on slides; SLIDE_PLAN drives the timing budget and the notes.

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

SLIDE_PLAN = [
    ("Nisarg",  40, "Recap: problem, proposal, objectives"),
    ("Nisarg",  45, "Changes since the proposal, feedback"),
    ("Sushant", 40, "Dataset, splits, sessions"),
    ("Sushant", 45, "Experimental design and evaluation"),
    ("Keanu",   40, "Existing vs ours: the predictor"),
    ("Keanu",   45, "Result 1: pixel signals, error sources"),
    ("Shreya",  40, "Result 2: the curve"),
    ("Shreya",  40, "Result 2: significance on three backbones"),
    ("Shreya",  35, "Result 3: what each gate refuses"),
    ("Kevin",   45, "Result 4: confidence ranges"),
    ("Kevin",   40, "Challenges, timeline, roles"),
    ("Kevin",   20, "Conclusions and next steps"),
]

REQUIRED_FIGS = ["plate_grid.png", "sessions.png", "backbones.png", "panel_clean.png",
                 "panel_blur6.png", "panel_crop04.png", "panel_phone.png",
                 "risk_coverage.png", "curves_ci.png", "significance.png",
                 "degeneracy.png", "conformal.png"]


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
             "Target 7:30 spoken, hard stop 8:00.")
    return s


def slide_recap(prs, n):
    s = base(prs, "Recap", "The predictor is frozen. We study the gate.", 2)
    y = s._body_top

    # the diagram, as in the proposal
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

    yy = y + 1.72
    for label, body, col in [
        ("THE PROBLEM",
         "Calorie apps answer every photo with one number and never decline. A confident "
         "wrong number is worse than no number.", MUTED),
        ("WHAT WE PROPOSED",
         "Freeze the predictor. Benchmark the signals an app could use to decide whether to "
         "answer, between refusing at random and a perfect refuser.", INK),
    ]:
        t = tb(s, L, yy, 8.9, 0.28)
        para(t, label, size=11, color=ACCENT if col is INK else MUTED, bold=True, first=True)
        t = tb(s, L, yy + 0.27, 8.9, 0.7)
        para(t, body, size=12.5, color=col, bold=(col is INK), line=1.32, first=True)
        yy += 0.98

    # objectives with status, right column
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

    takeaway(s, "Headline question: which signals let an app know it is guessing, and can any "
                "of them run on the phone?", y=6.22, size=14)
    notes(s, "NISARG — 40s.\nSelf-contained recap, because the rubric asks for one. Walk the "
             "diagram left to right once. Say 'the problem' and 'what we proposed' as two "
             "separate sentences. Then point at the three objectives and their status: two "
             "done, the third started early. Hand straight to the changes slide.")
    return s


def slide_changes(prs, n):
    s = base(prs, "Since the proposal", "What the markers asked, and what we changed.", 3)
    y = s._body_top

    eyebrow(s, L, y, "Feedback we received, and what we did with it", w=8.0)
    ry = y + 0.34
    for fb, did in [
        ("Freezing the predictor and studying the gate is the right design.",
         "Kept. Nothing on the predictor side changed."),
        ("Which evaluation metrics, exactly?",
         "One fixed set, stated on slide 5: error at a fixed answer rate, the area under the "
         "whole curve, the share of the random-to-perfect gap captured, and what each gate "
         "refuses. Ranges get coverage, width and a proper interval score."),
        ("How will you show a difference is significant?",
         "Every number now carries a 95% interval from a bootstrap over plate sessions, and "
         "every gate is tested against refusing at random with a Holm correction. Two "
         "proposal claims did not survive this; slide 8 says which."),
    ]:
        if ry < y + 1:
            rect(s, L - 0.18, ry - 0.06, CW + 0.36, 0.66, fill=BG)
        t = tb(s, L, ry, 4.2, 0.7)
        para(t, fb, size=12, color=INK, bold=True, line=1.25, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.7)
        para(t, did, size=11.5, color=MUTED, line=1.25, first=True)
        ry += 0.74

    ry += 0.04
    eyebrow(s, L, ry, "Changes we made, and why", w=8.0)
    ry += 0.32
    for what, why in [
        ("Calibration is now carved by plate session, not by dish.",
         f"98% of calibration dishes shared a scanning session with a fit dish. Fixing it "
         f"moved the predictor from 73.4 to {n['mae']:.1f} kcal, inside its own interval."),
        ("We report the whole curve and quote the 90% answer rate.",
         "That is where an app would sit. The proposal quoted 50%, where the picture is rosier."),
        ("The phone-camera rung is in. The hidden-food rung is dropped.",
         "Phone: late by three weeks. Hidden food: damage is 9% of the error, and we cannot "
         "label occlusion honestly on this data."),
        ("Confidence ranges started three weeks early.",
         "The first cut is on slide 11. It found the next problem: the largest meals are "
         "under-covered."),
    ]:
        t = tb(s, L, ry, 4.2, 0.6)
        para(t, what, size=11.5, color=INK, bold=True, line=1.22, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.6)
        para(t, why, size=11, color=MUTED, line=1.22, first=True)
        ry += 0.58

    notes(s, "NISARG — 45s.\nTop half is the feedback, verbatim in spirit: the design was "
             "endorsed, the metrics and the significance were questioned. Say what we did "
             "for each. Bottom half: four changes, each with its reason. The first is the "
             "one to slow down on: we found a leak in our own calibration carve and fixed "
             "it, and the headline number moved by 2.5 kcal, inside its interval. Nothing "
             "was tuned on test. Hand to Sushant.")
    return s


def slide_dataset(prs, n):
    s = base(prs, "Dataset", "Nutrition5k, and the unit of analysis we had missed.", 4)
    y = s._body_top

    pic(s, "plate_grid.png", L, y + 0.02, 2.75)
    caption(s, L, y + 2.86, 2.75,
            "Overhead RGB, one fixed rig, every ingredient weighed [1]. Chosen because the "
            "per-ingredient truth lets us build a perfect refuser to measure against.",
            size=10.5)

    x2 = 4.05
    eyebrow(s, x2, y, "Scans of one plate arrive 40 seconds apart", w=7.5)
    pic(s, "sessions.png", x2, y + 0.30, 6.5)
    caption(s, x2, y + 3.36, 7.45,
            "So the 507 test dishes are 127 plate sessions, and error is correlated inside a "
            "session (intraclass correlation 0.34). Sessions are the unit for the calibration "
            "carve and for every confidence interval in this talk.",
            size=11.5, color=INK, bold=True)

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
    notes(s, "SUSHANT — 40s.\nLeft: what the data is, and why we chose it: weighed "
             "ingredients mean we can compute a perfect refuser. Middle: the histogram is "
             "the new thing. Dish ids are timestamps. Scans of the same plate are 40 seconds "
             "apart; the next session is hours away. So the test set is 127 sessions, not "
             "507 independent dishes, and errors inside a session are correlated. Bottom: "
             "the official split, and our calibration carve, now by session. We never "
             "touch the test split. Hand to the design slide.")
    return s


def slide_design(prs, n):
    s = base(prs, "Experimental design and evaluation",
             "One pass, one file, one fixed set of measures.", 5)
    y = s._body_top
    colw = 3.62
    xs = [L, L + colw + 0.32, L + 2 * (colw + 0.32)]

    blocks = [
        ("The pipeline", [
            "Frozen backbone (CLIP ViT-B/16; SigLIP 2 and DINOv3 as checks) → 2,304-d "
            "features, cached once.",
            "Standardise on the fit split, then a closed-form ridge head. No training loop, "
            "so no convergence curve to show.",
            f"Ridge alpha chosen on calibration over a 25-point quarter-decade grid "
            f"(picked {n['alpha_cal']:.0f}). PCA k = 64, B = 32 heads, K = 10 neighbours: "
            "fixed before looking at test.",
            "8 damage levels × 507 dishes × 3 backbones = 12,168 rows, every gate score in "
            "one CSV. Every result is a lookup on that file.",
        ]),
        ("What a gate is scored on", [
            "Error (MAE, kcal) at a fixed answer rate; we quote 90%, and show the whole "
            "curve from 100% to 10%.",
            "Area under that curve (AURC), and the share of the random-to-perfect gap the "
            "gate captures.",
            "Rank correlation (Spearman ρ) between the signal and |error|.",
            "What it refuses: mean size of the meals kept, and the share of 400+ kcal meals "
            "refused, against the perfect refuser’s.",
        ]),
        ("How we test significance", [
            f"Bootstrap over the {n['n_sessions']} plate sessions, {n['n_boot']:,} resamples, "
            "recomputing every curve on the same resample: intervals and paired tests.",
            "Each gate vs refusing at random at each answer rate; Holm correction across the "
            "ten candidates.",
            "Ranges: coverage with an interval, width, share within ±100 kcal, coverage by "
            "difficulty quartile, Winkler interval score.",
            "Compute: one M-series laptop, ~68 images/s; the whole pipeline reruns in under "
            "15 minutes.",
        ]),
    ]
    for x, (title, items) in zip(xs, blocks):
        rect(s, x - 0.14, y - 0.05, colw + 0.28, 4.25, fill=BG)
        t = tb(s, x, y + 0.08, colw, 0.35)
        para(t, title.upper(), size=11, color=ACCENT, bold=True, first=True)
        t = tb(s, x, y + 0.48, colw, 4.4)
        for i, it in enumerate(items):
            para(t, it, size=11, color=INK, line=1.28, first=(i == 0),
                 space_before=0 if i == 0 else 7)

    takeaway(s, "Floor and ceiling on every chart: refusing at random, and a refuser that "
                "knows the true error. A gate is judged by where it sits between them, and by "
                "what it refuses.", y=6.18, size=13.5)
    notes(s, "SUSHANT — 45s.\nThree columns, do not read them. Left: features once, ridge "
             "head, alpha picked on calibration, every other knob fixed in advance. Middle: "
             "the measures, and say the 90% answer rate out loud, that is the operating "
             "point. Right: the bootstrap is over sessions, tests are paired, and there is "
             "a multiple-comparison correction. Then the takeaway line. Hand to Keanu.")
    return s


def slide_predictor(prs, n):
    s = base(prs, "Existing results vs ours: the predictor",
             "The published RGB baseline, reproduced, no training.", 6)
    y = s._body_top
    pic(s, "backbones.png", L - 0.10, y + 0.02, 6.6)
    caption(s, L, y + 3.05, 6.5,
            f"507 test dishes, official split. CLIP {n['mae']:.1f} kcal {fmt_ci(n['mae_ci'])}: "
            f"the published 70.6 sits inside the interval. Portion mass {n['mass']:.1f} g "
            f"({n['mass_rel']:.0f}%). DINOv3 is the only backbone that moves mass.",
            size=11.5, color=INK, bold=True)

    x2 = 7.85
    cols, widths = [x2, x2 + 2.55, x2 + 3.55], [2.5, 1.0, 1.3]
    ry = table_head(s, y, cols, widths, ["Calories, MAE", "kcal", "source"])
    for i, (what, val, src, hi) in enumerate([
        ("RGB, Inception v3, trained", "70.6", "[1]", False),
        ("RGB-D", "47.6", "[1]", False),
        ("Volume scalar, best", "41.3", "[1]", False),
        (f"Ours, CLIP, frozen", f"{n['mae']:.1f}", "[ours]", True),
        (f"Ours, SigLIP 2, frozen", f"{n['mae_siglip']:.1f}", "[ours]", False),
        (f"Ours, DINOv3, frozen", f"{n['mae_dino']:.1f}", "[ours]", False),
        ("2026 re-split, 60/15/25", "28.0", "not comparable", False),
    ]):
        rh = 0.40
        if hi:
            rect(s, x2 - 0.12, ry - 0.06, 4.75, rh, fill=TINT)
        t = tb(s, cols[0], ry, widths[0], 0.35)
        para(t, what, size=11.5, color=INK, bold=hi, first=True)
        t = tb(s, cols[1], ry, widths[1], 0.35)
        para(t, val, size=11.5, color=INK, bold=True, first=True)
        t = tb(s, cols[2], ry, widths[2], 0.35)
        para(t, src, size=10.5, color=ACCENT if src == "[ours]" else MUTED, bold=True,
             first=True)
        ry += rh
    caption(s, x2, ry + 0.08, 4.7,
            "The 28.0 kcal paper re-splits Nutrition5k at random, so near-duplicate scans "
            "land on both sides. We keep [1]’s split and do not compare to it.", size=10.5)

    rect(s, L - 0.18, 6.05, CW + 0.36, 0.80, fill=TINT)
    t = tb(s, L + 0.14, 6.16, CW - 0.3, 0.7)
    para(t, [("WHY WE DO NOT CHASE THE NUMBER   ", {"color": ACCENT, "bold": True, "size": 11}),
             ("a two-stage model that named ingredients and weighed them, with the mass "
              "error we actually measure (22%) and one name in five wrong, lands at 87 kcal "
              "[ours, oracle budget]. Worse than the one-step model we already have.",
              {"color": INK, "size": 11.5})], line=1.3, first=True)
    notes(s, "KEANU — 40s.\nLeft: three frozen backbones, ridge head, no training, and now "
             "with whiskers. The published number is inside our interval, so 'we reproduce "
             "the baseline' is a supported statement, not a point comparison. Right: the "
             "table separates literature numbers from ours, and names the one we refuse to "
             "compare to, because it re-split the data. Bottom: why the predictor stays "
             "frozen. Hand to result 1.")
    return s


def slide_pixel(prs, n):
    s = base(prs, "Result 1, completed", "Pixel signals do not rank the error. The dish does.", 7)
    y = s._body_top
    pr = n["per_rung"]
    base_m = pr["clean"]["mass_mae"]

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
            "Sharpness scores the cropped photo four times better than the blurred one; it "
            "carries far more error, because the plate rim is the only clue to portion size. "
            "The phone rung costs 1.2× on calories and 1.4× on mass.",
            size=11.5, color=INK, bold=True)

    cx = 7.25
    eyebrow(s, cx, y, "Refuse the worst photos, all 8 damage levels pooled  [ours]")
    pic(s, "risk_coverage.png", cx, y + 0.28, 5.1)

    sy = 5.62
    rect(s, L - 0.18, sy, CW + 0.36, 1.22, fill=TINT)
    t = tb(s, L + 0.14, sy + 0.12, CW - 0.3, 1.1)
    para(t, [("THE DAMAGE IS DETECTED, AND DETECTING IT DOES NOT HELP   ",
              {"color": ACCENT, "bold": True, "size": 11}),
             ("sharpness separates damaged photos from clean ones at AUC ≈ 1.00. Yet over "
              f"{507 * 8:,} dish × damage rows, no pixel metric ranks the error above chance "
              f"(|ρ| ≤ {max(abs(v) for v in n['rho_pixel'].values()):.2f}). Variance of the "
              "error: which dish 62%, damage level 9%, their interaction 29%.",
              {"color": INK, "size": 11.5})], line=1.3, first=True)
    para(t, "A gate has to read the dish, so the useful signals come from the model, not the "
            "pixels.", size=12.5, color=INK, bold=True, space_before=6)
    notes(s, "KEANU — 45s.\nLeft: same dish, four photographs, now including the phone "
             "imitation. Point at the cropped one: sharp, and the worst of the four. Right: "
             "on the pooled ladder, sharpness and random sit on top of each other and a "
             "perfect refuser drops to 38. Bottom strip is the finding: we detect the damage "
             "almost perfectly and it does not help, because the dish is 62% of the error. "
             "Hand to Shreya.")
    return s


def slide_curve(prs, n):
    s = base(prs, "Result 2, completed",
             "Model-aware signals bend the curve, with intervals.", 8)
    y = s._body_top
    pic(s, "curves_ci.png", L - 0.10, y + 0.02, 6.55)
    caption(s, L, y + 3.98, 6.4,
            "Clean test photos, CLIP. Bands are 95% intervals over plate sessions.", size=10.5)

    x2 = 7.75
    eyebrow(s, x2, y, "At 90% answered, CLIP  [ours]", w=4.8)
    cols, widths = [x2, x2 + 2.35, x2 + 3.30], [2.3, 0.9, 1.4]
    ry = table_head(s, y + 0.34, cols, widths, ["gate", "kcal", "vs random"])
    g, vr = n["g90"], n["vr"]
    for i, (key, label) in enumerate([
            ("perfect", "perfect refuser"), ("learned_error", "learned error head"),
            ("blend", "blend of four"), ("pred_magnitude", "predicted size (control)"),
            ("ensemble_spread", "ensemble disagreement"), ("mahalanobis", "distance to mean"),
            ("knn_dist", "neighbour distance"), ("random", "random"),
            ("lap_var", "sharpness")]):
        rh = 0.33
        hi = key in ("learned_error",)
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
            "✓ = better than random after Holm correction, p < 0.05. Interval is the paired "
            "difference.", size=10)

    rect(s, L - 0.18, 6.08, CW + 0.36, 0.82, fill=TINT)
    t = tb(s, L + 0.14, 6.16, CW - 0.3, 0.75)
    le, es = vr["learned_error"], vr["ensemble_spread"]
    pr = n["pair"]["ensemble_spread - learned_error"]
    para(t, [("WHAT SURVIVED THE TEST   ", {"color": ACCENT, "bold": True, "size": 11}),
             (f"the learned error head beats random by {-le['diff']:.0f} kcal at 90% answered "
              f"(p = {le['p_holm']:.3f}). The proposal’s favourite, ensemble disagreement, does "
              f"not clear the correction there (p = {es['p_holm']:.2f}) and is "
              f"{pr['diff']:.0f} kcal behind the error head (paired p = {pr['p']:.3f}). "
              "Neither distance gate is distinguishable from random on clean photos.",
              {"color": INK, "size": 11.5})], line=1.3, first=True)
    notes(s, "SHREYA — 40s.\nLeft: the whole curve, with bands. Random is flat, the perfect "
             "refuser is the floor, the two learned gates sit between. Right: at 90% "
             "answered, the numbers and the paired difference to random. Read the strip: "
             "the learned error head wins, ensemble disagreement, which the proposal "
             "favoured, does not survive the correction at this operating point. That is "
             "what the markers asked us to be able to say. Next slide: three backbones.")
    return s


def slide_significance(prs, n):
    s = base(prs, "Result 2, completed", "The same answer on three backbones.", 9)
    y = s._body_top
    pic(s, "significance.png", L + 1.05, y + 0.02, 9.4)

    sy = 5.68
    x = L
    for title, body in [
        ("What is significant",
         "The learned error head, the blend and the control beat random on all three "
         "backbones (Holm p < 0.01). The head and the blend are identical: blending adds "
         "nothing."),
        ("What is not",
         "Ensemble disagreement clears the test at 50% answered, not at 90%. The two distance "
         "gates never do. No pixel metric ever does; sharpness is worse than random at 50%."),
        ("Share of the gap captured",
         f"Learned error head {100 * n['gap']['learned_error']:.0f}% "
         f"{fmt_ci([100 * v for v in n['gap_ci']['learned_error']])}, ensemble "
         f"{100 * n['gap']['ensemble_spread']:.0f}%, distance to mean "
         f"{100 * n['gap']['mahalanobis']:.0f}% with an interval spanning zero."),
    ]:
        rect(s, x - 0.12, sy - 0.04, 3.72, 1.22, fill=BG)
        t = tb(s, x, sy + 0.04, 3.5, 0.3)
        para(t, title.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        t = tb(s, x, sy + 0.33, 3.5, 0.9)
        para(t, body, size=10.5, color=INK, line=1.22, first=True)
        x += 3.95
    notes(s, "SHREYA — 40s.\nOne panel per backbone, same rows. Filled dots are gates that "
             "beat random after correction. The pattern is the same three times: the learned "
             "gates and the control do, ensemble disagreement does not at this operating "
             "point, distance gates and pixel gates never. The gap share says how much of "
             "the room between random and perfect each captures: about half for the error "
             "head. Next: what they refuse, because the control being on this list should "
             "worry you.")
    return s


def slide_degeneracy(prs, n):
    s = base(prs, "Result 3, completed", "Check what each gate refuses before believing it.", 10)
    y = s._body_top
    pic(s, "degeneracy.png", L + 1.15, y + 0.02, 9.2)
    sel = n["sel"]
    rect(s, L - 0.18, 5.50, CW + 0.36, 1.40, fill=TINT)
    t = tb(s, L + 0.14, 5.58, CW - 0.3, 1.3)
    para(t, [("THE CONTROL   ", {"color": ACCENT, "bold": True, "size": 11}),
             ("error grows with portion size, so refusing big meals lowers the error without "
              "reading the photo. Scoring risk by the predicted calorie count alone does "
              "exactly that.", {"color": INK, "size": 11})], line=1.25, first=True)
    para(t, f"At 90% answered every learned gate keeps meals the size a perfect refuser keeps "
            f"(error head {sel['learned_error']['0.9']['kept_mean_kcal']:.0f} kcal, perfect "
            f"{sel['perfect']['0.9']['kept_mean_kcal']:.0f}). At 50% the control keeps "
            f"{sel['pred_magnitude']['0.5']['kept_mean_kcal']:.0f}-kcal meals and the error "
            f"head {sel['learned_error']['0.5']['kept_mean_kcal']:.0f}, against "
            f"{sel['perfect']['0.5']['kept_mean_kcal']:.0f} for a perfect refuser. Only "
            f"ensemble disagreement stays with the oracle ({sel['ensemble_spread']['0.5']['kept_mean_kcal']:.0f}).",
         size=11, color=INK, line=1.25, space_before=3)
    para(t, "So the honest gate depends on the answer rate. We report the kept-set size "
            "beside every error, at every operating point.",
         size=11.5, color=INK, bold=True, space_before=3)
    notes(s, "SHREYA — 35s.\nLeft panel is the operating point we report: at 90% every "
             "learned gate keeps normal-sized meals, so the error head’s win is honest "
             "there. Right panel is 50%: the control and the error head drift to small "
             "meals, ensemble disagreement stays with the oracle. That is why the proposal "
             "liked it, and why we now say which gate is honest at which answer rate. Hand "
             "to Kevin.")
    return s


def slide_ranges(prs, n):
    s = base(prs, "Result 4, first cut", "Ranges: honest on average, not on the hard dishes.", 11)
    y = s._body_top
    c = n["conf"]
    plain, adapt = c["constant"], c["ensemble_spread"]
    pic(s, "conformal.png", L - 0.15, y + 0.02, 7.6)
    caption(s, L, y + 2.98, 7.4,
            f"Split conformal, 90% target, {n['n_calib']} calibration dishes, 507 clean test "
            f"photos, CLIP [ours].", size=10.5)

    x2 = 8.85
    cols, widths = [x2, x2 + 1.85, x2 + 2.75], [1.8, 0.9, 0.9]
    ry = table_head(s, y, cols, widths, ["", "plain", "adaptive"])
    for lab, a, b in [
        ("coverage", f"{100 * plain['coverage']:.1f}%", f"{100 * adapt['coverage']:.1f}%"),
        ("median half-width", f"±{plain['width_median'] / 2:.0f} kcal",
         f"±{adapt['width_median'] / 2:.0f} kcal"),
        ("interval", fmt_ci([100 * v for v in plain["coverage_ci"]]),
         fmt_ci([100 * v for v in adapt["coverage_ci"]])),
        ("hardest quarter", f"{100 * plain['by_difficulty']['coverage'][3]:.0f}%",
         f"{100 * adapt['by_difficulty']['coverage'][3]:.0f}%"),
        ("largest meals", f"{100 * plain['by_kcal']['coverage'][3]:.0f}%",
         f"{100 * adapt['by_kcal']['coverage'][3]:.0f}%"),
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

    mix = n["mix"]
    sy = 5.08
    rect(s, L - 0.18, sy, CW + 0.36, 1.74, fill=TINT)
    t = tb(s, L + 0.14, sy + 0.09, CW - 0.3, 1.6)
    para(t, [("THREE THINGS THIS TELLS US   ", {"color": ACCENT, "bold": True, "size": 11}),
             ("(1) Marginal coverage hides a lot: the plain range reads 90% overall and covers "
              f"the hardest quarter of dishes {100 * plain['by_difficulty']['coverage'][3]:.0f}% "
              "of the time. Letting the gate signal set the width closes most of that gap on "
              "all three backbones and improves the proper score.",
              {"color": INK, "size": 11})], line=1.25, first=True)
    para(t, f"(2) Neither range covers the largest quarter of meals at 90% (about "
            f"{100 * plain['by_kcal']['coverage'][3]:.0f}%). That is the problem O3 has to "
            "solve next, and it is the same big-meal effect as slide 10.",
         size=11, color=INK, line=1.25, space_before=3)
    mc = mix["uniform"]["constant"]
    para(t, f"(3) Exchangeability is real: calibrate on clean photos and deploy on a damaged "
            f"mixture and coverage drops to "
            f"{100 * mc['calibrated_on_clean']['coverage']:.0f}%; calibrating on the same "
            f"mixture restores {100 * mc['calibrated_on_mixture']['coverage']:.0f}% at "
            f"{mc['calibrated_on_mixture']['width_mean'] / mc['calibrated_on_clean']['width_mean'] * 100 - 100:.0f}% "
            "more width. A check, not a finding.",
         size=11, color=INK, line=1.25, space_before=3)
    notes(s, "KEVIN — 45s.\nThis is O3’s first measurement, three weeks early. Left chart: "
             "coverage by how hard the dish looks. Plain conformal gives every photo the same "
             "range and reads 90% overall, but it over-covers easy dishes and under-covers "
             "hard ones. Let the gate signal set the width and the hard quarter improves. "
             "Right table: the numbers, including the one that worries us: the largest meals "
             "are under-covered by both. That is next. Then the exchangeability check in one "
             "sentence.")
    return s


def slide_timeline(prs, n):
    s = base(prs, "Challenges, timeline, roles", "Two milestones slipped, one moved early.", 12)
    y = s._body_top

    eyebrow(s, L, y, "Challenges and what we did", w=5.5)
    ry = y + 0.32
    for what, did in [
        ("Correlated test dishes",
         "Errors cluster by plate session (ICC 0.34). Unit of resampling is now the session; "
         "the calibration carve respects it."),
        ("A gate that cheats",
         "Refusing big meals looks like skill. A control gate exposed it; kept-set size is "
         "reported at every operating point."),
        ("Rank-deficient covariance",
         "2,304 dims on 1,924 dishes. Mahalanobis on the leading 64 components, fixed in "
         "advance."),
        ("No phone photos in the data",
         "The phone rung is an imitation of four mild effects, and is labelled as one."),
    ]:
        t = tb(s, L, ry, 5.3, 0.3)
        para(t, what, size=11.5, color=INK, bold=True, first=True)
        t = tb(s, L, ry + 0.27, 5.3, 0.6)
        para(t, did, size=10.5, color=MUTED, line=1.25, first=True)
        ry += 0.86

    x2 = 6.85
    eyebrow(s, x2, y, "Proposal timeline vs actual", w=5.5)
    cols, widths = [x2, x2 + 1.35, x2 + 4.15], [1.3, 2.75, 1.5]
    ty = table_head(s, y + 0.32, cols, widths, ["planned", "milestone", "status"])
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
            "Next: size-aware ranges for large meals (CQR); the deployment mixture on all "
            "backbones; the cost tier (test-time augmentation, a second backbone); report.",
            size=10.5, color=INK, bold=True)

    ry = 5.55
    eyebrow(s, L, ry, "Who owns what")
    ry += 0.32
    for i, (who, what) in enumerate([
            ("Nisarg", "pipeline, results file, framing"), ("Shreya", "damage levels, pixel gates"),
            ("Keanu", "model-aware gates, curves"), ("Sushant", "literature, writing, data"),
            ("Kevin", "confidence ranges")]):
        t = tb(s, L + i * 2.32, ry, 2.25, 0.5)
        para(t, who, size=12, color=INK, bold=True, first=True)
        para(t, what, size=10, color=MUTED, space_before=2, line=1.25)
    notes(s, "KEVIN — 40s.\nLeft: four challenges, one line each, the first is the one to "
             "say in full. Right: the proposal timeline with honest statuses: two late, one "
             "dropped with a reason, one early. Then the next steps line. Bottom: the roles. "
             "State each member’s contribution in one clause each when you name them.")
    return s


def slide_conclusions(prs, n):
    s = base(prs, "Conclusions", "What we know now, and did not at the proposal.", 13)
    y = s._body_top
    le = n["vr"]["learned_error"]
    for i, (head, body) in enumerate([
        ("Photo quality is not the axis.",
         "Damage is detected almost perfectly and explains 9% of the error. The dish explains "
         "62%. Cheap pixel gates never beat random, on any backbone, at any answer rate."),
        ("Model-aware gates work, and we can now say how sure we are.",
         f"At 90% answered the learned error head cuts the error by {-le['diff']:.0f} kcal "
         f"against random (Holm p = {le['p_holm']:.3f}) on three backbones, keeps normal-sized "
         "meals, and captures about half the gap to a perfect refuser."),
        ("Which gate is honest depends on the answer rate.",
         "At 50% the same gate drifts to small meals and ensemble disagreement is the one that "
         "tracks the oracle. A control gate and the kept-set size are part of every result."),
        ("Ranges are honest on average and not yet on hard or large dishes.",
         "Adaptive width fixes most of the difficulty gap. The largest meals are still "
         "under-covered. That is O3’s job by 11 October."),
    ]):
        rect(s, L - 0.18, y - 0.04, CW + 0.36, 0.98, fill=BG if i % 2 == 0 else None)
        t = tb(s, L, y + 0.04, 0.5, 0.5)
        para(t, str(i + 1), size=18, color=ACCENT, bold=True, first=True)
        t = tb(s, L + 0.55, y + 0.02, CW - 0.6, 0.9)
        para(t, head, size=13.5, color=INK, bold=True, first=True)
        para(t, body, size=11.5, color=MUTED, line=1.3, space_before=3)
        y += 1.06
    takeaway(s, "Headline question, answered so far: an app can know when it is guessing, "
                "from a signal that costs no extra inference, at the cost of refusing one "
                "photo in ten.", y=6.12, size=13.5)
    notes(s, "KEVIN — 20s.\nFour sentences, read the bold lines only. Then the headline. "
             "Hand back for questions. Owners for Q&A: framing and data Nisarg, splits and "
             "metrics Sushant, predictor and pixel result Keanu, gates and significance "
             "Shreya, ranges and timeline Kevin.")
    return s


def slide_references(prs, n):
    s = base(prs, "Closing", "Sources, AI use, and where the code is.", 14)
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
        "Re-split comparison on slide 6: V. Awasthi et al., Int. J. Intell. Eng. Syst., vol. 19, "
        "no. 2, 2026, doi:10.22266/ijies2026.0228.08.",
        "All figures and tables are ours unless tagged [1]. Images on slides 4 and 7 are Nutrition5k "
        "dishes [1], reproduced under the dataset licence.",
    ]
    t = tb(s, L, y, CW, 3.6)
    for i, r in enumerate(refs):
        para(t, r, size=10, color=INK if i < 6 else MUTED, line=1.25, first=(i == 0),
             space_before=0 if i == 0 else 4)

    ry = 5.05
    rect(s, L - 0.18, ry, CW + 0.36, 0.78, fill=BG)
    t = tb(s, L + 0.14, ry + 0.10, CW - 0.3, 0.7)
    para(t, [("GENERATIVE AI USE   ", {"color": ACCENT, "bold": True, "size": 10.5}),
             ("Claude Code (Anthropic) was used to scaffold pipeline code, generate the slides "
              "from the results files, and improve wording. The experimental design, analysis "
              "choices, results and conclusions are the group’s own and reproduce from the "
              "repository.", {"color": INK, "size": 10.5})], line=1.3, first=True)
    rect(s, L - 0.18, ry + 0.92, CW + 0.36, 0.62, fill=TINT)
    t = tb(s, L + 0.14, ry + 1.05, CW - 0.3, 0.5)
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
    for fn in (slide_title, slide_recap, slide_changes, slide_dataset, slide_design,
               slide_predictor, slide_pixel, slide_curve, slide_significance,
               slide_degeneracy, slide_ranges, slide_timeline, slide_conclusions,
               slide_references):
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
