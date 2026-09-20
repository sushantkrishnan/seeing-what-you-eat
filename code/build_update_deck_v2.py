"""Build v2 of the COMPSCI 760 Group 1 update deck: the size-controlled narrative.

  ../.venv/bin/python build_update_deck_v2.py     -> 01_MethodResults_v2.pptx

Run gate_bench.py, gate_probe.py --all --json, conformal.py --all --json and
figures.py first. v1 (build_update_deck.py) is untouched; v2 imports its primitives and
its unchanged slides and redefines the rest.

WHAT CHANGED FROM v1 (20 Sep review):
  * The random reference is analytic, not one fixed draw: every "vs random" moved by
    up to a kcal, and ensemble disagreement now clears the Holm correction at 90%.
  * Every gate is scored again INSIDE predicted-size bands (slide 11, new). The control
    collapses to zero, the learned head keeps about half its win, the perfect refuser
    keeps two thirds. That is the deck's answer to "is it size or is it the dish".
  * Relative error is reported as the secondary loss (slide 12): it has the opposite
    bias, and there no gate beats random.
  * Slide 9 no longer claims "no pixel gate at any answer rate": on the pooled ladder
    two of them do, at ~45% answered, by dropping whole damage levels. The one-dish
    panel now carries that dish's own numbers, read from gate_scores.csv.
  * Slide 3 says which stage error is measured and which is assumed, and puts the real
    gate's number beside the perfect refuser's.
  * Citations: [1] pagination, Inception V2, [2] cited by its own title, [2] and [4]
    now cited on a slide. "Floor" is gone; random is a reference.

House style, unchanged: short declarative sentences, no aphorisms, no "not X but Y"
inversions, few dashes, no novelty-by-absence claims. Every number is tagged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_deck import (ACCENT, BG, BLUE, CW, FAINT, GREEN, H, INK, L, MUTED, R, TINT, W,
                        arrow, base, caption, eyebrow, notes, para, pic, rect, table_head,
                        takeaway, tb)
from build_update_deck import (CLIP, DINO, FIGS, PIXEL, PUBLISHED_BEST, PUBLISHED_RGB,
                               REPO_URL, RESULTS, ROOT, SIGLIP, TWO_STAGE_KCAL, fmt_ci,
                               fmt_p, load_numbers, slide_dataset, slide_design,
                               slide_recap, slide_title, strip)

OUT = ROOT / "01_MethodResults_v2.pptx"
DEMO_DISH = "dish_1565811091"                       # the dish on slide 9 (figures.py)
BACKBONES = [(CLIP, "CLIP"), (SIGLIP, "SigLIP 2"), (DINO, "DINOv3")]

SLIDE_PLAN = [
    ("Nisarg",  35, "Recap: problem, proposal, objectives"),
    ("Nisarg",  30, "Why the gate: 71, 50, 60, against 87"),
    ("Nisarg",  30, "Feedback and changes"),
    ("Sushant", 30, "Dataset, splits, sessions"),
    ("Sushant", 30, "Pipeline, the one knob, compute"),
    ("Sushant", 35, "How to read every chart"),
    ("Keanu",   30, "Existing vs ours: the predictor"),
    ("Keanu",   40, "Result 1: pixel signals"),
    ("Shreya",  35, "Result 2: the curve, and the control"),
    ("Shreya",  40, "Result 3: size held fixed"),
    ("Shreya",  30, "Result 3: where the other half went"),
    ("Kevin",   40, "Result 4: ranges"),
    ("Kevin",   25, "Challenges, timeline, roles"),
    ("Kevin",   30, "Conclusions"),
]

REQUIRED_FIGS = ["plate_grid.png", "sessions.png", "backbones.png", "panel_clean.png",
                 "panel_blur6.png", "panel_crop04.png", "panel_phone.png",
                 "risk_coverage.png", "curves_ci.png", "significance_sized.png",
                 "degeneracy.png", "conformal_kcal.png", "reader.png"]


# ---------------------------------------------------------------- numbers
def load_numbers_v2() -> dict:
    n = load_numbers()
    gp = json.loads((RESULTS / "gate_probe.json").read_text())
    c = gp[CLIP]
    n["sized"], n["rel"] = c["clean_sized"], c["clean_rel"]
    n["sel_sized"] = {k: v["selectivity"] for k, v in c["clean_sized"]["gates"].items()}
    n["rho_size"] = {k: v.get("rho_size") for k, v in c["clean"]["gates"].items()}
    n["rho_within"] = {k: v.get("rho_within_size") for k, v in c["clean"]["gates"].items()}

    # the three-backbone strip on slide 11: plain / sized / relative differences at 90%
    per = {}
    for key, short in BACKBONES:
        b = gp[key]
        d = lambda blk, g: b[blk]["gates"][g]["vs_random"]["0.9"]
        perf = lambda blk: (b[blk]["gates"]["perfect"]["mae"]["0.9"]
                            - b[blk]["gates"]["random"]["mae"]["0.9"])
        per[short] = {
            "plain": {g: d("clean", g) for g in ("learned_error", "pred_magnitude",
                                                 "ensemble_spread")},
            "sized": {g: d("clean_sized", g) for g in ("learned_error", "pred_magnitude",
                                                       "ensemble_spread")},
            "rel": {g: d("clean_rel", g) for g in ("learned_error", "pred_magnitude",
                                                   "ensemble_spread")},
            "perfect_plain": perf("clean"), "perfect_sized": perf("clean_sized"),
            "head_vs_control_sized": b["clean_sized"]["pairwise"]["0.9"]
                                      ["learned_error - pred_magnitude"],
            "head_vs_control_plain": b["clean"]["pairwise"]["0.9"]
                                      ["learned_error - pred_magnitude"],
            "kept_sized": {g: b["clean_sized"]["gates"][g]["selectivity"]["0.9"]["kept_mean_kcal"]
                           for g in ("learned_error", "pred_magnitude", "ensemble_spread",
                                     "random", "perfect")},
        }
    n["per"] = per

    # the pixel gates on the pooled ladder, every backbone: how far from random at 90%,
    # and the most any of them captures over all answer rates
    n["pixel_90_pooled_max"] = max(abs(c2["pooled"]["gates"][g]["vs_random"]["0.9"]["diff"])
                                   for c2 in gp.values() for g in PIXEL)
    n["pixel_gap_pooled_max"] = max(c2["pooled"]["gates"][g]["gap_share"]
                                    for c2 in gp.values() for g in PIXEL)
    n["sharpness_worse_90_all"] = all(
        c2["pooled"]["gates"]["lap_var"]["vs_random"]["0.9"]["diff"] > 0
        and c2["pooled"]["gates"]["lap_var"]["vs_random"]["0.9"]["p_holm"] < 0.05
        for c2 in gp.values())
    n["pixel_clean_any"] = any(
        c2["clean"]["gates"][g]["vs_random"][cov].get("p_holm", 1) < 0.05
        and c2["clean"]["gates"][g]["vs_random"][cov]["diff"] < 0
        for c2 in gp.values() for g in PIXEL for cov in [str(x) for x in c2["coverages"]])

    # the demo dish's own numbers for the slide 9 panel
    panel = {}
    with open(RESULTS / "gate_scores.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["backbone"] == CLIP and r["dish_id"] == DEMO_DISH:
                panel[r["rung"]] = {"lap_var": float(r["lap_var"]),
                                    "mass_abs_err": float(r["mass_abs_err"]),
                                    "cal_abs_err": float(r["cal_abs_err"])}
    n["panel"] = panel
    return n


def fmt_sharp(v: float) -> str:
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


# ---------------------------------------------------------------- slides
def slide_why(prs, n):
    s = base(prs, "Why the gate, not the predictor",
             "Refusing one photo in ten beats a better predictor.", 3)
    y = s._body_top
    mae, perf, real = n["mae"], n["g90"]["perfect"], n["g90"]["learned_error"]

    eyebrow(s, L, y, "Our own numbers, 507 test dishes, calorie error in kcal  [ours]", w=9)
    ty = y + 0.42
    th = 2.55
    tiles = [
        (L, 3.15, BG, INK, f"{mae:.0f}", "answering every photo", "our frozen predictor"),
        (L + 3.75, 3.15, TINT, GREEN, f"{perf:.0f}",
         "with a perfect refuser declining one photo in ten",
         f"{round(mae) - round(perf):.0f} kcal of headroom. Our best real gate lands at "
         f"{real:.0f} (slide 10)."),
        (L + 7.90, 3.60, BG, ACCENT, str(TWO_STAGE_KCAL),
         "a two-stage model that names and weighs the ingredients",
         "with 22% mass error, which we measure, and one name in five wrong, which we "
         "assume  [ours, oracle budget]"),
    ]
    for x, w, fill, col, big, lab, sub in tiles:
        rect(s, x, ty, w, th, fill=fill)
        t = tb(s, x + 0.25, ty + 0.15, w - 0.5, 1.0)
        para(t, big, size=54, color=col, bold=True, first=True)
        t = tb(s, x + 0.25, ty + 1.12, w - 0.5, 1.35)
        para(t, lab, size=12.5, color=INK, bold=True, line=1.28, first=True)
        para(t, sub, size=11, color=MUTED, line=1.28, space_before=4)
    arrow(s, L + 3.22, ty + th / 2 - 0.12, 0.42, 0.24, color=GREEN)

    caption(s, L, ty + th + 0.22, CW,
            f"The best published number, {PUBLISHED_BEST:.0f} kcal, used a depth channel "
            f"[1]; most phone photos carry none. Published RGB: {PUBLISHED_RGB} kcal [1].",
            size=12, color=INK)
    takeaway(s, "The headroom is in refusing, not in predicting from colour alone. So the "
                "predictor is frozen and the gate is the object of study.", y=5.85, size=15)
    notes(s, "NISARG — 30s.\nThree numbers, left to right. 71 is where we are. 50 is what "
             "a perfect refuser would give us by declining one photo in ten, and our best "
             "real gate already gets 60 of those 21 kcal back. 87 is where the ambitious "
             "route lands, naming and weighing the ingredients, with the mass error we "
             "measure and a naming error we assume: worse than what we have. The only "
             "number that beats us used depth. That is why the predictor is frozen. Hand "
             "to feedback.")
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
         "share of the gap captured, and what each gate refuses. Then every gate is scored "
         "again with meal size held fixed (slide 11)."),
        ("How will you show a difference is significant?",
         "Every number carries a 95% interval from a bootstrap over plate sessions; every "
         "gate is tested against refusing at random, paired, with a Holm correction "
         "(slides 10 and 11)."),
    ]:
        if ry < y + 1:
            rect(s, L - 0.18, ry - 0.06, CW + 0.36, 0.58, fill=BG)
        t = tb(s, L, ry, 4.2, 0.65)
        para(t, fb, size=12, color=INK, bold=True, line=1.22, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.65)
        para(t, did, size=11, color=MUTED, line=1.22, first=True)
        ry += 0.64

    ry += 0.06
    eyebrow(s, L, ry, "Changes we made, and why", w=8.0)
    ry += 0.34
    for what, why in [
        ("Calibration carved by plate session, not by dish.",
         f"98% of calibration dishes shared a session with a fit dish. Fixed; the predictor "
         f"moved 73.4 → {n['mae']:.1f} kcal, inside its interval  → slide 5"),
        ("We quote 90% answered and show the whole curve.",
         "Where an app would sit. The proposal quoted 50%, where every gate looks better  "
         "→ slide 7"),
        ("Every gate scored again inside size bands.",
         "A control that reads only the predicted size tied our best gate. Holding size "
         "fixed separates reading the dish from reading its size  → slide 11"),
        ("Phone rung in, late. Hidden-food rung dropped.",
         "Damage is 9% of the error, and occlusion cannot be labelled honestly  → slide 9"),
        ("Confidence ranges started three weeks early.",
         "First cut done; it found the next problem, large meals  → slide 13"),
    ]:
        t = tb(s, L, ry, 4.2, 0.55)
        para(t, what, size=11.5, color=INK, bold=True, line=1.2, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.55)
        para(t, why, size=11, color=MUTED, line=1.2, first=True)
        ry += 0.50

    notes(s, "NISARG — 30s.\nTop: the three things the markers said, and one sentence each "
             "on what we did. Bottom: five changes, one line each, each pointing at the "
             "slide where it bites. Slow down on the first and the third: we found a leak "
             "in our own calibration split and fixed it; and we found that our best gate "
             "was tied by a control that only reads the size of the meal, so every gate is "
             "now scored again with size held fixed. Nothing was tuned on test. Hand to "
             "Sushant.")
    return s


def slide_reader(prs, n):
    s = base(prs, "How we score a gate", "How to read every chart that follows.", 7)
    y = s._body_top
    pic(s, "reader.png", L - 0.10, y + 0.02, 6.2)
    caption(s, L, y + 3.82, 6.2,
            "Clean test photos, CLIP [ours]. Protocol from biometrics [3] and selective "
            "prediction [6]; random is a reference, not a bound.", size=10.5)

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

    strip(s, 6.02, 0.92, "Every number carries an interval",
          f"{n['n_boot']:,} resamples of the {n['n_sessions']} plate sessions, every curve "
          "recomputed on the same resample, so tests are paired. Each gate is tested "
          "against refusing at random and Holm-corrected across the ten candidates. Then "
          "every gate is scored twice more: with meal size held fixed (slide 11) and on "
          "relative error (slide 12).", size=11.5)
    notes(s, "SUSHANT — 35s.\nThis is the one chart to learn. Across: how many photos the "
             "app still answers. Down: the error on those photos. Refusing at random is "
             "flat, and it is a reference, not a floor: a bad gate sits above it. A perfect "
             "refuser, one that knows the true error, falls away. Every real gate lands "
             "between them, and we read it at 90% answered, one refusal in ten. Four "
             "measures on the right; the fourth is ours. Strip: every number gets an "
             "interval from resampling sessions, tests are paired, there is a correction "
             "for testing ten gates, and every gate is scored a second time with meal size "
             "held fixed. Hand to Keanu.")
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
        ("RGB, Inception V2, trained", f"{PUBLISHED_RGB}", "[1]", False),
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
    pr, pan = n["per_rung"], n["panel"]
    base_m, base_p = pr["clean"]["mass_mae"], pan["clean"]["mass_abs_err"]
    rho_px = max(abs(n["rho_pixel"][k]) for k in PIXEL)

    eyebrow(s, L, y, "One dish, four photographs, its own numbers  [ours]", w=6.5)
    py = y + 0.32
    pw, gap = 1.42, 0.12
    for i, (fn, label, tagk, hi) in enumerate([
            ("panel_clean.png", "clean", "clean", False),
            ("panel_blur6.png", "blurred", "blur6", False),
            ("panel_crop04.png", "cropped", "crop0p4", True),
            ("panel_phone.png", "phone", "phone", False)]):
        px = L + i * (pw + gap)
        pic(s, fn, px, py, pw)
        t = tb(s, px, py + pw * 0.75 + 0.08, pw, 0.9)
        para(t, label, size=12, color=ACCENT if hi else INK, bold=True,
             align=PP_ALIGN.CENTER, first=True)
        para(t, [("sharpness ", {"color": MUTED, "size": 10}),
                 (fmt_sharp(pan[tagk]["lap_var"]), {"color": INK, "size": 11, "bold": True})],
             align=PP_ALIGN.CENTER, space_before=3)
        para(t, [("mass error ", {"color": MUTED, "size": 10}),
                 (f"{pan[tagk]['mass_abs_err'] / base_p:.1f}×",
                  {"color": ACCENT if hi else INK, "size": 11, "bold": True})],
             align=PP_ALIGN.CENTER, space_before=1)
    sharp_ratio = pan["crop0p4"]["lap_var"] / pan["blur6"]["lap_var"]
    caption(s, L, py + pw * 0.75 + 0.95, 6.1,
            f"Sharpness scores this cropped photo {sharp_ratio:.0f}× better than the blurred "
            f"one, and it carries {pan['crop0p4']['mass_abs_err'] / base_p:.0f}× the mass "
            f"error: the plate rim is the only clue to portion size. Across all 507 dishes: "
            f"{pr['crop0p4']['lap_var'] / pr['blur6']['lap_var']:.0f}× sharper, "
            f"{pr['crop0p4']['mass_mae'] / base_m:.0f}× the error. The phone rung is an "
            f"imitation, and costs {pr['phone']['cal_mae'] / pr['clean']['cal_mae']:.1f}× on "
            "calories.", size=11, color=INK, bold=True)

    cx = 7.25
    eyebrow(s, cx, y, "Refuse the worst photos, 8 damage levels [4] pooled  [ours]", w=5.6)
    pic(s, "risk_coverage.png", cx, y + 0.28, 5.1)

    assert not n["pixel_clean_any"], "a pixel gate beat random on clean photos; reword"
    worse = ("sharpness is worse than random on all three backbones"
             if n["sharpness_worse_90_all"] else "sharpness is no better than random")
    strip(s, 5.72, 1.20, "Detected, and it barely helps",
          f"pixel metrics separate damaged photos from clean ones at AUC ≈ 1.00, the test "
          f"[2] asks for. Over {507 * 8:,} dish × damage rows every pixel gate sits within "
          f"{n['pixel_90_pooled_max']:.0f} kcal of random at 90% answered ({worse}); the "
          f"most any captures over all answer rates is {100 * n['pixel_gap_pooled_max']:.0f}% "
          f"of the gap, by dropping whole damage levels (|ρ| ≤ {rho_px:.2f}). On clean "
          "photos none beats random at any answer rate. Variance of the error: which dish "
          "62%, damage level 9%, their interaction 29%. A gate has to read the dish.",
          size=11)
    notes(s, "KEANU — 40s.\nLeft: one dish, four photographs, and that dish's own numbers. "
             "The cropped one is sharp and the worst of the four, because the rim is the "
             "only clue to portion size; sharpness ranks it sixteen times better than the "
             "blurred one. Right: refuse the photos each signal likes least; sharpness and "
             "random sit on top of each other, a perfect refuser drops from a hundred to "
             "thirty-eight. Strip: we detect the damage almost perfectly and it barely "
             "helps: nothing at one refusal in ten, and at best five percent of the gap at "
             "half, by throwing away whole damage levels. Damage is nine percent of the "
             "error; the dish is sixty-two. Hand to Shreya.")
    return s


def slide_curve(prs, n):
    s = base(prs, "Result 2, completed",
             "Model-aware signals bend the curve. So does size.", 10)
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
            ("knn_dist", "neighbour distance"), ("random", "random (answer all)"),
            ("lap_var", "sharpness")]:
        rh = 0.33
        hi = key in ("learned_error", "pred_magnitude")
        if hi:
            rect(s, x2 - 0.12, ry - 0.05, 4.75, rh, fill=TINT)
        t = tb(s, cols[0], ry, widths[0], 0.3)
        para(t, label, size=11, color=GREEN if key == "perfect" else INK, bold=hi, first=True)
        t = tb(s, cols[1], ry, widths[1], 0.3)
        para(t, f"{g[key]:.1f}", size=11, color=INK, bold=True, first=True)
        t = tb(s, cols[2], ry, widths[2], 0.3)
        if key in vr:
            d = vr[key]
            sig = d.get("p_holm", 1) < 0.05
            star = ("  ✓" if d["diff"] < 0 else "  ✗") if sig else ""
            txt = f"{d['diff']:+.1f} {fmt_ci(d['ci'])}{star}"
            col = (GREEN if d["diff"] < 0 else ACCENT) if sig else MUTED
        else:
            txt, col = "—", FAINT
        para(t, txt, size=10.5, color=col, bold=bool(key in vr and star), first=True)
        ry += rh
    caption(s, x2, ry + 0.06, 4.75,
            "✓ better, ✗ worse than random after Holm correction, p < 0.05. The control "
            "scores risk by the predicted calorie count alone.", size=10)

    le, pm, es = vr["learned_error"], vr["pred_magnitude"], vr["ensemble_spread"]
    pc = n["pair"]["learned_error - pred_magnitude"]
    strip(s, 6.12, 0.82, "What survived",
          f"the learned error head beats random by {-le['diff']:.0f} kcal at 90% answered "
          f"({fmt_p(le['p_holm'])}). So does the control, by {-pm['diff']:.0f}, and the two "
          f"are tied (paired p = {pc['p']:.2f}). Ensemble disagreement clears the "
          f"correction at {-es['diff']:.0f} kcal. Neither distance gate is distinguishable "
          "from random; sharpness is worse than it.", size=11.5)
    notes(s, "SHREYA — 35s.\nSame axes as slide 7, now with the gates drawn. Random flat, "
             "perfect refuser at the bottom, the learned gates between. Table: at 90% "
             "answered, the error head takes seventy-one to sixty, eleven kcal better than "
             "random, corrected p under one percent. Now the highlighted row under it: a "
             "gate that reads nothing but the predicted calorie count gets ten of those "
             "eleven, and the two are statistically tied. So before we believe the head, "
             "we have to ask how much of its win is just the size of the meal. Next slide.")
    return s


def slide_sized(prs, n):
    s = base(prs, "Result 3, completed", "Hold meal size fixed and half the win survives.", 11)
    y = s._body_top
    pic(s, "significance_sized.png", L - 0.15, y - 0.02, 7.2)
    per = n["per"]
    sep = "\u00a0/\u00a0"                     # non-breaking, so a triplet never wraps mid-way
    fmt3 = lambda blk, g: sep.join(f"{per[b][blk][g]['diff']:+.0f}" for _, b in BACKBONES)
    fmt3s = lambda blk, g: sep.join(f"{per[b][blk][g]['diff']:+.1f}" for _, b in BACKBONES)
    pperf = sep.join(f"{per[b]['perfect_plain']:+.0f}" for _, b in BACKBONES)
    sperf = sep.join(f"{per[b]['perfect_sized']:+.0f}" for _, b in BACKBONES)
    pmax = max(per[b]["head_vs_control_sized"]["p"] for _, b in BACKBONES)
    kept = [per[b]["kept_sized"]["learned_error"] for _, b in BACKBONES]

    x2, cw_ = 8.75, 3.75
    cy = y + 0.02
    for title, body in [
        ("The control goes to zero",
         f"Predicted size: {fmt3('plain', 'pred_magnitude')} kcal overall; "
         f"{fmt3s('sized', 'pred_magnitude')} inside bands. By construction, and it shows "
         "the bands do their job."),
        ("The head keeps half",
         f"Learned error head: {fmt3('plain', 'learned_error')} overall; "
         f"{fmt3s('sized', 'learned_error')} inside bands, Holm p < 0.01 on all three, and "
         f"now ahead of the control (paired p ≤ {pmax:.2f})."),
        ("The ceiling moves too",
         f"A perfect refuser: {pperf} overall; {sperf} inside bands. About a third of the "
         "headroom at 90% was meal size."),
    ]:
        rect(s, x2 - 0.12, cy, cw_ + 0.24, 1.30, fill=BG)
        t = tb(s, x2, cy + 0.08, cw_, 0.3)
        para(t, title.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        t = tb(s, x2, cy + 0.36, cw_, 0.95)
        para(t, body, size=10.5, color=INK, line=1.22, first=True)
        cy += 1.40
    caption(s, x2, cy - 0.02, cw_,
            "CLIP / SigLIP 2 / DINOv3, 90% answered, change against refusing at random. "
            f"Ten bands, edges from the calibration split’s predictions.", size=9.5)

    t = tb(s, L, 6.22, 7.3, 0.7)
    para(t, f"Inside a size band every gate keeps meals the size random keeps "
            f"({min(kept):.0f}–{max(kept):.0f} kcal against 256). What survives is the "
            "dish, not its size.", size=13, color=INK, bold=True, line=1.3, first=True)
    notes(s, "SHREYA — 40s.\nSame forest plot as before, twice. Top row: refuse one photo "
             "in ten overall. Bottom row: refuse one in ten inside each of ten predicted-"
             "size bands, so a gate cannot win by dropping the big meals. Read the blue "
             "dot: the size control goes from ten kcal to zero on all three backbones, "
             "which is what a control should do. Read the top orange dot: the learned head "
             "goes from eleven to about six, still clear of random after correction, and "
             "now ahead of the control. The green line, the perfect refuser, moves from "
             "twenty to fourteen: a third of the headroom was size too. Ensemble "
             "disagreement loses least on two of three, because it was never mostly size. "
             "Next: where the other half went.")
    return s


def slide_losses(prs, n):
    s = base(prs, "Result 3, completed",
             "Absolute error rewards refusing big meals.", 12)
    y = s._body_top
    pic(s, "degeneracy.png", L - 0.15, y + 0.02, 7.55)
    sel, rs, rw = n["sel"], n["rho_size"], n["rho_within"]
    rel = n["rel"]["gates"]

    x2 = 8.55
    cols, widths = [x2, x2 + 2.05, x2 + 2.8], [2.0, 0.7, 1.15]
    eyebrow(s, x2, y, "Is the signal the size?  [ours]", w=4.0)
    ry = table_head(s, y + 0.34, cols, widths, ["Spearman ρ with", "size", "error, in band"])
    for key, label in [("learned_error", "learned error head"),
                       ("pred_magnitude", "predicted size (control)"),
                       ("ensemble_spread", "ensemble disagreement"),
                       ("perfect", "the true error itself")]:
        t = tb(s, cols[0], ry, widths[0], 0.3)
        para(t, label, size=10.5, color=GREEN if key == "perfect" else INK, first=True)
        t = tb(s, cols[1], ry, widths[1], 0.3)
        para(t, f"{rs[key]:.2f}", size=10.5, color=INK, bold=True, first=True)
        t = tb(s, cols[2], ry, widths[2], 0.3)
        para(t, "—" if key == "perfect" else f"{rw[key]:.2f}", size=10.5, color=INK,
             bold=True, first=True)
        ry += 0.34
    caption(s, x2, ry + 0.02, 3.9,
            "Spearman ρ, clean test photos, CLIP. Right column: mean ρ with the true error "
            "inside predicted-size quartiles.", size=9.5)

    k = lambda g, c: sel[g][c]["kept_mean_kcal"]
    big = lambda g: 100 * sel[g]["0.9"]["big_refused"]
    rp = n["rel"]["gates"]["perfect"]["selectivity"]["0.9"]
    ap = sel["perfect"]["0.9"]
    pm, le, es = (rel[g]["vs_random"]["0.9"] for g in ("pred_magnitude", "learned_error",
                                                       "ensemble_spread"))
    t = strip(s, 4.95, 1.38, "Two losses, two oracles",
          f"at 90% the absolute-error oracle refuses {100 * ap['big_refused']:.0f}% of 400+ "
          f"kcal meals and {100 * ap['small_refused']:.0f}% of small ones; the head and the "
          f"control follow it ({k('learned_error', '0.9'):.0f} / {k('pred_magnitude', '0.9'):.0f} "
          f"kcal kept, against 256). The relative-error oracle (|error| / meal size, floor "
          f"{n['rel']['floor_kcal']} kcal) refuses {100 * rp['big_refused']:.0f}% and "
          f"{100 * rp['small_refused']:.0f}%. Scored that way, no gate beats random at 90%: "
          f"the control is worse ({100 * pm['diff']:+.1f} points, {fmt_p(pm['p_holm'])}), "
          f"the head is null ({100 * le['diff']:+.1f}), ensemble disagreement leans the "
          f"right way ({100 * es['diff']:+.1f}, {fmt_p(es['p_holm'])}).", size=10.5)
    para(t, "Neither loss can judge a gate alone. The within-band score on slide 11 can, "
            "and the kept-set size sits beside every error we report.", size=11.5,
         color=INK, bold=True, space_before=3)
    notes(s, "SHREYA — 30s.\nLeft: the average size of the meals a gate still answers, "
             "against its error; the dashed line is the perfect refuser. At one in ten, "
             "the head, the control and the oracle all keep meals around 220 kcal against "
             "256 overall: the oracle itself is refusing big meals, because absolute error "
             "grows with size. Table: the head's score has rank correlation 0.77 with the "
             "predicted size; inside a size band it keeps 0.18 with the error. Strip: score "
             "on relative error instead and the oracle refuses small meals, and nothing "
             "beats random. Neither loss is safe alone; that is why slide 11 exists. Hand "
             "to Kevin.")
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
    bk, bd, ad = plain["by_kcal"]["coverage"], plain["by_difficulty"]["coverage"], \
        adapt["by_difficulty"]["coverage"]
    t = strip(s, 5.35, 1.50, "Two things this tells us",
              f"(1) The 90% average is {100 * bk[0]:.0f}% on the smaller meals and "
              f"{100 * bk[3]:.0f}% on the largest quarter; the adaptive width barely grows "
              "with size. The same size effect as slides 11 and 12, and the next O3 problem.",
              size=11)
    para(t, f"(2) By difficulty, the adaptive width moves coverage from the easiest quarter "
            f"({100 * bd[0]:.0f}% → {100 * ad[0]:.0f}%) to the hardest "
            f"({100 * bd[3]:.0f}% → {100 * ad[3]:.0f}%) and improves the proper score. "
            f"Shift check: calibrate on clean, deploy on damaged, coverage "
            f"{100 * mc['calibrated_on_clean']['coverage']:.0f}%; recalibrate, "
            f"{100 * mc['calibrated_on_mixture']['coverage']:.0f}% at "
            f"{mc['calibrated_on_mixture']['width_mean'] / mc['calibrated_on_clean']['width_mean'] * 100 - 100:.0f}% "
            "more width. A check, not a finding.", size=11, color=INK, line=1.3,
         space_before=3)
    notes(s, "KEVIN — 40s.\nA range is a promise: the truth is inside it nine times in "
             "ten. Split conformal keeps that promise on average, 89.5 percent. Left chart: "
             "cut by meal size, the promise is 98 percent on small meals and 67 on the "
             "largest quarter. The average is made of two wrong numbers, and it is the "
             "same size effect the gates showed. Right chart: cut by how hard the dish "
             "looks, letting the gate signal set the width moves coverage from the easy "
             "quarter to the hard one. Table: the numbers. The shift check in one "
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
         "Refusing big meals looks like skill. A control gate exposed it; every gate is now "
         "scored again inside predicted-size bands, and the kept-set size is reported "
         "beside every error."),
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
        ("20 Sep", "Literature review", "submitted 20 Sep", GREEN),
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
            "CQR); the within-band threshold as a deployable gate; the deployment mixture "
            "on all backbones; the cost tier (test-time augmentation, a second backbone); "
            "the report.", size=10.5, color=INK, bold=True)

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
             "within a session, so we changed the unit of analysis. The second in one "
             "line: a gate that reads only meal size tied our best one, so every gate is "
             "now scored with size held fixed. Timeline: first milestone on time, two "
             "slipped by three weeks, one dropped with a reason, the ranges early. Next "
             "steps in one breath. Roles as on the proposal; name each member and one "
             "thing they did.")
    return s


def slide_conclusions(prs, n):
    s = base(prs, "Conclusions", "What we know now that we did not in August.", 15)
    y = s._body_top
    per = n["per"]
    le, pm = n["vr"]["learned_error"], n["vr"]["pred_magnitude"]
    sz = [per[b]["sized"]["learned_error"]["diff"] for _, b in BACKBONES]
    for i, (head, body) in enumerate([
        ("Photo quality is not the axis.",
         "Damage is detected almost perfectly and explains 9% of the error; the dish "
         "explains 62%. On clean photos no pixel gate beats random at any answer rate; on "
         f"the damage ladder the most any captures is {100 * n['pixel_gap_pooled_max']:.0f}% "
         "of the gap, by dropping whole damage levels."),
        ("Half of the best gate’s win is meal size. The other half is real.",
         f"At 90% answered the learned error head cuts the error by {-le['diff']:.0f} kcal "
         f"against random; so does reading the predicted size ({-pm['diff']:.0f}), and the "
         f"two tie. Inside predicted-size bands the control goes to zero and the head keeps "
         f"{-max(sz):.0f}–{-min(sz):.0f} kcal (p < 0.01, three backbones)."),
        ("The loss decides which gate looks honest.",
         "Absolute error rewards refusing big meals; relative error rewards refusing small "
         "ones, and there no gate beats random. Ensemble disagreement is the least "
         "size-bound signal and keeps most of its gain inside bands on two of three "
         "backbones."),
        ("Ranges are honest on average, not on large meals.",
         "The largest quarter of meals is covered 67% of the time: the same size effect. "
         "Adaptive width moves coverage from the easiest quarter to the hardest. Size-aware "
         "ranges (CQR) are O3’s job by 11 October."),
    ]):
        rect(s, L - 0.18, y - 0.04, CW + 0.36, 0.90, fill=BG if i % 2 == 0 else None)
        t = tb(s, L, y + 0.02, 0.5, 0.5)
        para(t, str(i + 1), size=18, color=ACCENT, bold=True, first=True)
        t = tb(s, L + 0.55, y + 0.02, CW - 0.6, 0.85)
        para(t, head, size=13.5, color=INK, bold=True, first=True)
        para(t, body, size=11, color=MUTED, line=1.28, space_before=2)
        y += 0.97
    t = tb(s, L, y + 0.10, CW, 1.2)
    para(t, f"So far: an app can tell when it is guessing about a meal of a given size, by "
            f"about {-sum(sz) / 3:.0f} kcal at one refusal in ten, and it already knows the "
            "size. The signals that need no model all fail; the ones that work are a linear "
            "head on features the app already computes.", size=13, color=INK, bold=True,
         line=1.3, first=True)
    para(t, "Every method here is borrowed and cited. The benchmark, the size control, the "
            "sessions and the findings are ours.", size=11.5, color=MUTED, line=1.3,
         space_before=6)
    notes(s, "KEVIN — 30s.\nRead the four bold lines. Then the answer so far: the app can "
             "tell when it is guessing about a meal of a given size, by about six kcal at "
             "one refusal in ten, and half of what the plain curve promised was the size of "
             "the meal, which the app already knows. The cheap signals fail; the working "
             "ones are free if the model is on the device. Last line: everything is "
             "borrowed except the benchmark, the control and what they found. Questions. "
             "Owners: framing Nisarg, splits and metrics Sushant, predictor and pixel "
             "result Keanu, gates, the control and the size test Shreya, ranges and "
             "timeline Kevin.")
    return s


def slide_references(prs, n):
    s = base(prs, "Closing", "Sources, AI use, and where the code is.", 16)
    y = s._body_top
    refs = [
        "[1] Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, “Nutrition5k: "
        "Towards automatic nutritional understanding of generic food,” in Proc. IEEE/CVF CVPR, 2021, "
        "pp. 8899–8907.",
        "[2] B. Coburn, J. He, M. E. Rollo, S. S. Dhaliwal, D. A. Kerr, and F. Zhu, “Comprehensive "
        "evaluation of large multimodal models for nutrition analysis: A new benchmark enriched with "
        "contextual metadata,” arXiv:2507.07048, 2025 (extended version of the IEEE BHI 2025 paper).",
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
    rect(s, L - 0.18, ry + 0.84, CW + 0.36, 0.80, fill=TINT)
    t = tb(s, L + 0.14, ry + 0.94, CW - 0.3, 0.7)
    para(t, [("CODE AND RESULTS   ", {"color": ACCENT, "bold": True, "size": 10.5}),
             (REPO_URL, {"color": INK, "bold": True, "size": 12}),
             ("   ·   every result in this deck is regenerated from results/ by the deck "
              "builder; the oracle budget and the variance split reproduce from "
              "oracle_ladder.py and error_sources.py", {"color": MUTED, "size": 10})],
         first=True)
    notes(s, "Not spoken. Stays up during questions.")
    return s


# ---------------------------------------------------------------- main
def main() -> None:
    missing = [f for f in REQUIRED_FIGS if not (FIGS / f).exists()]
    if missing:
        raise SystemExit(f"missing figures, run figures.py first: {missing}")
    n = load_numbers_v2()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for fn in (slide_title, slide_recap, slide_why, slide_changes, slide_dataset,
               slide_design, slide_reader, slide_predictor, slide_pixel, slide_curve,
               slide_sized, slide_losses, slide_ranges, slide_timeline,
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
