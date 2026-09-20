"""Build v3 of the COMPSCI 760 Group 1 update deck: v2's argument in plainer language.

  ../.venv/bin/python build_update_deck_v3.py     -> 01_MethodResults_v3.pptx

Run gate_bench.py, gate_probe.py --all --json, conformal.py --all --json and
figures.py first. Numbers come from load_numbers_v2 (build_update_deck_v2.py); the
speaker notes are read from update-talk-script-v3.md so the script and the notes cannot
drift apart.

WHAT CHANGED FROM v2 (20 Sep, evening):
  * The recap carries the three objectives and their status, no due dates.
  * The phone rung's "late · dropped" row is gone from the changes slide and from the
    timeline; the rung itself stays on slide 9 (labelled an imitation, as before).
  * Every slide's text is rewritten as ordinary sentences: fewer fragments and colons,
    the same numbers, the same one-claim-per-slide shape. Slide 14's headline follows
    the table it now has: one milestone slipped, one moved early.
"""
from __future__ import annotations

import re
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_deck import (ACCENT, BG, CW, FAINT, GREEN, H, INK, L, MUTED, TINT, W, arrow, base,
                        caption, eyebrow, notes, para, pic, rect, table_head, takeaway, tb)
from build_update_deck import (FIGS, PIXEL, PUBLISHED_BEST, PUBLISHED_RGB, REPO_URL, ROOT,
                               TWO_STAGE_KCAL, fmt_ci, fmt_p, strip)
from build_update_deck_v2 import (BACKBONES, REQUIRED_FIGS, fmt_sharp, load_numbers_v2,
                                  slide_references, slide_title)

OUT = ROOT / "01_MethodResults_v3.pptx"
SCRIPT = ROOT / "update-talk-script-v3.md"

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


# ---------------------------------------------------------------- notes from the script
def script_notes() -> dict[int, str]:
    """Slide number -> 'SPEAKER — Ns.' plus the spoken text, from the v3 script."""
    if not SCRIPT.exists():
        return {}
    text = SCRIPT.read_text(encoding="utf-8")
    body = text.split("\n---\n")[1] if "\n---\n" in text else text
    out = {}
    for block in re.split(r"^## ", body, flags=re.M)[1:]:
        head, _, rest = block.partition("\n")
        m = re.match(r"Slide (\d+),.*\((\w+), (\d+) s\)", head)
        if not m:
            continue
        spoken = re.sub(r"\n{2,}", "\n", rest.strip())
        spoken = re.sub(r"(?<!\n)\n(?!\n)", " ", spoken)     # unwrap the hard-wrapped lines
        out[int(m.group(1))] = f"{m.group(2).upper()} — {m.group(3)}s.\n{spoken}"
    return out


def neg(s: str) -> str:
    """Typographic minus for numbers on slides."""
    return s.replace("-", "−")


# ---------------------------------------------------------------- slides
def slide_recap(prs, n, sn):
    s = base(prs, "Recap", "Every app answers every photo. Should it always?", 2)
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
        ("The problem", "Calorie apps give one number for every photo, and they never "
                        "decline."),
        ("What we proposed", "Keep the predictor fixed, and benchmark the signals an app "
                             "could use to decide whether to answer at all, measured "
                             "between refusing at random and a perfect refuser."),
    ]:
        t = tb(s, L, yy, 8.9, 0.28)
        para(t, label.upper(), size=11, color=ACCENT, bold=True, first=True)
        t = tb(s, L, yy + 0.27, 8.9, 0.7)
        para(t, body, size=13, color=INK, line=1.32, first=True)
        yy += 1.05

    x2 = 10.35
    eyebrow(s, x2, y, "Objectives · status", w=3.0)
    ry = y + 0.34
    for name, q, status, col in [
        ("O1 Benchmark", "Which signals predict\nwhen the answer is wrong?", "done", GREEN),
        ("O2 Ceiling", "How much is on the table,\nand on which axis?", "done", GREEN),
        ("O3 Ranges", "Does the best signal give\nranges worth having?", "first cut done",
         ACCENT),
    ]:
        rect(s, x2 - 0.12, ry - 0.06, 2.35, 1.22, fill=BG)
        t = tb(s, x2, ry, 2.2, 1.2)
        para(t, name, size=12, color=INK, bold=True, first=True)
        para(t, q, size=10.5, color=MUTED, line=1.25, space_before=2)
        para(t, status, size=10.5, color=col, bold=True, space_before=3)
        ry += 1.30

    takeaway(s, "Which signals let an app know when it is guessing, and can any of them "
                "run on the phone?", y=6.22, size=15)
    notes(s, sn.get(2, "NISARG — 35s."))
    return s


def slide_why(prs, n, sn):
    s = base(prs, "Why the gate, not the predictor",
             "Refusing one photo in ten beats a better predictor.", 3)
    y = s._body_top
    mae, perf, real = n["mae"], n["g90"]["perfect"], n["g90"]["learned_error"]

    eyebrow(s, L, y, "Our own numbers, 507 test dishes, calorie error in kcal  [ours]", w=9)
    ty = y + 0.42
    th = 2.55
    tiles = [
        (L, 3.15, BG, INK, f"{mae:.0f}", "when we answer every photo", "our frozen predictor"),
        (L + 3.75, 3.15, TINT, GREEN, f"{perf:.0f}",
         "if a perfect refuser declined one photo in ten",
         f"That is {round(mae) - round(perf):.0f} kcal of headroom; our best real gate "
         f"gets to {real:.0f} (slide 10)."),
        (L + 7.90, 3.60, BG, ACCENT, str(TWO_STAGE_KCAL),
         "if we named and weighed the ingredients instead",
         "with the 22% mass error we measure, and one ingredient in five misnamed, which "
         "we assume  [ours, oracle budget]"),
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
            f"The best published number, {PUBLISHED_BEST:.0f} kcal, needed a depth channel "
            f"[1], which most phone photos do not have. The published colour-only result is "
            f"{PUBLISHED_RGB} kcal [1].", size=12, color=INK)
    takeaway(s, "The room to improve is in refusing, not in predicting from colour alone. "
                "That is why we froze the predictor and study the gate.", y=5.85, size=15)
    notes(s, sn.get(3, "NISARG — 30s."))
    return s


def slide_changes(prs, n, sn):
    s = base(prs, "Since the proposal", "What the markers asked, and what we changed.", 4)
    y = s._body_top

    eyebrow(s, L, y, "Feedback we received, and what we did with it", w=8.0)
    ry = y + 0.34
    for fb, did in [
        ("Freezing the predictor and studying the gate is the right design.",
         "We kept it. Nothing on the predictor side has changed."),
        ("Which evaluation metrics, exactly?",
         "One fixed set, shown on slide 7: error at a fixed answer rate, area under the "
         "curve, the share of the gap captured, and what each gate refuses. We then score "
         "every gate a second time with meal size held fixed (slide 11)."),
        ("How will you show a difference is significant?",
         "Every number carries a 95% interval from a bootstrap over plate sessions, and "
         "every gate is tested against refusing at random, paired, with a Holm correction "
         "(slides 10 and 11)."),
    ]:
        if ry < y + 1:
            rect(s, L - 0.18, ry - 0.06, CW + 0.36, 0.60, fill=BG)
        t = tb(s, L, ry, 4.2, 0.75)
        para(t, fb, size=12, color=INK, bold=True, line=1.22, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.75)
        para(t, did, size=11, color=MUTED, line=1.22, first=True)
        ry += 0.76

    ry += 0.04
    eyebrow(s, L, ry, "Changes we made, and why", w=8.0)
    ry += 0.34
    for what, why in [
        ("Calibration is now carved by plate session, not by dish.",
         f"98% of calibration dishes shared a session with a fit dish. We fixed that, and "
         f"the predictor moved from 73.4 to {n['mae']:.1f} kcal, inside its interval  "
         "→ slide 5"),
        ("We report 90% answered and show the whole curve.",
         "That is where an app would sit. The proposal quoted 50%, where every gate looks "
         "better  → slide 7"),
        ("Every gate is scored again inside size bands.",
         "A control that reads only the predicted size tied our best gate. Holding size "
         "fixed separates reading the dish from reading its size  → slide 11"),
        ("Confidence ranges started three weeks early.",
         "The first cut is done, and it found the next problem: large meals  → slide 13"),
    ]:
        t = tb(s, L, ry, 4.2, 0.55)
        para(t, what, size=11.5, color=INK, bold=True, line=1.2, first=True)
        t = tb(s, L + 4.4, ry, 7.1, 0.55)
        para(t, why, size=11, color=MUTED, line=1.2, first=True)
        ry += 0.53

    notes(s, sn.get(4, "NISARG — 30s."))
    return s


def slide_dataset(prs, n, sn):
    s = base(prs, "Dataset", "507 test dishes are 127 plate sessions.", 5)
    y = s._body_top

    pic(s, "plate_grid.png", L, y + 0.02, 2.75)
    caption(s, L, y + 2.86, 2.75,
            "Nutrition5k [1]: overhead photos from one rig, with every ingredient weighed. "
            "That weighed truth is what lets us build a perfect refuser to compare "
            "against.", size=10.5)

    x2 = 4.05
    eyebrow(s, x2, y, "Scans of one plate arrive 40 seconds apart", w=7.5)
    pic(s, "sessions.png", x2, y + 0.30, 6.5)
    caption(s, x2, y + 3.36, 7.45,
            "Errors within a session are correlated (intraclass correlation 0.34), so the "
            "session is our unit for the calibration carve and for every interval in this "
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
    notes(s, sn.get(5, "SUSHANT — 30s."))
    return s


def slide_design(prs, n, sn):
    s = base(prs, "Experimental design", "One frozen model, one linear head, one tuned knob.", 6)
    y = s._body_top

    boxes = [("a photo", "224 × 224", BG, MUTED),
             ("frozen backbone", "CLIP ViT-B/16; SigLIP 2,\nDINOv3 as checks", TINT, INK),
             ("2,304 features", "cached once, per\nphoto and damage level", BG, INK),
             ("standardise", "mean 0, spread 1,\non the fit split", BG, INK),
             ("ridge head", "closed form; one for\nkcal, one for grams", TINT, INK),
             ("“620 kcal”", "plus a gate score\n(slide 10)", BG, INK)]
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
        ("The one tuned knob", "The ridge penalty, alpha, chosen on the calibration split "
                              f"from a 25-point quarter-decade grid. It picked "
                              f"{n['alpha_cal']:,.0f}."),
        ("Fixed in advance", "64 PCA components for the distance gate, 32 ensemble heads, "
                             "and 10 nearest neighbours."),
        ("One results file", "8 damage levels × 507 dishes × 3 backbones = 12,168 rows. "
                             "Every result in this talk is a lookup on that file."),
        ("Compute", "One M-series laptop at about 68 images per second through the "
                    "backbone. The whole pipeline reruns in under 15 minutes."),
    ]
    cw_, cg = 2.72, 0.21
    for i, (title, body) in enumerate(cards):
        x = L + i * (cw_ + cg)
        rect(s, x, cy, cw_, 1.55, fill=BG)
        t = tb(s, x + 0.16, cy + 0.12, cw_ - 0.32, 0.3)
        para(t, title.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        t = tb(s, x + 0.16, cy + 0.44, cw_ - 0.32, 1.1)
        para(t, body, size=11, color=INK, line=1.28, first=True)

    takeaway(s, "There is no training loop, so there is no convergence curve to show, and "
                "nothing was tuned on the test split.", y=cy + 1.55 + 0.35, size=14.5)
    caption(s, L, cy + 1.55 + 0.85, CW,
            "Leak discipline: the heads are fit on the fit dishes; their errors are measured "
            "on the calibration dishes, which they never saw and which share no session with "
            "them; the error head is trained on those errors; and everything is reported on "
            "the 507 test dishes.", size=11)
    notes(s, sn.get(6, "SUSHANT — 30s."))
    return s


def slide_reader(prs, n, sn):
    s = base(prs, "How we score a gate", "How to read every chart that follows.", 7)
    y = s._body_top
    pic(s, "reader.png", L - 0.10, y + 0.02, 6.2)
    caption(s, L, y + 3.82, 6.2,
            "Clean test photos, CLIP [ours]; protocol from [3] and [6]. Random is a "
            "reference, not a bound.", size=10.5)

    x2 = 7.75
    eyebrow(s, x2, y, "A gate is scored on four things", w=4.8)
    ry = y + 0.36
    for i, (title, body) in enumerate([
        ("Error at 90% answered", "The MAE on the photos it still answers, at the operating "
                                  "point. We show the whole curve as well."),
        ("Area under the curve", "AURC, and the share of the random-to-perfect gap the gate "
                                 "closes across all answer rates."),
        ("Rank correlation", "Spearman ρ between the signal and the true error."),
        ("What it refuses", "The mean size of the meals it keeps, and the share of 400+ kcal "
                            "meals it refuses, compared with the perfect refuser. Ours."),
    ]):
        t = tb(s, x2, ry, 4.75, 0.3)
        para(t, [(f"{i + 1}  ", {"color": ACCENT, "bold": True, "size": 12}),
                 (title, {"color": INK, "bold": True, "size": 12})], first=True)
        t = tb(s, x2 + 0.32, ry + 0.28, 4.45, 0.6)
        para(t, body, size=10.5, color=MUTED, line=1.25, first=True)
        ry += 0.86

    strip(s, 6.02, 0.92, "Every number carries an interval",
          f"we resample the {n['n_sessions']} plate sessions {n['n_boot']:,} times and "
          "recompute every curve on the same resample, so the tests are paired. Each gate "
          "is tested against refusing at random and Holm-corrected across the ten "
          "candidates. Every gate is then scored twice more: with meal size held fixed "
          "(slide 11) and on relative error (slide 12).", size=11.5)
    notes(s, sn.get(7, "SUSHANT — 35s."))
    return s


def slide_predictor(prs, n, sn):
    s = base(prs, "Existing results vs ours: the predictor",
             "The published baseline, reproduced without training.", 8)
    y = s._body_top
    pic(s, "backbones.png", L - 0.10, y + 0.12, 6.6)
    caption(s, L, y + 3.15, 6.5,
            f"On the 507 official test dishes CLIP gives {n['mae']:.1f} kcal "
            f"{fmt_ci(n['mae_ci'])}, so the published {PUBLISHED_RGB} sits inside our "
            f"interval. Portion mass: {n['mass']:.1f} g ({n['mass_rel']:.0f}%).",
            size=11.5, color=INK, bold=True)

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

    takeaway(s, "We reproduced the baseline, so the predictor stays frozen. Everything from "
                "here is about the gate.", y=6.15, size=15)
    notes(s, sn.get(8, "KEANU — 30s."))
    return s


def slide_pixel(prs, n, sn):
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
            f"Sharpness rates this cropped photo {sharp_ratio:.0f}× better than the blurred "
            f"one, yet it carries {pan['crop0p4']['mass_abs_err'] / base_p:.0f}× the mass "
            f"error, because the plate rim is the only clue to portion size. Across all 507 "
            f"dishes the ratios are {pr['crop0p4']['lap_var'] / pr['blur6']['lap_var']:.0f}× "
            f"and {pr['crop0p4']['mass_mae'] / base_m:.0f}×. The phone rung is an imitation "
            f"and costs {pr['phone']['cal_mae'] / pr['clean']['cal_mae']:.1f}× on calories.",
            size=11, color=INK, bold=True)

    cx = 7.25
    eyebrow(s, cx, y, "Refuse the worst photos, 8 damage levels [4] pooled  [ours]", w=5.6)
    pic(s, "risk_coverage.png", cx, y + 0.28, 5.1)

    assert not n["pixel_clean_any"], "a pixel gate beat random on clean photos; reword"
    worse = ("sharpness is worse than random on all three backbones"
             if n["sharpness_worse_90_all"] else "sharpness is no better than random")
    strip(s, 5.72, 1.20, "Detected, and it barely helps",
          f"pixel metrics separate damaged photos from clean ones almost perfectly "
          f"(AUC ≈ 1.00), which is the test [2] asks for. Yet over {507 * 8:,} dish × damage "
          f"rows every pixel gate sits within {n['pixel_90_pooled_max']:.0f} kcal of random "
          f"at 90% answered, and {worse}. The most any of them captures, at any answer rate, "
          f"is {100 * n['pixel_gap_pooled_max']:.0f}% of the gap, by dropping whole damage "
          f"levels (|ρ| ≤ {rho_px:.2f}); on clean photos none beats random at all. Which dish "
          "it is explains 62% of the error variance, the damage level 9%, their interaction "
          "29%. A gate has to read the dish.", size=11)
    notes(s, sn.get(9, "KEANU — 40s."))
    return s


def slide_curve(prs, n, sn):
    s = base(prs, "Result 2, completed",
             "Model-aware signals bend the curve. So does size.", 10)
    y = s._body_top
    pic(s, "curves_ci.png", L - 0.10, y + 0.02, 6.2)
    caption(s, L, y + 3.82, 6.2,
            "Clean test photos, CLIP [ours]. The bands are each curve’s own 95% interval; "
            "the tests are paired, so the differences are tighter than the bands suggest "
            "(slide 11).", size=10.5)

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
            txt = neg(f"{d['diff']:+.1f} {fmt_ci(d['ci'])}") + star
            col = (GREEN if d["diff"] < 0 else ACCENT) if sig else MUTED
        else:
            txt, col = "—", FAINT
        para(t, txt, size=10.5, color=col, bold=bool(key in vr and star), first=True)
        ry += rh
    caption(s, x2, ry + 0.06, 4.75,
            "✓ better and ✗ worse than random after Holm correction, p < 0.05. The control "
            "scores risk by the predicted calorie count alone.", size=10)

    le, pm, es = vr["learned_error"], vr["pred_magnitude"], vr["ensemble_spread"]
    pc = n["pair"]["learned_error - pred_magnitude"]
    strip(s, 6.12, 0.82, "What survived",
          f"the learned error head beats random by {-le['diff']:.0f} kcal at 90% answered "
          f"({fmt_p(le['p_holm'])}). So does the control, by {-pm['diff']:.0f}, and the two "
          f"are statistically tied (paired p = {pc['p']:.2f}). Ensemble disagreement clears "
          f"the correction at {-es['diff']:.0f} kcal. Neither distance gate can be told "
          "apart from random.", size=11.5)
    notes(s, sn.get(10, "SHREYA — 35s."))
    return s


def slide_sized(prs, n, sn):
    s = base(prs, "Result 3, completed", "Hold meal size fixed and half the win survives.", 11)
    y = s._body_top
    pic(s, "significance_sized.png", L - 0.15, y - 0.02, 7.2)
    per = n["per"]
    sep = " / "
    fmt3 = lambda blk, g: neg(sep.join(f"{per[b][blk][g]['diff']:+.0f}" for _, b in BACKBONES))
    fmt3s = lambda blk, g: neg(sep.join(f"{per[b][blk][g]['diff']:+.1f}" for _, b in BACKBONES))
    pperf = neg(sep.join(f"{per[b]['perfect_plain']:+.0f}" for _, b in BACKBONES))
    sperf = neg(sep.join(f"{per[b]['perfect_sized']:+.0f}" for _, b in BACKBONES))
    pmax = max(per[b]["head_vs_control_sized"]["p"] for _, b in BACKBONES)
    kept = [per[b]["kept_sized"]["learned_error"] for _, b in BACKBONES]

    x2, cw_ = 8.75, 3.75
    cy = y + 0.02
    for title, body in [
        ("The control goes to zero",
         f"The predicted-size control goes from {fmt3('plain', 'pred_magnitude')} kcal "
         f"overall to {fmt3s('sized', 'pred_magnitude')} inside bands. That is by "
         "construction, and it shows the bands are doing their job."),
        ("The head keeps half",
         f"The learned error head goes from {fmt3('plain', 'learned_error')} overall to "
         f"{fmt3s('sized', 'learned_error')} inside bands, still clear of random on all "
         f"three (Holm p < 0.01) and now ahead of the control (paired p ≤ {pmax:.2f})."),
        ("The ceiling moves too",
         f"Even a perfect refuser goes from {pperf} to {sperf}. About a third of the "
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
            "Ten bands, with edges from the calibration split’s predictions.", size=9.5)

    t = tb(s, L, 6.22, 7.3, 0.7)
    para(t, f"Inside a size band every gate keeps meals about the size random keeps "
            f"({min(kept):.0f}–{max(kept):.0f} kcal against 256), so what survives is the "
            "dish, not its size.", size=13, color=INK, bold=True, line=1.3, first=True)
    notes(s, sn.get(11, "SHREYA — 40s."))
    return s


def slide_losses(prs, n, sn):
    s = base(prs, "Result 3, completed", "Absolute error rewards refusing big meals.", 12)
    y = s._body_top
    pic(s, "degeneracy.png", L - 0.15, y + 0.02, 7.55)
    sel, rs, rw = n["sel"], n["rho_size"], n["rho_within"]
    rel = n["rel"]["gates"]

    x2 = 8.55
    cols, widths = [x2, x2 + 2.05, x2 + 2.8], [2.0, 0.7, 1.15]
    eyebrow(s, x2, y, "Is the signal just the size?  [ours]", w=4.0)
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
            "Spearman ρ on clean test photos, CLIP. The right column is the mean ρ with the "
            "true error inside predicted-size quartiles.", size=9.5)

    k = lambda g, c: sel[g][c]["kept_mean_kcal"]
    rp = n["rel"]["gates"]["perfect"]["selectivity"]["0.9"]
    ap = sel["perfect"]["0.9"]
    pm, le, es = (rel[g]["vs_random"]["0.9"] for g in ("pred_magnitude", "learned_error",
                                                       "ensemble_spread"))
    pm_s, le_s, es_s = (neg(f"{100 * v['diff']:+.1f}") for v in (pm, le, es))
    t = strip(s, 4.95, 1.42, "Two losses, two oracles",
          f"at 90% answered the absolute-error oracle refuses {100 * ap['big_refused']:.0f}% "
          f"of 400+ kcal meals and only {100 * ap['small_refused']:.0f}% of small ones, and "
          f"the head and the control follow it (they keep {k('learned_error', '0.9'):.0f} and "
          f"{k('pred_magnitude', '0.9'):.0f} kcal meals, against 256 overall). The "
          f"relative-error oracle (|error| / meal size, with a {n['rel']['floor_kcal']} kcal "
          f"floor) refuses {100 * rp['big_refused']:.0f}% and {100 * rp['small_refused']:.0f}%. "
          f"Scored that way, no gate beats random at 90%: the control is worse ({pm_s} "
          f"points, {fmt_p(pm['p_holm'])}), the head is no different from random ({le_s}), "
          f"and only ensemble disagreement leans the right way ({es_s}, "
          f"{fmt_p(es['p_holm'])}).", size=10.5)
    para(t, "Neither loss can judge a gate on its own. The within-band score on slide 11 "
            "can, and we report the kept-set size beside every error.", size=11.5,
         color=INK, bold=True, space_before=3)
    notes(s, sn.get(12, "SHREYA — 30s."))
    return s


def slide_ranges(prs, n, sn):
    s = base(prs, "Result 4, first cut", "Ranges: honest on average, not on large meals.", 13)
    y = s._body_top
    c = n["conf"]
    plain, adapt = c["constant"], c["ensemble_spread"]
    pic(s, "conformal_kcal.png", L - 0.15, y + 0.02, 7.6)
    caption(s, L, y + 2.98, 7.4,
            f"Split conformal [5] with a 90% target, {n['n_calib']} calibration dishes and "
            f"507 clean test photos, CLIP [ours].", size=10.5)

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
    caption(s, x2, ry + 0.04, 3.6, "The interval score is the width plus a penalty for "
                                   "misses; lower is better.", size=9.5)

    mc = n["mix"]["uniform"]["constant"]
    bk, bd, ad = plain["by_kcal"]["coverage"], plain["by_difficulty"]["coverage"], \
        adapt["by_difficulty"]["coverage"]
    t = strip(s, 5.35, 1.50, "Two things this tells us",
              f"(1) The 90% average hides {100 * bk[0]:.0f}% on the smaller meals and "
              f"{100 * bk[3]:.0f}% on the largest quarter, and the adaptive width barely "
              "grows with size. It is the same size effect as on slides 11 and 12, and it is "
              "the next O3 problem.", size=11)
    para(t, f"(2) Cut by difficulty, the adaptive width moves coverage from the easiest "
            f"quarter ({100 * bd[0]:.0f}% → {100 * ad[0]:.0f}%) to the hardest "
            f"({100 * bd[3]:.0f}% → {100 * ad[3]:.0f}%) and improves the proper score. As a "
            f"check: calibrated on clean photos and deployed on damaged ones, coverage falls "
            f"to {100 * mc['calibrated_on_clean']['coverage']:.0f}%; recalibrated, it returns "
            f"to {100 * mc['calibrated_on_mixture']['coverage']:.0f}% at "
            f"{mc['calibrated_on_mixture']['width_mean'] / mc['calibrated_on_clean']['width_mean'] * 100 - 100:.0f}% "
            "more width. That is a check, not a finding.", size=11, color=INK, line=1.3,
         space_before=3)
    notes(s, sn.get(13, "KEVIN — 40s."))
    return s


def slide_timeline(prs, n, sn):
    s = base(prs, "Challenges, timeline, roles", "One milestone slipped, one moved early.", 14)
    y = s._body_top

    eyebrow(s, L, y, "Two challenges, and what we did", w=5.5)
    ry = y + 0.34
    for what, did in [
        ("Correlated test dishes",
         "Errors cluster by plate session (ICC 0.34), so the session is now the unit of "
         "resampling and the calibration carve respects it."),
        ("A gate that cheats",
         "Refusing big meals looks like skill. A control gate exposed that, so every gate "
         "is now scored again inside predicted-size bands, and we report the kept-set size "
         "beside every error."),
    ]:
        t = tb(s, L, ry, 5.3, 0.3)
        para(t, what, size=12, color=INK, bold=True, first=True)
        t = tb(s, L, ry + 0.30, 5.3, 0.8)
        para(t, did, size=11, color=MUTED, line=1.28, first=True)
        ry += 1.15
    caption(s, L, ry + 0.05, 5.3,
            "Also: a rank-deficient covariance (2,304 dimensions on 1,924 dishes, so "
            "Mahalanobis runs on 64 components, fixed in advance), and no phone photos in "
            "the data (our phone rung is an imitation, and labelled as one).", size=10)

    x2 = 6.85
    eyebrow(s, x2, y, "Proposal timeline vs actual", w=5.5)
    cols, widths = [x2, x2 + 1.35, x2 + 4.15], [1.3, 2.75, 1.5]
    ty = table_head(s, y + 0.34, cols, widths, ["planned", "milestone", "status"])
    for when, what, status, col in [
        ("17 Aug", "Pipeline, 3 backbones, first results", "done", GREEN),
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
    caption(s, x2, ty + 0.06, 5.6,
            "Next: size-aware ranges for large meals (conformalised quantile regression); the "
            "within-band threshold as a deployable gate; the deployment mixture on all "
            "backbones; the costlier tier (test-time augmentation, a second backbone); and "
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
    notes(s, sn.get(14, "KEVIN — 25s."))
    return s


def slide_conclusions(prs, n, sn):
    s = base(prs, "Conclusions", "What we know now that we did not in August.", 15)
    y = s._body_top
    per = n["per"]
    le, pm = n["vr"]["learned_error"], n["vr"]["pred_magnitude"]
    sz = [per[b]["sized"]["learned_error"]["diff"] for _, b in BACKBONES]
    for i, (head, body) in enumerate([
        ("Photo quality is not the axis.",
         "We detect damage almost perfectly, yet it explains only 9% of the error while the "
         "dish explains 62%. On clean photos no pixel gate beats random at any answer rate, "
         f"and on the damage ladder the most any captures is "
         f"{100 * n['pixel_gap_pooled_max']:.0f}% of the gap, by dropping whole damage "
         "levels."),
        ("Half of the best gate’s win is meal size. The other half is real.",
         f"At 90% answered the learned error head cuts the error by {-le['diff']:.0f} kcal "
         f"against random, but so does simply reading the predicted size ({-pm['diff']:.0f}), "
         f"and the two tie. Inside predicted-size bands the control drops to zero while the "
         f"head keeps {-max(sz):.0f}–{-min(sz):.0f} kcal (p < 0.01 on three backbones)."),
        ("The loss decides which gate looks honest.",
         "Absolute error rewards refusing big meals; relative error rewards refusing small "
         "ones, and under it no gate beats random. Ensemble disagreement is the signal least "
         "tied to size, and it keeps most of its gain inside bands on two of three "
         "backbones."),
        ("Ranges are honest on average, not on large meals.",
         "The largest quarter of meals is covered only 67% of the time, the same size effect "
         "again. Adaptive width moves coverage from the easiest quarter to the hardest. "
         "Size-aware ranges are O3’s job by 11 October."),
    ]):
        rect(s, L - 0.18, y - 0.04, CW + 0.36, 0.90, fill=BG if i % 2 == 0 else None)
        t = tb(s, L, y + 0.02, 0.5, 0.5)
        para(t, str(i + 1), size=18, color=ACCENT, bold=True, first=True)
        t = tb(s, L + 0.55, y + 0.02, CW - 0.6, 0.85)
        para(t, head, size=13.5, color=INK, bold=True, first=True)
        para(t, body, size=11, color=MUTED, line=1.28, space_before=2)
        y += 0.97
    t = tb(s, L, y + 0.10, CW, 1.2)
    para(t, f"So far: an app can tell when it is guessing about a meal of a given size, "
            f"worth about {-sum(sz) / 3:.0f} kcal at one refusal in ten, and it already knows "
            "the size. The signals that need no model all fail; the ones that work are a "
            "linear head on features the app already computes.", size=13, color=INK,
         bold=True, line=1.3, first=True)
    para(t, "Every method here is borrowed and cited. The benchmark, the size control, the "
            "sessions and the findings are ours.", size=11.5, color=MUTED, line=1.3,
         space_before=6)
    notes(s, sn.get(15, "KEVIN — 30s."))
    return s


# ---------------------------------------------------------------- main
def main() -> None:
    missing = [f for f in REQUIRED_FIGS if not (FIGS / f).exists()]
    if missing:
        raise SystemExit(f"missing figures, run figures.py first: {missing}")
    n = load_numbers_v2()
    sn = script_notes()
    if not sn:
        print(f"note: {SCRIPT.name} not found; speaker notes carry only the speaker and time")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    slide_title(prs)
    for fn in (slide_recap, slide_why, slide_changes, slide_dataset, slide_design,
               slide_reader, slide_predictor, slide_pixel, slide_curve, slide_sized,
               slide_losses, slide_ranges, slide_timeline, slide_conclusions):
        fn(prs, n, sn)
    slide_references(prs, n)
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
