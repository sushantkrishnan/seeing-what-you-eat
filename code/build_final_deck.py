"""Build the final-presentation deck (Presentation 3, 19 Oct 2026).

  ../.venv/bin/python build_final_deck.py      -> 01_FinalResults.pptx

Run gate_probe.py, replicate.py, robustness.py, conformal.py and figures.py first; this
script refuses to build if a figure or results file is missing. Every number on the
slides is read from results/*.json here, except the error-variance split and the
detection AUC, which error_sources.py prints (62.2% / 8.5%; AUC ~1.00).

THEME. The markers asked after the update talk whether the result was conclusive or
could tip either way with more data. The talk answers that: four confirmatory checks
(new sessions, 20 retrainings, one pooled estimate, placebo heads), a verdict claim by
claim, two exploratory ideas and the finished ranges, inside the self-contained summary the brief
asks for (problem, objectives, data, method, design, results, conclusions, limitations,
reflection, contributions).

CLAIMS ARE KEPT TO WHAT THE EVIDENCE BACKS (final-presentation-plan.md, E1-E9):
  * on clean photos no pixel gate beats random; the best case of any is
    under a tenth of a perfect refuser (95% bound); sharpness points the
    wrong way (after Holm, significant on CLIP only)                     holds
  * most of a model-aware gate's win is refusing big meals                holds
  * inside size bands a difficulty signal remains, about a tenth of a
    perfect refuser at 90% answered                                      holds, small
  * the update talk's -6.5 kcal                                          overstated
  * head vs size heuristic                                               whole curve 3/3, 90% 2/3
  * bagged head, backbone disagreement                                   exploratory

SPEAKER NOTES are talking points, not a script: the brief (lane 2) rules out a GenAI
script that substitutes for the presenters' own understanding. Each presenter speaks in
their own words from the points and the numbers on the slide.

House style: short declarative sentences, no aphorisms, few dashes, every number tagged
[ours], [ours, replicated] or a literature reference.
"""
from __future__ import annotations

import json

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_deck import (ACCENT, BG, BLUE, CW, FAINT, GREEN, H, INK, L, MUTED, TINT, W, arrow,
                        base, caption, eyebrow, notes, para, pic, rect, takeaway, tb)
from build_update_deck import FIGS, PUBLISHED_BEST, PUBLISHED_RGB, REPO_URL, ROOT, strip
from build_update_deck_v4 import table_head

RESULTS = ROOT / "results"
OUT = ROOT / "01_FinalResults.pptx"
R_EDGE = L + CW

REQUIRED_FIGS = ["plate_grid.png", "sessions.png", "risk_coverage_pooled.png", "size_split.png",
                 "carves.png", "pooled.png", "stabilised.png", "ranges_size.png"]
BB = ["vit_base_patch16_clip_224.openai", "vit_base_patch16_siglip_224.v2_webli",
      "vit_base_patch16_dinov3.lvd1689m"]

# Speaker, seconds. 675 s = 11:15 spoken, plus four handoffs, inside the 12:00 stop.
SLIDE_PLAN = [
    ("Nisarg", 5), ("Nisarg", 35), ("Nisarg", 45), ("Nisarg", 45),
    ("Shreya", 45), ("Shreya", 45), ("Shreya", 45),
    ("Keanu", 40), ("Keanu", 45), ("Keanu", 50),
    ("Sushant", 55), ("Sushant", 45), ("Sushant", 40),
    ("Kevin", 50), ("Kevin", 45), ("Kevin", 40),
    ("-", 0),
]

# error_sources.py, 507 dishes x 8 damage levels (not written to a results file)
VAR_DISH, VAR_DAMAGE = 62.2, 8.5


def k(x: float, nd: int = 1) -> str:
    """A number with a typographic minus."""
    return f"{x:.{nd}f}".replace("-", "−")


def trio(vals, nd: int = 1, sep=" / ") -> str:
    return sep.join(k(v, nd) for v in vals)


def rng_str(vals, nd: int = 1, abs_=False) -> str:
    v = [abs(x) for x in vals] if abs_ else list(vals)
    lo, hi = min(v), max(v)
    return f"{k(lo, nd)}–{k(hi, nd)}" if round(lo, nd) != round(hi, nd) else k(lo, nd)


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


# ---------------------------------------------------------------- numbers
def load_numbers() -> dict:
    gp = json.loads((RESULTS / "gate_probe.json").read_text())
    rep = json.loads((RESULTS / "replication.json").read_text())
    rob = json.loads((RESULTS / "robustness.json").read_text())
    conf = json.loads((RESULTS / "conformal.json").read_text())
    import csv
    import numpy as np
    n = {}

    # motivation, CLIP clean test, from the one results file
    rows = [r for r in csv.DictReader(open(RESULTS / "gate_scores.csv", encoding="utf-8"))
            if r["rung"] == "clean" and r["backbone"] == BB[0]]
    err = np.array([float(r["cal_abs_err"]) for r in rows])
    true = np.array([float(r["cal_true"]) for r in rows])
    worst = np.sort(err)[::-1][:int(round(0.1 * len(err)))]
    n["mae"] = [gp[b]["predictor"]["cal_mae"] for b in BB]
    n["rho_pixel"] = max(abs(gp[b]["pooled"]["gates"][g]["rho"]) for b in BB
                         for g in ("lap_var", "tenengrad", "blockiness", "hf_ratio"))
    n["mae_ci"] = gp[BB[0]]["predictor"]["cal_mae_ci"]
    n["mean_meal"] = float(true.mean())
    n["worst_tenth_err"] = float(worst.mean())
    n["worst_tenth_share"] = float(worst.sum() / err.sum())
    n["over100"] = float((err > 100).mean())
    perf90 = gp[BB[0]]["clean"]["gates"]["perfect"]["mae"]["0.9"]
    n["perfect90"] = perf90
    n["perfect_cut"] = 1 - perf90 / n["mae"][0]
    n["n_fit"], n["n_calib"], n["n_test"] = (gp[BB[0]]["n_fit"], gp[BB[0]]["n_calib"],
                                             gp[BB[0]]["n_test"])

    # September claim (one carve, test, inside bands)
    n["sept"] = [gp[b]["clean_sized"]["gates"]["learned_error"]["vs_random"]["0.9"] for b in BB]

    # checks 1-2
    n["carve_rank"] = [rep[b]["carves"]["official_rank"]["learned_error_sized"] for b in BB]
    n["carve_avg"] = [rep[b]["carves"]["averaged"]["learned_error_sized"]["mean"] for b in BB]
    n["n_carves"] = len(rep[BB[0]]["carves"]["seeds"])
    n["new_head"] = [rep[b]["clean_sized"]["gates"]["learned_error"]["vs_random"]["0.9"] for b in BB]
    n["new_ens"] = [rep[b]["clean_sized"]["gates"]["ensemble_spread"]["vs_random"]["0.9"] for b in BB]
    n["new_head_sig"] = sum(1 for v in n["new_head"] if v["p_holm"] < 0.05)
    n["new_ens_sig"] = sum(1 for v in n["new_ens"] if v["p_holm"] < 0.05)
    n["new_sessions"] = rep[BB[0]]["n_sessions"]
    n["new_dishes"] = rep[BB[0]]["n"]

    # checks 3-4, pooled
    P = {b: rob[b]["pooled"] for b in BB}
    g10 = lambda b, g, c="0.9": P[b]["10"]["gates"][g]["coverage"][c]
    n["pool_n"], n["pool_clusters"] = P[BB[0]]["10"]["n"], P[BB[0]]["10"]["n_sessions"]
    n["pool_repeats"] = P[BB[0]]["10"]["repeats"]
    n["pool_head"] = [g10(b, "learned_error") for b in BB]
    n["pool_ens"] = [g10(b, "ensemble_spread") for b in BB]
    n["pool_ctrl"] = [g10(b, "pred_magnitude") for b in BB]
    n["pool_perf"] = [g10(b, "perfect")["diff"] for b in BB]
    n["pool_lap"] = [g10(b, "lap_var") for b in BB]
    # equivalence bound: the most any pixel gate could remove (lower 95% end), as a share of
    # what a perfect refuser removes, worst case over the four gates and three backbones
    n["pix_best_share"] = max(-g10(b, g)["ci"][0] / -g10(b, "perfect")["diff"] for b in BB
                              for g in ("lap_var", "tenengrad", "blockiness", "hf_ratio"))
    n["pool_lap_holm"] = sum(1 for v in n["pool_lap"] if v["p_holm"] < 0.05)
    n["pool_share_head"] = [rob[b]["pooled"]["10"]["share_of_perfect_90"]["learned_error"] for b in BB]
    n["pool_share_ens"] = [rob[b]["pooled"]["10"]["share_of_perfect_90"]["ensemble_spread"] for b in BB]
    gn = lambda b, g: P[b]["none"]["gates"][g]
    n["plain_ctrl"] = [gn(b, "pred_magnitude")["coverage"]["0.9"]["diff"] for b in BB]
    n["plain_head"] = [gn(b, "learned_error")["coverage"]["0.9"]["diff"] for b in BB]
    n["kept_ctrl"] = [gn(b, "pred_magnitude")["kept_mean_kcal_90"] for b in BB]
    n["kept_perf"] = [gn(b, "perfect")["kept_mean_kcal_90"] for b in BB]
    n["kept_head"] = [gn(b, "learned_error")["kept_mean_kcal_90"] for b in BB]
    n["b20_ctrl"] = [P[b]["20"]["gates"]["pred_magnitude"]["coverage"]["0.9"]["diff"] for b in BB]
    n["b20_head"] = [P[b]["20"]["gates"]["learned_error"]["coverage"]["0.9"]["diff"] for b in BB]
    n["b20_ens"] = [P[b]["20"]["gates"]["ensemble_spread"]["coverage"]["0.9"]["diff"] for b in BB]
    n["h50"] = [g10(b, "learned_error", "0.5")["diff"] for b in BB]
    n["e50"] = [g10(b, "ensemble_spread", "0.5")["diff"] for b in BB]
    n["head_ctrl_90"] = [P[b]["10"]["pairs_90"]["learned_error - pred_magnitude"] for b in BB]
    n["head_ctrl_aurc"] = [P[b]["10"]["pairs_aurc"]["learned_error - pred_magnitude"] for b in BB]
    n["ens_head_90"] = [P[b]["10"]["pairs_90"]["ensemble_spread - learned_error"] for b in BB]
    pl = [rob[b]["placebo"]["placebo_global"]["averaged"] for b in BB]
    n["plac_mean"] = [x["mean"] for x in pl]
    n["plac_sd"] = [x["sd"] for x in pl]
    n["plac_z"] = [x["z"] for x in pl]
    n["plac_p"] = [x["p"] for x in pl]
    n["n_placebo"] = rob[BB[0]]["placebo"]["n_placebo"]

    # exploratory
    n["sd_head"] = [P[b]["10"]["gates"]["learned_error"]["repeat_sd_90"] for b in BB]
    n["sd_bag"] = [P[b]["10"]["gates"]["learned_error_bag"]["repeat_sd_90"] for b in BB]
    n["bag_aurc"] = [P[b]["10"]["pairs_aurc"]["learned_error_bag - learned_error"] for b in BB]
    n["oof_aurc"] = [P[b]["10"]["pairs_aurc"]["learned_error_oof - learned_error"] for b in BB]
    n["bb_spread"] = [g10(b, "backbone_spread")["diff"] for b in BB]

    # ranges
    rs = {b: rob[b]["ranges"] for b in BB}
    n["rng_const_big"] = [rs[b]["carve_summary"]["constant"]["largest_mean"] for b in BB]
    n["rng_cqr_big"] = [rs[b]["carve_summary"]["cqr"]["largest_mean"] for b in BB]
    n["rng_cqr_small"] = [rs[b]["carve_summary"]["cqr"]["smallest_mean"] for b in BB]
    n["rng_new_cqr_big"] = [rs[b]["new_sessions"]["cqr"]["by_kcal"][3] for b in BB]
    n["rng_new_const_big"] = [rs[b]["new_sessions"]["constant"]["by_kcal"][3] for b in BB]
    n["rng_new_cqr_small"] = [rs[b]["new_sessions"]["cqr"]["by_kcal"][0] for b in BB]
    n["rng_cqr_is"] = [rs[b]["carve_summary"]["cqr"]["interval_score_mean"] for b in BB]
    n["rng_const_is"] = [rs[b]["carve_summary"]["constant"]["interval_score_mean"] for b in BB]
    n["rng_test_const_big"] = [conf[b]["clean"]["constant"]["by_kcal"]["coverage"][3] for b in BB]
    ao = {b: conf[b]["answered_only"] for b in BB}
    n["gate_answered"] = [ao[b]["answered_test"] for b in BB]
    n["gate_cov"] = [ao[b]["ranges"]["cqr"]["answered_calib"]["coverage"] for b in BB]
    n["gate_cov_big"] = [ao[b]["ranges"]["cqr"]["answered_calib"]["by_kcal"]["coverage"][3]
                         for b in BB]
    return n


# ---------------------------------------------------------------- shared pieces
def fact(s, x, y, w, big, body, *, big_col=INK, big_size=26, body_size=11.5):
    t = tb(s, x, y, w, 0.5)
    para(t, big, size=big_size, color=big_col, bold=True, first=True)
    t = tb(s, x, y + big_size / 72 * 1.25 + 0.02, w, 1.2)
    para(t, body, size=body_size, color=INK, line=1.27, first=True)


def talk(s, who, sec, points):
    """Speaker notes as talking points (see module docstring)."""
    notes(s, f"{who.upper()} · {sec} s\n" + "\n".join(f"• {p}" for p in points))


# ---------------------------------------------------------------- slides
def slide_title(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, fill=BG)
    rect(s, 0, 0, 0.16, H, fill=ACCENT)
    t = tb(s, L + 0.25, 1.95, 11.0, 1.2)
    para(t, "Seeing what you eat", size=52, color=INK, bold=True, first=True)
    t = tb(s, L + 0.25, 3.10, 11.0, 0.8)
    para(t, "Can a nutrition app tell when it is guessing? Final results.", size=23,
         color=ACCENT, first=True)
    rect(s, L + 0.25, 4.15, 1.40, 0.037, fill=FAINT)
    t = tb(s, L + 0.25, 4.50, 11.5, 1.6)
    para(t, "Nisarg Patel · Shreya Tigga · Keanu De Cleene · Sushant Krishnan · Kevin Zou",
         size=16, color=INK, first=True)
    para(t, "COMPSCI 760 Advanced Machine Learning · Group 1 · 19 October 2026", size=13,
         color=MUTED, space_before=8)
    para(t, [("Code and results  ", {"color": MUTED, "size": 13}),
             (REPO_URL, {"color": INK, "bold": True, "size": 13})], space_before=8)
    talk(s, "Nisarg", 5, ["Group name, the question in one sentence, hand straight to slide 2."])
    return s


def slide_motivation(prs, n):
    s = base(prs, "Motivation", "Every calorie app answers every photo.", 2)
    y = s._body_top
    pic(s, "plate_grid.png", L, y + 0.05, 3.55)
    caption(s, L, y + 3.68, 3.55, "Nutrition5k dishes [1]: cafeteria meals, every "
            "ingredient weighed.", size=10.5)

    x2, w2 = 4.95, 7.45
    rows = [
        (f"{k(n['mae'][0], 0)} kcal", ACCENT,
         f"is our model’s average miss, on meals that average {n['mean_meal']:.0f} kcal. "
         f"The typical miss is smaller; the trouble is the tail.  [ours]"),
        (f"{n['worst_tenth_err']:.0f} kcal", ACCENT,
         f"is the average miss on its worst tenth of photos: about a whole meal. That tenth "
         f"carries {pct(n['worst_tenth_share'])}% of all the error, and the app shows those "
         f"numbers with the same confidence as the rest.  [ours]"),
        (f"{pct(n['perfect_cut'])}% less", GREEN,
         f"error if the app could decline exactly those photos and answer the other nine in "
         f"ten ({k(n['mae'][0])} → {k(n['perfect90'])} kcal).  [ours]"),
    ]
    ry = y + 0.02
    for big, col, body in rows:
        rect(s, x2, ry, w2, 1.13, fill=BG)
        t = tb(s, x2 + 0.22, ry + 0.17, 2.0, 0.8)
        para(t, big, size=24, color=col, bold=True, first=True)
        t = tb(s, x2 + 2.35, ry + 0.14, w2 - 2.55, 0.95)
        para(t, body, size=11.5, color=INK, line=1.27, first=True)
        ry += 1.25
    takeaway(s, "An app that knew when it was guessing could ask “please check this one” "
                "instead. The latest nutrition benchmark asks for exactly this: "
                "“assessing reliability under degraded conditions” [2].", y=6.1, size=14.5)
    talk(s, "Nisarg", 35, [
        "Why care: people log meals from photos every day, and every app returns one "
        "confident number.",
        f"Our model is off by {k(n['mae'][0], 0)} kcal on average, but on its worst tenth of "
        f"photos by {n['worst_tenth_err']:.0f}: roughly a whole meal.",
        "If it could decline exactly those photos, its error would fall by "
        f"{pct(n['perfect_cut'])}%. That is the prize this project chases.",
        "Point at the quote: the most recent benchmark [2] ends by asking for reliability, "
        "not more accuracy."])
    return s


def slide_problem(prs, n):
    s = base(prs, "Problem and objectives", "We froze the predictor and studied the gate.", 3)
    y = s._body_top
    lw = 5.25
    for i, (lab, body) in enumerate([
        ("Problem", "Before it shows a number, an app has to decide: answer, or ask for help. "
                    "It needs a signal that ranks photos by how wrong the answer will be, "
                    "without knowing the truth."),
        ("Our approach", "Keep the predictor fixed (it matches the published baseline, "
                         "slide 6) and benchmark the gate: every signal an app could use, from "
                         "free pixel checks to model-aware scores, on the error-versus-reject "
                         "protocol [3], between refusing at random and a perfect refuser."),
    ]):
        bh_ = 1.62 if i else 1.25
        rect(s, L, y, lw, bh_, fill=TINT if i else BG)
        t = tb(s, L + 0.2, y + 0.15, lw - 0.4, bh_ - 0.2)
        para(t, lab.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        para(t, body, size=11.5, color=INK, line=1.27, space_before=3)
        y += bh_ + 0.15

    x2 = L + lw + 0.45
    w2 = R_EDGE - x2
    ty = s._body_top
    eyebrow(s, x2, ty, "Objectives, and where they ended", w=w2)
    ty += 0.36
    for code, what, status in [
        ("O1", "Benchmark gate signals on Nutrition5k, three backbones.",
         "Done: ten signals, three more tried since the update. After our first results the "
         "main test moved from damaged photos to clean ones: the dish drives the error."),
        ("O2", "Show which wins are real, with intervals.",
         "Done. Session-level intervals and a size control; since the update, four checks "
         "of whether the result holds."),
        ("O3", "Honest ranges on the photos the app answers.",
         "Done. Size-aware ranges, checked the same way."),
    ]:
        t = tb(s, x2, ty, 0.5, 0.4)
        para(t, code, size=15, color=ACCENT, bold=True, first=True)
        t = tb(s, x2 + 0.55, ty, w2 - 0.55, 1.0)
        para(t, what, size=12, color=INK, bold=True, line=1.2, first=True)
        para(t, status, size=10.5, color=MUTED, line=1.25, space_before=2)
        ty += 0.98
    strip(s, 5.35, 0.92, "Borrowed vs ours",
          "Borrowed: the data and split [1], frozen backbones, error-versus-reject [3], the "
          "corruption ladder [4], the gate signals [6]–[9], conformal ranges [5], [11], [12]. "
          "Ours: the benchmark and its size control, plate sessions as the unit, the "
          "confirmation checks, and every result in this talk.", size=11)
    talk(s, "Nisarg", 45, [
        "Separate the problem (decide whether to answer) from our approach (benchmark the "
        "signals that could make that decision).",
        "Why freeze the predictor: competing on accuracy is a crowded lane; refusing well is "
        "the question we can answer, and a fixed predictor isolates it.",
        "One line per objective, with where it ended. Mention the change: damage photos "
        "became a test bench, clean photos the main test.",
        "Borrowed vs ours: every method is cited; the benchmark and the checks are ours."])
    return s


def slide_feedback(prs, n):
    s = base(prs, "Response to feedback", "You asked if it was conclusive. We checked.", 4)
    y = s._body_top
    sept = n["sept"]
    # The update feedback is quoted verbatim (Presentation3_rubric.txt); the proposal feedback
    # was not recorded word for word, so it is paraphrased and shown without quote marks.
    for i, (when, said, did) in enumerate([
        ("After the proposal", "Asked which evaluation metrics we would use and how we would "
                               "show significance.",
         "One fixed protocol: error at 90% answered and over the whole curve, 95% intervals "
         "from a bootstrap over plate sessions, paired tests against random with a Holm "
         "correction, and a control that reads only the predicted meal size."),
        ("After the update", "“Are the results inconclusive? With p = 0.05 it could go either "
                             "way if we had more data.”",
         f"We had claimed that inside size bands the error head cuts the error by "
         f"{k(-sept[0]['diff'])} kcal (p < 0.01). We tested that claim four ways, tried two "
         f"exploratory ideas, and finished the ranges."),
    ]):
        rect(s, L - 0.18, y - 0.06, CW + 0.36, 1.0, fill=BG if i == 0 else TINT)
        t = tb(s, L, y, 4.1, 0.95)
        para(t, when.upper(), size=10, color=ACCENT, bold=True, first=True)
        para(t, said, size=12, color=INK, bold=True, line=1.18, space_before=2)
        t = tb(s, L + 4.25, y + 0.02, CW - 4.3, 0.95)
        para(t, did, size=11.5, color=INK, line=1.27, first=True)
        y += 1.12

    y += 0.12
    cw2 = 5.5
    eyebrow(s, L, y, "Four checks: does the claim hold?", w=cw2)
    eyebrow(s, L + 6.05, y, "Also new since the update", w=cw2)
    y += 0.36
    checks = [
        ("New data", f"{n['new_sessions']} plate sessions never used to score a gate; the "
                     "hypotheses were written down first."),
        ("Retrain 20 ways", "change which photos calibrate the model, and watch the result."),
        ("One pooled estimate", f"all {n['pool_n']:,} dishes, each scored by models that "
                                f"never saw it, {n['pool_repeats']} refits averaged."),
        ("Placebo", f"{n['n_placebo']} error heads trained on shuffled errors: what does "
                    "luck look like?"),
    ]
    tried = [
        ("A steadier error head", "average 10 heads, each calibrated on different photos. "
                                  "Exploratory."),
        ("Disagreement across backbones", "do three different models disagree on hard "
                                          "photos? Exploratory."),
        ("Size-aware ranges", "objective 3: ranges that widen with the meal, checked the "
                              "same way as the gates."),
    ]
    for col_x, items in ((L, checks), (L + 6.05, tried)):
        ry = y
        for i, (head, body) in enumerate(items):
            t = tb(s, col_x, ry, cw2, 0.55)
            para(t, [(f"{i + 1}  ", {"color": ACCENT, "bold": True, "size": 12}),
                     (head + "  ", {"color": INK, "bold": True, "size": 12}),
                     (body, {"color": MUTED, "size": 11})], line=1.25, first=True)
            ry += 0.55
    talk(s, "Nisarg", 45, [
        "The update feedback is quoted word for word; this slide is our response.",
        "Proposal feedback: we answered it in the update with one fixed protocol "
        "(name it briefly).",
        f"Update feedback: our September headline was {k(-sept[0]['diff'])} kcal inside "
        "size bands. The fair question was whether that survives more data.",
        "Left column: four checks of the claim. Right column: two exploratory ideas, and the "
        "ranges (objective 3), checked the same way. Shreya next: what the data and method "
        "are."])
    return s


def slide_dataset(prs, n):
    s = base(prs, "Dataset and preprocessing", "Nutrition5k: weighed meals, scanned in "
             "sessions.", 5)
    y = s._body_top
    stages = [("4,768", "dishes in the official\nsplit [1]"), ("3,490", "have an overhead\nphoto"),
              ("2,755 / 507", "train / test,\nofficial split"),
              (f"{n['n_fit']:,} / {n['n_calib']:,}", "fit / calibration,\ncarved by session"),
              (f"367 / 127", "plate sessions,\ntrain / test")]
    bw, gap = 2.02, 0.32
    for i, (num, lbl) in enumerate(stages):
        x = L + i * (bw + gap)
        rect(s, x, y, bw, 0.82, fill=TINT if i >= 3 else BG)
        t = tb(s, x, y + 0.09, bw, 0.75)
        para(t, num, size=14, color=ACCENT if i >= 3 else INK, bold=True,
             align=PP_ALIGN.CENTER, first=True)
        para(t, lbl, size=9.5, color=MUTED, align=PP_ALIGN.CENTER, space_before=2, line=1.15)
        if i < len(stages) - 1:
            arrow(s, x + bw + 0.06, y + 0.33, 0.20, 0.16)

    y2 = y + 1.12
    pic(s, "sessions.png", L - 0.05, y2 + 0.32, 5.15)
    eyebrow(s, L, y2, "Scans of one plate arrive ~40 s apart", w=5.2)
    caption(s, L, y2 + 2.75, 5.1, f"Errors within a session are correlated (intraclass "
            f"correlation 0.34), so the session is our unit for every split and interval. "
            f"Train and test scans interleave in time; pooled, they chain into "
            f"{n['pool_clusters']} clusters, which we use as the unit there.", size=10.5,
            color=INK)

    x2 = L + 5.6
    w2 = R_EDGE - x2
    eyebrow(s, x2, y2, "Problems we hit, and what we did", w=w2)
    ry = y2 + 0.36
    for head, body in [
        ("Ragged metadata.", "Variable-length rows; we parse them by anchoring on ingredient "
                             "ids."),
        ("Missing photos.", "Only 3,490 dishes have an overhead photo, so 2,755 / 507 are "
                            "usable."),
        ("Labels are exact sums of weighed ingredients.",
         "That makes a perfect refuser computable. 83 test dishes are under 50 kcal, so "
         "percentage error is unstable: we score absolute kcal."),
        ("No real bad photos.", "Eight simulated damage levels [4]; our “phone” level is an "
                                "imitation, and labelled as one."),
        ("Why this dataset.", "It is the largest public set we found with weighed, per-dish "
                              "calories [1]."),
    ]:
        t = tb(s, x2, ry, w2, 0.6)
        para(t, [(head + "  ", {"color": INK, "bold": True, "size": 11.5}),
                 (body, {"color": MUTED, "size": 11})], line=1.25, first=True)
        ry += 0.6 if len(body) < 90 else 0.78
    talk(s, "Shreya", 45, [
        "Walk the strip left to right: from the official split to the photos we can use, "
        "then the fit / calibration / test carve.",
        "The one idea: repeated scans of the same plate are not independent, so the plate "
        "session is the unit for every split and every interval.",
        "Pooled train + test, sessions chain into clusters; we use the coarser unit, which is "
        "the cautious choice.",
        "Preprocessing problems: pick two (missing photos; exact labels give us a perfect "
        "refuser to compare against).",
        "If asked why no other dataset: the weighed ones we found are much smaller "
        "(NutritionVerse-Real, 889 photos); real phone photos are future work."])
    return s


def slide_method(prs, n):
    s = base(prs, "Method", "One frozen model, many signals, one decision.", 6)
    y = s._body_top
    boxes = [("a photo", "224 × 224", BG, MUTED),
             ("frozen backbone", "CLIP [13],\nSigLIP 2 [14],\nDINOv3 [15]", TINT, INK),
             ("2,304 features", "cached once\nper photo", BG, INK),
             ("ridge head", "closed form →\nkcal estimate", TINT, INK),
             ("gate score", "how risky is\nthis photo?", TINT, INK),
             ("answer + range", "or refuse the\nriskiest 10%", BG, INK)]
    bw, gap, bh = 1.62, 0.34, 1.12
    for i, (head, sub, fill, col) in enumerate(boxes):
        x = L + i * (bw + gap)
        rect(s, x, y, bw, bh, fill=fill)
        t = tb(s, x + 0.08, y + 0.12, bw - 0.16, bh - 0.2)
        para(t, head, size=12.5, color=col, bold=True, align=PP_ALIGN.CENTER, first=True)
        para(t, sub, size=9.5, color=MUTED, align=PP_ALIGN.CENTER, line=1.2, space_before=3)
        if i < len(boxes) - 1:
            arrow(s, x + bw + 0.07, y + bh / 2 - 0.08, 0.20, 0.16)

    y2 = y + bh + 0.25
    lw = 6.4
    eyebrow(s, L, y2, "The signals a gate can use", w=lw)
    ry = y2 + 0.36
    for fam, members in [
        ("No model", "photo sharpness (Laplacian variance [16]), Tenengrad, JPEG blockiness, "
                     "high-frequency ratio"),
        ("Distance", "from the training photos: Mahalanobis [6], nearest neighbours [7]"),
        ("Disagreement", "spread of 32 bootstrap heads [8]"),
        ("Learned", "an error head that predicts |error| from the features [9]"),
        ("Control", "the predicted calorie count alone (ours): does a gate just refuse big "
                    "meals?"),
    ]:
        t = tb(s, L, ry, 1.45, 0.4)
        para(t, fam, size=11.5, color=ACCENT, bold=True, first=True)
        t = tb(s, L + 1.5, ry, lw - 1.5, 0.5)
        para(t, members, size=11, color=INK, line=1.22, first=True)
        ry += 0.47

    x2 = L + lw + 0.5
    w2 = R_EDGE - x2
    eyebrow(s, x2, y2, "Baseline: calories, 507 test dishes", w=w2)
    cols, widths = [x2, x2 + 2.3, x2 + 3.25], [2.25, 0.95, 1.3]
    ty = table_head(s, y2 + 0.36, cols, widths, ["model", "MAE", "source"])
    for what, val, src, hi in [
        ("Inception V2, trained", f"{PUBLISHED_RGB}", "[1]", False),
        ("Ours: CLIP, frozen", k(n["mae"][0]), "[ours, replicated]", True),
        ("Ours: SigLIP 2 / DINOv3", f"{k(n['mae'][1])} / {k(n['mae'][2])}", "[ours]", False),
        ("Best published, needs depth", f"{PUBLISHED_BEST}", "[1]", False),
    ]:
        if hi:
            rect(s, x2 - 0.1, ty - 0.05, w2 + 0.1, 0.38, fill=TINT)
        for c, txt, wd, bold, colr in ((0, what, widths[0], hi, INK),
                                       (1, val, widths[1], True, INK),
                                       (2, src, widths[2], True,
                                        ACCENT if src.startswith("[ours") else MUTED)):
            t = tb(s, cols[c], ty, wd, 0.33)
            para(t, txt, size=11 if c != 2 else 10, color=colr, bold=bold, first=True)
        ty += 0.38
    caption(s, x2, ty + 0.08, w2, f"Our CLIP interval {k(n['mae_ci'][0])}–"
            f"{k(n['mae_ci'][1])} contains the published {PUBLISHED_RGB}, with no training. "
            f"MAE: mean absolute error, kcal.", size=10.5)
    takeaway(s, "A gate only has to rank photos by how wrong they will be. The weighed "
                "truth tells us every photo’s real error, so we can score that exactly.",
             y=6.15, size=13.5)
    talk(s, "Shreya", 45, [
        "Follow the pipeline once: photo, frozen backbone, cached features, a closed-form "
        "ridge head, a gate score, then answer with a range or refuse.",
        "Why it works: a gate is judged only on ranking photos by their real error, which "
        "the weighed labels give us.",
        "Name the families, not every signal. Stress the control: it reads only the "
        "predicted meal size.",
        f"Baseline: our frozen model gets {k(n['mae'][0])} against the published "
        f"{PUBLISHED_RGB}; that is why we could freeze it."])
    return s


def slide_design(prs, n):
    s = base(prs, "Experimental design", "Every split has one job.", 7)
    y = s._body_top
    cols, widths = [L, L + 2.45, L + 4.65], [2.4, 2.15, 6.8]
    ty = table_head(s, y, cols, widths, ["split", "size", "its only job"])
    for split, size, job, hi in [
        ("fit", f"{n['n_fit']:,} dishes", "fit the ridge heads and the gate signals", False),
        ("calibration (validation)", f"{n['n_calib']:,} dishes",
         "pick the ridge penalty; train the error head; set the size bands and range width",
         False),
        ("test", f"{n['n_test']} dishes, 127 sessions", "scored, never tuned on", False),
        ("new sessions (check 1)", f"{n['new_dishes']:,} dishes, {n['new_sessions']} sessions",
         "out-of-fold rerun on train, hypotheses written first", True),
        ("pooled (check 3)", f"{n['pool_n']:,} dishes, {n['pool_clusters']} clusters",
         f"5-fold cross-fit, {n['pool_repeats']} refits averaged", True),
    ]:
        if hi:
            rect(s, L - 0.1, ty - 0.05, CW + 0.1, 0.36, fill=TINT)
        for c, txt in enumerate((split, size, job)):
            t = tb(s, cols[c], ty, widths[c], 0.33)
            para(t, txt, size=11, color=INK, bold=(c == 0), first=True)
        ty += 0.37

    cy = ty + 0.25
    cards = [
        ("Measures", "MAE on the photos the app answers, at 90% answered and as the area "
                     "under the whole risk–coverage curve (AURC [10]), against refusing at "
                     "random and a perfect refuser; again inside 10 predicted-size bands. "
                     "Ranges: coverage by meal size and the interval score [18]."),
        ("Statistics", "95% intervals from 2,000 bootstrap resamples of plate sessions, "
                       "paired across gates; Holm correction across the gates [17]; placebo "
                       "heads trained on shuffled errors."),
        ("Tuning", "One knob, the ridge penalty, picked on calibration from a 25-value grid. "
                   "Fixed in advance: 64 principal components, 10 neighbours, 32 heads; 10 "
                   "size bands (deciles; 5 and 20 also run); 90% answered (an app should "
                   "rarely refuse; the whole curve is reported too)."),
        ("Compute", "One M-series laptop. Features at ~68 photos/s, computed once. Every "
                    "head is closed form, so there is no training loop to converge. The "
                    "whole study reruns in ~100 min: reproduce.sh."),
    ]
    cw_, cg = 2.72, 0.21
    for i, (title, body) in enumerate(cards):
        x = L + i * (cw_ + cg)
        rect(s, x, cy, cw_, 2.25, fill=BG)
        t = tb(s, x + 0.16, cy + 0.12, cw_ - 0.32, 0.3)
        para(t, title.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        t = tb(s, x + 0.16, cy + 0.42, cw_ - 0.32, 1.8)
        para(t, body, size=10.5, color=INK, line=1.27, first=True)
    talk(s, "Shreya", 45, [
        "Read the table as jobs: fit fits, calibration is our validation set and does all "
        "the tuning, test is only scored.",
        "The two shaded rows are new since the update: the data for checks 1 and 3.",
        "Measures: error at 90% answered (an app refusing one photo in ten), and the whole "
        "curve; always against random and a perfect refuser; again inside size bands.",
        "Tuning: one knob, chosen on calibration; everything else fixed in advance, with a "
        "reason: deciles for the size bands (5 and 20 also run), 90% because an app should "
        "rarely refuse.",
        "Compute: a laptop; no training loop, so no convergence curve; everything reruns from "
        "the repo in under two hours."])
    return s


def slide_pixel(prs, n):
    s = base(prs, "Result 1", "Photo quality is not the warning sign.", 8)
    y = s._body_top
    pic(s, "risk_coverage_pooled.png", L - 0.1, y + 0.05, 6.3)
    caption(s, L, y + 4.0, 6.2, "507 test dishes × 8 damage levels, CLIP [ours]. Error "
            "against the share of photos answered; olive, for contrast: a model-aware signal.",
            size=10.5)
    x2 = L + 6.6
    w2 = R_EDGE - x2
    lap = n["pool_lap"]
    fact(s, x2, y, w2, f"{VAR_DAMAGE:.1f}%",
         f"of the error variance comes from photo damage; {VAR_DISH:.0f}% comes from which "
         f"dish it is.  [ours]")
    fact(s, x2, y + 1.25, w2, "AUC ≈ 1",
         f"Area under the detection curve: pixel metrics spot damaged photos almost perfectly, "
         f"yet their rank correlation with the calorie error stays within "
         f"±{n['rho_pixel']:.2f}.  [ours]")
    fact(s, x2, y + 2.5, w2, f"≤ {pct(n['pix_best_share'])}%",
         f"of a perfect refuser’s saving is the most any pixel check could add on all "
         f"{n['pool_n']:,} clean photos (95% bound; 90% answered, inside size bands). "
         f"Sharpness points the wrong way: +{rng_str([v['diff'] for v in lap])} kcal.  [ours]",
         big_col=ACCENT)
    talk(s, "Keanu", 40, [
        "First result, from before the update and now checked on every photo.",
        "The chart: refusing by sharpness tracks the random line; a perfect refuser would "
        "save a lot.",
        f"Why: damage explains only {VAR_DAMAGE:.0f}% of the error; the dish explains "
        f"{VAR_DISH:.0f}%. The metrics see damage, they cannot tell what it costs.",
        "For contrast, the olive line is a model-aware signal (distance from the training "
        "photos): it bends the curve where sharpness does not.",
        f"What we can say: on all {n['pool_n']:,} clean photos no pixel check could be worth "
        f"more than {pct(n['pix_best_share'])}% of a perfect refuser, at 95% confidence. "
        f"Sharpness even points the wrong way on all three backbones (after correcting for "
        f"13 tests, significant on {n['pool_lap_holm']} of 3). A negative result with a "
        f"bound, not a non-result."])
    return s


def slide_size(prs, n):
    s = base(prs, "Result 2", "Most of a gate’s win is refusing big meals.", 9)
    y = s._body_top
    pic(s, "size_split.png", L - 0.05, y + 0.05, 6.4)
    caption(s, L, y + 3.6, 6.3, f"All {n['pool_n']:,} dishes out of fold; bars: mean of three "
            "backbones, dots: each backbone  [ours]", size=10.5)
    x2 = L + 6.75
    w2 = R_EDGE - x2
    t = tb(s, x2, y, w2, 4.0)
    para(t, "Without a size control", size=12, color=ACCENT, bold=True, first=True)
    para(t, f"Refusing the meals with the biggest predicted calories removes "
            f"{rng_str(n['plain_ctrl'], abs_=True)} kcal, more than the learned error head "
            f"({rng_str(n['plain_head'], abs_=True)}). It keeps meals averaging "
            f"{min(n['kept_ctrl']):.0f}–{max(n['kept_ctrl']):.0f} kcal; a perfect refuser "
            f"keeps {min(n['kept_perf']):.0f}–{max(n['kept_perf']):.0f}.", size=11.5,
         line=1.3, space_before=3)
    para(t, "So every gate is scored twice", size=12, color=ACCENT, bold=True, space_before=12)
    para(t, "The second time, it must refuse the same share inside each of 10 predicted-size "
            f"bands. Size can no longer be the lever. The size heuristic drops to under 1 kcal; "
            f"the learned gates keep "
            f"{rng_str([v['diff'] for v in n['pool_head'] + n['pool_ens']], abs_=True)} kcal; "
            f"a perfect refuser keeps about 15.", size=11.5,
         line=1.3, space_before=3)
    talk(s, "Keanu", 45, [
        "The trap: big meals have big errors, so refusing big meals looks like skill.",
        "Grey bars: refuse one photo in ten overall. The predicted-size heuristic beats every "
        "learned gate. But it only answers small meals, which is useless for a diet app.",
        "Orange bars: the same gates, made to refuse the same share inside each size band. "
        "Size drops to under 1 kcal; the gates keep about 2; a perfect refuser 15.",
        "That orange difficulty signal is the claim the markers questioned. Next: is it real?"])
    return s


def slide_checks12(prs, n):
    s = base(prs, "Checks 1 and 2: new data, and the model retrained 20 ways",
             "Our September headline was a lucky draw.", 10)
    y = s._body_top
    pic(s, "carves.png", L - 0.05, y + 0.05, 6.2)
    caption(s, L, y + 2.95, 6.1, f"Error head inside size bands, 90% answered, official test "
            f"split; each dot is one retraining with different calibration photos  [ours]",
            size=10.5)
    x2 = L + 6.55
    w2 = R_EDGE - x2
    sept, ranks = n["sept"], n["carve_rank"]
    t = tb(s, x2, y, w2, 4.5)
    para(t, "Retrained 20 ways", size=12, color=ACCENT, bold=True, first=True)
    para(t, f"The split we reported ({k(sept[0]['diff'])} / {k(sept[1]['diff'])} / "
            f"{k(sept[2]['diff'])} kcal) was the most favourable of {n['n_carves']} on two "
            f"backbones and {ranks[2]}th on the third. The average is "
            f"{trio(n['carve_avg'])}.", size=11.5, line=1.3, space_before=3)
    para(t, f"{n['new_sessions']} new sessions", size=12, color=ACCENT, bold=True,
         space_before=10)
    nh, ne = n["new_head"], n["new_ens"]
    words = {1: "one", 2: "two", 3: "all three"}
    para(t, f"Error head {trio([v['diff'] for v in nh])} (significant on "
            f"{words[n['new_head_sig']]}); ensemble disagreement {trio([v['diff'] for v in ne])} "
            f"kcal (on {words[n['new_ens_sig']]}).", size=11.5, line=1.3, space_before=3)
    para(t, "Why our intervals missed it", size=12, color=ACCENT, bold=True, space_before=10)
    para(t, "They resampled test photos but held the trained model fixed, so they left out "
            "how much the result moves when the model is retrained.", size=11.5, line=1.3,
         space_before=3)
    strip(s, 5.85, 0.62, "Not the effect we reported",
          "the direction holds; the size is about a third of what we said.", size=12)
    talk(s, "Keanu", 50, [
        "Check 1, new data: 367 plate sessions that had never been used to score a gate, "
        "with the hypotheses written down before running.",
        "Check 2: we retrained 20 times, changing only which photos calibrate the model. "
        "Orange is the one we reported. It was the luckiest of 20 on two backbones.",
        "Both checks agree: the direction holds, the size is about a third.",
        "Say plainly why: our intervals resampled the test photos but not the training. "
        "Sushant: what does the best estimate look like?"])
    return s


def slide_checks34(prs, n):
    s = base(prs, "Checks 3 and 4: one pooled estimate, and placebos",
             "With size held fixed, a small signal is real.", 11)
    y = s._body_top
    pic(s, "pooled.png", L - 0.15, y + 0.02, 7.95)
    x2 = L + 8.0
    w2 = R_EDGE - x2
    h, e = n["pool_head"], n["pool_ens"]
    t = tb(s, x2, y, w2, 3.4)
    para(t, "Pooled", size=12, color=ACCENT, bold=True, first=True)
    para(t, f"Error head {trio([v['diff'] for v in h])}, ensemble "
            f"{trio([v['diff'] for v in e])} kcal: significant on all three; about a tenth "
            f"of a perfect refuser ({rng_str(n['pool_perf'], abs_=True)}).", size=11,
         line=1.28, space_before=3)
    para(t, "Placebo", size=12, color=ACCENT, bold=True, space_before=9)
    para(t, f"Heads trained on shuffled errors score about zero (standard deviation ≤ "
            f"{max(n['plac_sd']):.1f}). The real head sits {rng_str(n['plac_z'], abs_=True)} "
            f"standard deviations away; none of {n['n_placebo']} matched it.", size=11,
         line=1.28, space_before=3)
    strip(s, 4.6, 0.98, "And it behaves like a real signal",
          f"With 20 size bands instead of 10, the size heuristic falls to "
          f"{rng_str(n['b20_ctrl'], abs_=True)} kcal while the error head holds "
          f"({trio(n['b20_head'])}). When the app refuses half the photos, the head removes "
          f"{rng_str(n['h50'], abs_=True)} kcal and the ensemble "
          f"{rng_str(n['e50'], abs_=True)}, inside size bands.", size=11)
    caption(s, L, 5.72, CW, f"All {n['pool_n']:,} dishes, each scored by models that never "
            f"saw its session ({n['pool_clusters']} clusters); {n['pool_repeats']} refits "
            f"averaged; 2,000 session resamples; Holm over 13 gates  [ours]", size=10)
    talk(s, "Sushant", 55, [
        "Check 3: every dish we have, each scored by models that never saw it, five refits "
        "averaged so the retraining luck cancels out.",
        "Filled dots differ from random after correction. Error head and ensemble "
        "disagreement are filled on all three backbones, at around 1.5 to 2 kcal; pixel "
        "sharpness sits on the wrong side (filled on CLIP only). The other gates are in the "
        "results file; the two steadier heads come on slide 13.",
        "Scale: a perfect refuser removes about 15. So the signal is about a tenth.",
        "Check 4, placebo: heads trained on shuffled errors average zero; the real head is "
        "5 to 7 standard deviations away. So it is learned, not luck.",
        "Strip: finer bands kill the size heuristic but not the gates; and refusing more "
        "makes the gates more useful."])
    return s


def slide_verdict(prs, n):
    s = base(prs, "Verdict", "Is it conclusive? Claim by claim.", 12)
    y = s._body_top
    hc90 = n["head_ctrl_90"]
    sig90 = sum(1 for v in hc90 if v["ci"][1] < 0)
    sigA = sum(1 for v in n["head_ctrl_aurc"] if v["ci"][1] < 0)
    rows = [
        ("Pixel checks do not tell an app it is guessing.", "Holds", GREEN,
         f"at most {pct(n['pix_best_share'])}% of a perfect refuser (95% bound, three "
         f"backbones); sharpness points the wrong way"),
        ("Most of a gate’s win is refusing big meals.", "Holds", GREEN,
         "size heuristic beats every learned gate without the control"),
        ("With size held fixed, a difficulty signal remains.", "Holds, small", GREEN,
         "about a tenth of a perfect refuser at 90%; 5–7 standard deviations from placebo"),
        ("“Half the win survives the size control” (September).", "Overstated", ACCENT,
         "a favourable training split; the robust value is about a third"),
        ("The error head beats the size heuristic inside size bands.", "Partly", MUTED,
         f"over the whole curve on {sigA} of 3 backbones; at 90% answered on {sig90} of 3"),
        ("Which model-aware gate is best.", "Not separable", MUTED,
         "error head and ensemble disagreement tie inside size bands"),
    ]
    cols, widths = [L, L + 5.35, L + 7.75], [5.2, 2.25, 3.75]
    ty = table_head(s, y, cols, widths, ["claim", "verdict", "evidence"])
    for i, (claim, verdict, col, ev) in enumerate(rows):
        rh = 0.66
        if i % 2 == 0:
            rect(s, L - 0.1, ty - 0.08, CW + 0.1, rh, fill=BG)
        t = tb(s, cols[0], ty, widths[0], rh)
        para(t, claim, size=12, color=INK, bold=True, line=1.2, first=True)
        rect(s, cols[1], ty - 0.01, 2.05, 0.34, fill=col)
        t = tb(s, cols[1], ty + 0.05, 2.05, 0.3)
        para(t, verdict.upper(), size=9.5, color=BG, bold=True, align=PP_ALIGN.CENTER,
             first=True)
        t = tb(s, cols[2], ty, widths[2], rh)
        para(t, ev, size=10.5, color=MUTED, line=1.22, first=True)
        ty += rh + 0.04
    takeaway(s, "On new sessions and under retraining, every claim we still make kept its "
                "direction. Only the September size shrank.", y=ty + 0.18, size=14)
    talk(s, "Sushant", 45, [
        "This is the direct answer to the feedback. Read the verdict column, not every row.",
        "Three claims hold, one of them small. The September size was overstated, "
        "and we say so.",
        "Partly: inside size bands, over the whole range of answer rates the head beats the "
        "size heuristic on all three backbones; at the single 90% point, on two. Without "
        "bands the heuristic wins (slide 9).",
        "Head versus ensemble: we cannot separate them, and we do not claim a winner."])
    return s


def slide_exploratory(prs, n):
    s = base(prs, "Exploratory", "Two things we tried.", 13)
    y = s._body_top
    pic(s, "stabilised.png", L - 0.05, y + 0.05, 6.1)
    caption(s, L, y + 3.32, 6.0, "Kcal removed over the whole answer-rate curve, inside size "
            "bands; bars: 95% interval  [ours]", size=10.5)
    x2 = L + 6.45
    w2 = R_EDGE - x2
    ba = n["bag_aurc"]
    t = tb(s, x2, y, w2, 4.2)
    para(t, "A steadier error head", size=12, color=ACCENT, bold=True, first=True)
    oa = n["oof_aurc"]
    para(t, f"The head changed with every retraining, so we averaged 10 heads, each "
            f"calibrated on different photos (orange). One fitted head becomes steadier (its "
            f"standard deviation across refits falls {n['sd_head'][0]:.2f} → "
            f"{n['sd_bag'][0]:.2f} kcal on CLIP, {n['sd_head'][1]:.2f} → {n['sd_bag'][1]:.2f} "
            f"on SigLIP 2) and removes {rng_str([v['diff'] for v in ba], abs_=True)} kcal more "
            f"over the whole curve on all three (p ≤ {max(v['p'] for v in ba):.3f}). One head "
            f"trained on three times the errors (blue) gains "
            f"{rng_str([v['diff'] for v in oa], abs_=True)} kcal. At 90% answered neither changes "
            f"anything.", size=11, line=1.28, space_before=3)
    bb = n["bb_spread"]
    para(t, "Disagreement across backbones", size=12, color=ACCENT, bold=True,
         space_before=10)
    para(t, f"Best gate for CLIP ({k(bb[0])} kcal), weakest model-aware gate for "
            f"DINOv3 ({k(bb[2])}): it mostly flags the weaker models’ mistakes.",
         size=11, line=1.28, space_before=3)
    strip(s, 5.75, 0.75, "Exploratory, not confirmed",
          "we designed both after seeing the instability and ran each once, with nothing "
          "tuned. They need new photos before we would claim them.", size=11.5)
    talk(s, "Sushant", 40, [
        "Motivated by check 2: the head moved a lot between retrainings.",
        "Averaging 10 heads makes a single fitted head steadier, and over the whole curve "
        "it removes 2 to 3 kcal more than the plain head on every backbone. Blue, one head "
        "trained on three times the errors, does nearly as well. At 90% answered neither "
        "changes anything.",
        "Cross-backbone disagreement is mixed: it helps the weakest model and hurts the "
        "strongest.",
        "Be explicit that both are exploratory. Kevin: the ranges."])
    return s


def slide_ranges(prs, n):
    s = base(prs, "Result 3: confidence ranges", "Ranges that stay honest for big meals.", 14)
    y = s._body_top
    pic(s, "ranges_size.png", L + 0.2, y, 10.9)
    x_cols = [L, L + 3.95, L + 7.9]
    ty = y + 3.75
    blocks = [
        ("Same width for every photo [5]",
         f"Covers about 90% overall, but only {pct(min(n['rng_test_const_big']))}–"
         f"{pct(max(n['rng_test_const_big']))}% of the largest quarter of meals."),
        ("Size-aware ranges, CQR [12]",
         f"Widen with the predicted meal: {pct(min(n['rng_cqr_big']))}–"
         f"{pct(max(n['rng_cqr_big']))}% of the largest meals over 20 retrainings, "
         f"{pct(min(n['rng_new_cqr_big']))}–{pct(max(n['rng_new_cqr_big']))}% on new sessions; "
         f"better interval score. The smallest meals drop to "
         f"{pct(min(n['rng_cqr_small'] + n['rng_new_cqr_small']))}–"
         f"{pct(max(n['rng_cqr_small'] + n['rng_new_cqr_small']))}%."),
        ("With the gate",
         f"A threshold set on calibration answers {pct(min(n['gate_answered']))}–"
         f"{pct(max(n['gate_answered']))}% of test photos; ranges calibrated on answered "
         f"photos cover {pct(min(n['gate_cov']))}–{pct(max(n['gate_cov']))}% of them."),
    ]
    for x, (head, body) in zip(x_cols, blocks):
        t = tb(s, x, ty, 3.65, 1.6)
        para(t, head, size=12, color=ACCENT, bold=True, first=True)
        para(t, body, size=11, color=INK, line=1.28, space_before=3)
    talk(s, "Kevin", 50, [
        "Objective 3: when the app answers, give a range that contains the truth 90% of the "
        "time.",
        "Grey: one width for every photo. Fine on average, but big meals are covered only "
        "about 70% of the time.",
        "Orange: ranges that widen with the predicted meal (conformalised quantile "
        "regression). Big meals come back to about 90%. The cost: small meals drop.",
        "We checked this one the same way: 20 retrainings and new sessions. It holds; our "
        "first number was, if anything, low.",
        "With the gate in front: the threshold answers about 90% of photos as intended, and "
        "the ranges stay honest on the answered ones."])
    return s


def slide_conclusions(prs, n):
    s = base(prs, "Conclusions, limitations and future work", "What we can and cannot claim.",
             15)
    y = s._body_top
    lw = 6.05
    eyebrow(s, L, y, "Conclusions", w=lw)
    ry = y + 0.38
    for i, (head, body) in enumerate([
        ("Photo quality is not the warning sign.",
         f"On clean photos no pixel check beats random; at most "
         f"{pct(n['pix_best_share'])}% of a perfect refuser."),
        ("Most of a gate’s win is meal size.",
         "Any reliability benchmark in this domain needs a size control."),
        ("A real but small difficulty signal remains.",
         "About a tenth of a perfect refuser at 90% answered; more when the app refuses more."),
        ("Size-aware ranges keep big meals honest.", "Small meals pay for it."),
    ]):
        t = tb(s, L, ry, 0.4, 0.4)
        para(t, str(i + 1), size=16, color=ACCENT, bold=True, first=True)
        t = tb(s, L + 0.42, ry, lw - 0.45, 0.8)
        para(t, head, size=12, color=INK, bold=True, first=True)
        para(t, body, size=10.5, color=MUTED, line=1.25, space_before=1)
        ry += 0.86

    x2 = L + lw + 0.45
    w2 = R_EDGE - x2
    eyebrow(s, x2, y, "Limitations → future work", w=w2)
    ry = y + 0.38
    for lim, fut in [
        ("One US cafeteria, one overhead camera.", "Real photos with weighed labels: "
                                                   "NutritionVerse-Real, 889 photos [19]."),
        ("Damage is simulated.", "Nutrition5k’s side cameras: 1,494 more dishes, real "
                                 "viewpoints."),
        ("The within-band signal is small.", "Stronger signals; confirm the steadier head on "
                                             "new photos."),
        ("Size-aware ranges under-cover small meals.", "Ranges that use size and difficulty "
                                                       "together."),
        ("Labels are sums of weighed ingredients.", "Ranges on self-reported labels, like "
                                                    "real logging: SNAPMe [20]."),
    ]:
        rect(s, x2 - 0.1, ry - 0.05, w2 + 0.1, 0.66, fill=BG)
        t = tb(s, x2, ry, w2, 0.65)
        para(t, lim, size=11, color=INK, bold=True, first=True)
        para(t, "→ " + fut, size=10.5, color=MUTED, line=1.2, space_before=1)
        ry += 0.71
    takeaway(s, "An app can learn to flag some photos it gets wrong, beyond their size, but "
                "only a little; and it can say how wrong it might be, honestly, if the range "
                "grows with the meal.", y=6.0, size=13.5)
    talk(s, "Kevin", 45, [
        "Four conclusions, each one line. Say the small one honestly: about a tenth.",
        "Limitations, each paired with the work that would fix it. Lead with: one cafeteria "
        "and one fixed camera.",
        "Side cameras are the cheapest next step: same labels, real viewpoint change, and "
        "1,494 dishes we could not use yet.",
        "Two public sets would test the rest: NutritionVerse-Real (889 real photos, weighed) "
        "and SNAPMe (3,311 phone photos, self-reported labels, like real logging)."])
    return s


def slide_reflection(prs, n):
    s = base(prs, "Reflection and contributions", "What we would do differently.", 16)
    y = s._body_top
    cols = [("Worked well", [
                "Freezing the predictor: every experiment reruns on a laptop in under two "
                "hours.",
                "A control that reads only meal size. It caught a gate that only looked "
                "skilful.",
                "Plate sessions as the unit, from September."]),
            ("We would do differently", [
                "Keep a confirmation set aside from day one and vary the training split "
                "before presenting a headline. Our September number was about three times "
                "too large.",
                "Look for real phone photos earlier."]),
            ("What we learned", [
                "An interval only covers the randomness you resample.",
                "Before trusting a gate, ask what a perfect refuser would refuse.",
                "Test against placebos: a smooth function of the features finds some signal "
                "by luck."])]
    cw_, cg = 3.7, 0.2
    for i, (head, items) in enumerate(cols):
        x = L + i * (cw_ + cg)
        rect(s, x, y, cw_, 2.95, fill=BG if i != 1 else TINT)
        t = tb(s, x + 0.18, y + 0.14, cw_ - 0.36, 2.8)
        para(t, head.upper(), size=10.5, color=ACCENT, bold=True, first=True)
        for it in items:
            para(t, it, size=11, color=INK, line=1.25, space_before=6)

    ry = y + 3.25
    eyebrow(s, L, ry, "Who did what", w=CW)
    ry += 0.34
    for i, (who, did, presents) in enumerate([
        ("Nisarg Patel", "nutrition-estimation papers; survey draft", "slides 1–4"),
        ("Shreya Tigga", "uncertainty and selective-prediction papers; survey draft",
         "slides 5–7"),
        ("Keanu De Cleene", "image-quality and corruption papers; survey draft", "slides 8–10"),
        ("Sushant Krishnan", "pipeline, experiments and analysis; survey curation",
         "slides 11–13"),
        ("Kevin Zou", "conformal-prediction papers; survey draft", "slides 14–16"),
    ]):
        t = tb(s, L + i * 2.32, ry, 2.22, 1.1)
        para(t, who, size=11.5, color=INK, bold=True, first=True)
        para(t, [("Did  ", {"color": ACCENT, "bold": True, "size": 9.5}),
                 (did, {"color": MUTED, "size": 9.5})], space_before=2, line=1.22)
        para(t, [("Presents  ", {"color": ACCENT, "bold": True, "size": 9.5}),
                 (presents, {"color": MUTED, "size": 9.5})], space_before=2, line=1.22)
    talk(s, "Kevin", 40, [
        "One item from each column, in your own words; the slide carries the rest.",
        "Most important lesson: an interval only covers the randomness you resample. That "
        "is exactly what the feedback caught.",
        "Contributions: each person can be asked about their part in Q&A."])
    return s


def slide_references(prs, n):
    s = base(prs, "Closing", "Sources, AI use, and the code.", 17)
    y = s._body_top
    refs = [
        "[1] Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, "
        "“Nutrition5k: Towards automatic nutritional understanding of generic food,” in Proc. "
        "IEEE/CVF CVPR, 2021, pp. 8899–8907.",
        "[2] B. Coburn, J. He, M. E. Rollo, S. S. Dhaliwal, D. A. Kerr, and F. Zhu, "
        "“Comprehensive evaluation of large multimodal models for nutrition analysis: A new "
        "benchmark enriched with contextual metadata,” arXiv:2507.07048, 2025 (extended "
        "version of the IEEE BHI 2025 paper).",
        "[3] P. Grother and E. Tabassi, “Performance of biometric quality measures,” IEEE "
        "Trans. Pattern Anal. Mach. Intell., vol. 29, no. 4, pp. 531–543, Apr. 2007.",
        "[4] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common "
        "corruptions and perturbations,” in Proc. ICLR, 2019.",
        "[5] A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal prediction "
        "and distribution-free uncertainty quantification,” arXiv:2107.07511, 2021.",
        "[6] K. Lee, K. Lee, H. Lee, and J. Shin, “A simple unified framework for detecting "
        "out-of-distribution samples and adversarial attacks,” in Adv. Neural Inf. Process. "
        "Syst. 31, 2018.",
        "[7] Y. Sun, Y. Ming, X. Zhu, and Y. Li, “Out-of-distribution detection with deep "
        "nearest neighbors,” in Proc. ICML, PMLR 162, 2022.",
        "[8] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive "
        "uncertainty estimation using deep ensembles,” in Adv. Neural Inf. Process. Syst. 30, "
        "2017.",
        "[9] C. Corbière, N. Thome, A. Bar-Hen, M. Cord, and P. Pérez, “Addressing failure "
        "prediction by learning model confidence,” in Adv. Neural Inf. Process. Syst. 32, 2019.",
        "[10] Y. Geifman, G. Uziel, and R. El-Yaniv, “Bias-reduced uncertainty estimation for "
        "deep neural classifiers,” in Proc. ICLR, 2019.",
        "[11] H. Papadopoulos, V. Vovk, and A. Gammerman, “Regression conformal prediction "
        "with nearest neighbours,” J. Artif. Intell. Res., vol. 40, pp. 815–840, 2011.",
        "[12] Y. Romano, E. Patterson, and E. J. Candès, “Conformalized quantile regression,” "
        "in Adv. Neural Inf. Process. Syst. 32, 2019.",
        "[13] A. Radford et al., “Learning transferable visual models from natural language "
        "supervision,” in Proc. ICML, PMLR 139, 2021, pp. 8748–8763.",
        "[14] M. Tschannen et al., “SigLIP 2: Multilingual vision-language encoders with "
        "improved semantic understanding, localization, and dense features,” arXiv:2502.14786, "
        "2025.",
        "[15] O. Siméoni et al., “DINOv3,” arXiv:2508.10104, 2025.",
        "[16] J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez, and J. Fernández-Valdivia, "
        "“Diatom autofocusing in brightfield microscopy: A comparative study,” in Proc. ICPR, "
        "vol. 3, 2000, pp. 314–317.",
        "[17] S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. "
        "Statist., vol. 6, no. 2, pp. 65–70, 1979.",
        "[18] T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and "
        "estimation,” J. Amer. Statist. Assoc., vol. 102, no. 477, pp. 359–378, 2007.",
        "[19] C. A. Tai et al., “NutritionVerse-Real: An open access manually collected 2D food "
        "scene dataset for dietary intake estimation,” arXiv:2401.08598, 2023.",
        "[20] J. A. Larke et al., “Surveying Nutrient Assessment with Photographs of Meals "
        "(SNAPMe): A benchmark dataset of food photos for dietary assessment,” Nutrients, "
        "vol. 15, no. 23, Art. no. 4972, 2023.",
        "All figures and tables are ours unless tagged; photos on slide 2 are Nutrition5k "
        "dishes [1], reproduced under the dataset licence.",
    ]
    t = tb(s, L, y - 0.12, CW, 4.2)
    for i, r in enumerate(refs):
        para(t, r, size=7.9, color=INK if i < len(refs) - 1 else MUTED, line=1.1,
             first=(i == 0), space_before=0 if i == 0 else 1.4)
    ry = 5.42
    rect(s, L - 0.18, ry, CW + 0.36, 0.92, fill=BG)
    t = tb(s, L + 0.14, ry + 0.08, CW - 0.3, 0.85)
    para(t, [("GENERATIVE AI USE   ", {"color": ACCENT, "bold": True, "size": 10}),
             ("Claude Code (Anthropic) was used throughout, under the group’s direction: it "
              "wrote most of the pipeline and analysis code, proposed and ran the robustness "
              "checks after we asked how to answer the update feedback, and drafted the "
              "figures, these slides and the speaker notes. The group chose the direction "
              "and which checks to run, and is responsible for every claim; every number on "
              "these slides is read from the results files by the deck builder.",
              {"color": INK, "size": 10})], line=1.26, first=True)
    rect(s, L - 0.18, ry + 1.0, CW + 0.36, 0.55, fill=TINT)
    t = tb(s, L + 0.14, ry + 1.09, CW - 0.3, 0.5)
    para(t, [("CODE AND RESULTS   ", {"color": ACCENT, "bold": True, "size": 10}),
             (REPO_URL, {"color": INK, "bold": True, "size": 12}),
             ("   ·   bash reproduce.sh reruns everything; code/README.md says what is "
              "ours", {"color": MUTED, "size": 10})],
         first=True)
    notes(s, "Not spoken. Stays up during questions.")
    return s


# ---------------------------------------------------------------- main
def main() -> None:
    missing = [f for f in REQUIRED_FIGS if not (FIGS / f).exists()]
    missing += [f for f in ("gate_probe.json", "replication.json", "robustness.json",
                            "conformal.json", "gate_scores.csv") if not (RESULTS / f).exists()]
    if missing:
        raise SystemExit(f"missing inputs, run the pipeline first: {missing}")
    n = load_numbers()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for fn in (slide_title, slide_motivation, slide_problem, slide_feedback, slide_dataset,
               slide_method, slide_design, slide_pixel, slide_size, slide_checks12,
               slide_checks34, slide_verdict, slide_exploratory, slide_ranges,
               slide_conclusions, slide_reflection, slide_references):
        fn(prs, n)
    prs.save(OUT)
    total = sum(sec for _, sec in SLIDE_PLAN)
    per = {}
    for who, sec in SLIDE_PLAN:
        if who != "-":
            per[who] = per.get(who, 0) + sec
    print(f"wrote {OUT}  ({len(prs.slides)} slides)")
    print(f"spoken budget {total // 60}:{total % 60:02d}  " +
          "  ".join(f"{w} {s}s" for w, s in per.items()))


if __name__ == "__main__":
    main()
