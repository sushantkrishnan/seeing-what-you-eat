"""Figures for the proposal deck. Run gate_bench.py and gate_probe.py --all --json first.

  ../.venv/bin/python figures.py

Writes into results/figures/:
  panel_clean/blur6/crop04.png  the crop-vs-blur photo strip          (slide 8)
  risk_coverage.png             the error-versus-reject curve         (slide 8)
  dataset_scatter.png           mass vs calories, why weight is not enough  (slide 6)
  plate_grid.png                nine real dishes                      (slide 6)
  ladder_visible/_invisible.png  the eight damage families, grouped   (slide 10)
  backbones.png                 three frozen backbones compared       (slide 7)
  gates.png                     every gate at 50% coverage            (slide 9)
  degeneracy.png                which gates win by refusing big meals (slide 9)
Update deck (build_update_deck.py) additionally uses:
  curves_ci.png, significance.png, sessions.png, panel_phone.png, backbones.png,
  reader.png                    the risk–coverage axes with only reference and ceiling (slide 7)
  significance_sized.png        the forest plot plain and inside predicted-size bands (v2 slide 11)
  conformal_kcal.png            coverage by true meal size beside by-difficulty (slide 13)
Final deck additionally uses:
  replication.png               the within-band result: as reported, carve-averaged, new data
  carves.png                    twenty training carves, the reported one highlighted
  ranges_size.png               coverage by true meal size, same width vs CQR, three backbones
  pooled.png                    the best estimate: the main gates inside size bands, all dishes
                                out of fold, five refits averaged (robustness.py)
  size_split.png                each gate's win overall vs inside size bands (pooled)
  stabilised.png                whole-curve gain: plain, bagged and 3x-error heads

Series colours are validated for colour-vision deficiency (OKLab dE, adjacent pairs,
scripts/validate_palette.js) and every mark is directly labelled, so identity never
rests on colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import n5k
from degrade import apply_degradation

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = RESULTS / "figures"
PRIMARY = "vit_base_patch16_clip_224.openai"

# The dish on slides 6, 8 and 10. A round plate on a dark tray: the rim and the table
# edge are the only scale references in a fixed-camera overhead shot, and crop:0.4
# removes both.
DEMO_DISH = "dish_1565811091"

INK, MUTED, GRID, FAINT = "#1B1E28", "#6B7180", "#E0DAD2", "#C0BAB2"
# Validated categorical set (validate_palette.js, light mode). The trio passes all
# checks; OLIVE was added 17 Sep 2026 and the four pass all-pairs at the CVD floor
# (worst pair is green–orange, dE 7.1), which is legal because every series here is
# direct-labelled and line-styled, never identified by colour alone.
GREEN, ORANGE, BLUE, OLIVE = "#0E7C5A", "#C45A1B", "#2F5FA8", "#9E9D24"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _clean(ax, yaxis_grid=True):
    if yaxis_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _save(fig, name):
    fig.savefig(FIGS / name, facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- slide 6, dataset
def dataset_scatter() -> None:
    """Mass against calories. Weighing the food gets you r=0.76 and no further."""
    d = n5k.load_dishes()
    ids = (n5k.usable_split("train", local_only=True)
           + n5k.usable_split("test", local_only=True))
    mass = np.array([d[i].mass for i in ids])
    cal = np.array([d[i].cal for i in ids])
    band = (mass >= 150) & (mass < 250)

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=200)
    ax.axvspan(150, 250, color=ORANGE, alpha=0.10, zorder=1)
    ax.scatter(mass[~band], cal[~band], s=7, color=MUTED, alpha=0.30,
               linewidths=0, zorder=2)
    ax.scatter(mass[band], cal[band], s=8, color=ORANGE, alpha=0.55,
               linewidths=0, zorder=3)

    lo, hi = cal[band].min(), cal[band].max()
    ax.annotate("", xy=(200, lo), xytext=(200, hi),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3), zorder=4)
    ax.text(272, 855, f"{band.sum()} dishes weigh 150–250 g.\n"
                      f"Their calories run {lo:.0f} to {hi:.0f}.",
            fontsize=10.5, color=INK, fontweight="bold", linespacing=1.5, va="top")

    ax.set_xlabel("mass on the scale (grams)", fontsize=10.5)
    ax.set_ylabel("calories (kcal)", fontsize=10.5)
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 900)
    _clean(ax)
    fig.tight_layout(pad=0.4)
    _save(fig, "dataset_scatter.png")


def plate_grid(n=9) -> None:
    """Nine real dishes, so the audience sees what the data is."""
    d = n5k.load_dishes()
    test = n5k.usable_split("test", local_only=True)
    picks = [i for i in test if 150 < d[i].mass < 500 and len(d[i].ingredients) >= 3]
    chosen = [picks[i] for i in (4, 16, 19, 11, 15, 22, 0, 7, 12)][:n]

    th, gap = 300, 10
    side = int(n ** 0.5)
    sheet = Image.new("RGB", (side * th + (side - 1) * gap,) * 2, "white")
    for k, did in enumerate(chosen):
        im = Image.open(n5k.IMAGES / did / "rgb.png").convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        sheet.paste(im.resize((th, th), Image.LANCZOS),
                    ((k % side) * (th + gap), (k // side) * (th + gap)))
    sheet.save(FIGS / "plate_grid.png")
    print("  plate_grid.png")


# ---------------------------------------------------------------- slide 10, ladder
# The grouping IS the experiment, so the two rows ship as separate strips: the deck
# labels one "damage sharpness can see" and the other "damage it cannot", and the
# claim is that the second group drives the error while only the first moves the metrics.
LADDER_VISIBLE = [("blur:6", "blur"), ("motion:17", "camera shake"),
                  ("downscale:8", "shrunk"), ("noise:0.16", "noise")]
LADDER_INVISIBLE = [("crop:0.4", "cropped"), ("dark:0.2", "dark"),
                    ("temp:0.5", "colour shift"), ("jpeg:10", "compressed")]


def ladder() -> None:
    """One dish, eight damage families, split by whether a sharpness metric sees them."""
    src = Image.open(n5k.IMAGES / DEMO_DISH / "rgb.png").convert("RGB")
    w, h = src.size
    s = min(w, h)
    src = src.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))

    th, gap = 300, 9
    for name, fams in [("ladder_visible", LADDER_VISIBLE),
                       ("ladder_invisible", LADDER_INVISIBLE)]:
        strip = Image.new("RGB", (len(fams) * th + (len(fams) - 1) * gap, th), "white")
        for k, (spec, _) in enumerate(fams):
            strip.paste(apply_degradation(src, spec).resize((th, th), Image.LANCZOS),
                        (k * (th + gap), 0))
        strip.save(FIGS / f"{name}.png")
        print(f"  {name}.png   " + " | ".join(lbl for _, lbl in fams))


# ---------------------------------------------------------------- slide 8, reject curve
def risk_coverage_chart() -> None:
    data = json.loads((RESULTS / "gate_bench.json").read_text())["risk_coverage_dense"]

    def xy(key):
        pts = sorted((float(k), v) for k, v in data[key].items())
        return [100 * p[0] for p in pts], [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
    series = [("oracle", GREEN, "-", 2.4, "a perfect refuser"),
              ("mahalanobis", ORANGE, "-", 2.4, "distance to mean (model-aware)"),
              ("lap_var", BLUE, "-", 2.0, "sharpness gate"),
              ("random", MUTED, (0, (4, 3)), 1.8, "refusing at random")]
    for key, colour, style, width, _ in series:
        x, y = xy(key)
        ax.plot(x, y, color=colour, linestyle=style, linewidth=width,
                solid_capstyle="round", zorder=3)

    anchors = {"lap_var": (62.0, 4, 10), "random": (14.0, 0, -16),
               "mahalanobis": (36.0, 0, -17), "oracle": (70.0, 0, 10)}
    for key, colour, _, _, label in series:
        x, y = xy(key)
        ax_x, dx, dy = anchors[key]
        i = min(range(len(x)), key=lambda j: abs(x[j] - ax_x))
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(dx, dy),
                    color=colour, fontsize=10.5, fontweight="bold", zorder=4)

    ax.set_xlabel("share of photos the app still answers", fontsize=10.5)
    ax.set_ylabel("calorie error on those photos (kcal)", fontsize=10.5)
    ax.set_xlim(8, 102)
    ax.set_ylim(0, 125)
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.set_xticklabels([f"{t}%" for t in [10, 25, 50, 75, 100]], fontsize=10)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    _clean(ax)

    xs, y_or = xy("oracle")
    _, y_lp = xy("lap_var")
    i50 = min(range(len(xs)), key=lambda j: abs(xs[j] - 50.0))
    ax.annotate("", xy=(50, y_or[i50]), xytext=(50, y_lp[i50]),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2,
                                shrinkA=2, shrinkB=2), zorder=5)
    ax.text(12.5, 74, f"Refuse half the photos and a perfect\nrefuser saves "
                      f"{y_lp[i50] - y_or[i50]:.0f} kcal. No pixel\nsignal gets any of it.",
            fontsize=10.5, color=INK, fontweight="bold", linespacing=1.5,
            ha="left", va="top", zorder=5)
    fig.tight_layout(pad=0.4)
    _save(fig, "risk_coverage.png")


def risk_coverage_pooled(model=PRIMARY) -> None:
    """The reject curve on the pooled damage ladder, from gate_probe.json.

    Replaces risk_coverage_chart() on the update deck (20 Sep, v4): the random reference
    is the analytic one (flat at the answer-all error), the distance gate takes the olive
    the other update-deck charts give the distance gates, and the annotation states what
    sharpness saves at 50% instead of claiming it saves nothing.
    """
    res = json.loads((RESULTS / "gate_probe.json").read_text())[model]
    covs = res["coverages"]
    g = res["pooled"]["gates"]

    def xy(key):
        pts = sorted((c, g[key]["mae"][str(c)]) for c in covs)
        return [100 * p[0] for p in pts], [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
    series = [("perfect", GREEN, "-", 2.4, "a perfect refuser"),
              ("mahalanobis", OLIVE, "-", 2.4, "distance to mean\n(model-aware)"),
              ("lap_var", BLUE, "-", 2.0, "sharpness gate"),
              ("random", MUTED, (0, (4, 3)), 1.8, "refusing at random")]
    for key, colour, style, width, _ in series:
        x, y = xy(key)
        ax.plot(x, y, color=colour, linestyle=style, linewidth=width,
                solid_capstyle="round", zorder=3)

    anchors = {"lap_var": (68.0, 4, 9), "random": (12.0, 0, 7),
               "mahalanobis": (13.0, 0, -30), "perfect": (72.0, 0, 10)}
    for key, colour, _, _, label in series:
        x, y = xy(key)
        ax_x, dx, dy = anchors[key]
        i = min(range(len(x)), key=lambda j: abs(x[j] - ax_x))
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(dx, dy),
                    color=colour, fontsize=10.5, fontweight="bold", zorder=4)

    ax.set_xlabel("share of photos the app still answers", fontsize=10.5)
    ax.set_ylabel("calorie error on those photos (kcal)", fontsize=10.5)
    ax.set_xlim(8, 102)
    ax.set_ylim(0, 125)
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.set_xticklabels([f"{t}%" for t in [10, 25, 50, 75, 100]], fontsize=10)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    _clean(ax)

    i50 = covs.index(0.5)
    rnd, prf, lap = (g[k]["mae"]["0.5"] for k in ("random", "perfect", "lap_var"))
    ax.annotate("", xy=(50, prf), xytext=(50, rnd),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2,
                                shrinkA=2, shrinkB=2), zorder=5)
    ax.text(12.5, 62, f"Refuse half the photos: a perfect\nrefuser saves {rnd - prf:.0f} kcal, "
                      f"sharpness\nsaves {rnd - lap:.0f}.",
            fontsize=10.5, color=INK, fontweight="bold", linespacing=1.5,
            ha="left", va="top", zorder=5)
    fig.tight_layout(pad=0.4)
    _save(fig, "risk_coverage_pooled.png")


def photo_panels() -> None:
    src = Image.open(n5k.IMAGES / DEMO_DISH / "rgb.png").convert("RGB")
    for name, spec in [("clean", None), ("blur6", "blur:6"), ("crop04", "crop:0.4")]:
        img = src if spec is None else apply_degradation(src, spec)
        img.resize((720, 540), Image.LANCZOS).save(FIGS / f"panel_{name}.png")
    print("  panel_clean.png / panel_blur6.png / panel_crop04.png")


# ---------------------------------------------------------------- slide 7, backbones
def backbones() -> None:
    """Three frozen backbones. Only DINOv3 moves the mass number, which is the point."""
    res = json.loads((RESULTS / "gate_probe.json").read_text())
    names = {"vit_base_patch16_clip_224.openai": "CLIP\n2021",
             "vit_base_patch16_siglip_224.v2_webli": "SigLIP 2\n2025",
             "vit_base_patch16_dinov3.lvd1689m": "DINOv3\n2025"}
    keys = [k for k in names if k in res]
    labels = [names[k] for k in keys]
    cal = [res[k]["predictor"]["cal_mae"] for k in keys]
    cal_ci = [res[k]["predictor"].get("cal_mae_ci", [v, v]) for k, v in zip(keys, cal)]
    mass = [100 * res[k]["predictor"]["mass_rel"] for k in keys]
    colours = [MUTED, MUTED, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=200)
    for ax, vals, title, unit, ref in [
            (axes[0], cal, "calories", "kcal error", 70.6),
            (axes[1], mass, "portion size", "% error", None)]:
        bars = ax.bar(range(len(vals)), vals, width=0.6, color=colours, zorder=3)
        if vals is cal:
            ax.errorbar(range(len(vals)), vals,
                        yerr=[[v - lo for v, (lo, hi) in zip(vals, cal_ci)],
                              [hi - v for v, (lo, hi) in zip(vals, cal_ci)]],
                        fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=5)
        for i, (b, v) in enumerate(zip(bars, vals)):
            top = cal_ci[i][1] + 1.5 if vals is cal else v + max(vals) * 0.035
            ax.text(b.get_x() + b.get_width() / 2, top,
                    f"{v:.1f}", ha="center", fontsize=11, fontweight="bold", color=INK)
        if ref:
            ax.axhline(ref, color=ORANGE, linestyle=(0, (4, 3)), linewidth=1.6, zorder=4)
            ax.text(0.5, ref + 1.2, f"published {ref}", ha="center", va="bottom",
                    fontsize=9.5, color=ORANGE, fontweight="bold", zorder=6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold", pad=8)
        ax.set_ylabel(unit, fontsize=10)
        ax.set_ylim(0, max(vals) * (1.55 if vals is cal else 1.28))
        _clean(ax)
    axes[0].text(0.02, 0.97, "whiskers: 95% interval,\nbootstrap over plate sessions",
                 transform=axes[0].transAxes, fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout(pad=0.6, w_pad=2.4)
    _save(fig, "backbones.png")


# ---------------------------------------------------------------- slide 9, gates
GATE_LABELS = {"perfect": "a perfect refuser", "blend": "blend of the four",
               "learned_error": "learned error head", "pred_magnitude": "predicted size  (control)",
               "ensemble_spread": "ensemble disagreement", "knn_dist": "distance to neighbours",
               "mahalanobis": "distance to training mean", "random": "refusing at random"}


def gates(model=PRIMARY) -> None:
    """Every gate at 50% coverage, between the random floor and the perfect ceiling."""
    blk = json.loads((RESULTS / "gate_probe.json").read_text())[model]["clean"]
    order = sorted((g for g in blk["gates"] if g in GATE_LABELS),
                   key=lambda g: -blk["gates"][g]["mae"]["0.5"])
    vals = [blk["gates"][g]["mae"]["0.5"] for g in order]
    colour = {"perfect": GREEN, "ensemble_spread": ORANGE, "pred_magnitude": BLUE}

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=200)
    bars = ax.barh(range(len(order)), vals, height=0.66, zorder=3,
                   color=[colour.get(g, "#B9BEC7") for g in order])
    for b, g, v in zip(bars, order, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=10.5, fontweight="bold",
                color=colour.get(g, MUTED))
    ax.axvline(blk["answer_all"], color=INK, linestyle=(0, (4, 3)), linewidth=1.5, zorder=4)
    ax.text(blk["answer_all"] - 1.8, len(order) - 0.30,
            f"answer everything\n{blk['answer_all']:.1f} kcal", fontsize=9.5,
            color=INK, fontweight="bold", va="top", ha="right", linespacing=1.4)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([GATE_LABELS.get(g, g) for g in order], fontsize=10.5)
    ax.set_xlabel("calorie error once the worst half is refused (kcal)", fontsize=10.5)
    ax.set_xlim(0, max(vals) * 1.20)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    _save(fig, "gates.png")


def degeneracy(model=PRIMARY) -> None:
    """The trap: a gate can post a good error by only answering for small meals.

    Two operating points side by side. At 90% coverage every learned gate keeps meals
    the size a perfect refuser keeps; at 50% the control and the error head drift to
    small meals while ensemble disagreement stays with the oracle.
    """
    blk = json.loads((RESULTS / "gate_probe.json").read_text())[model]["clean"]
    g = blk["gates"]
    colour = {"perfect": GREEN, "ensemble_spread": OLIVE, "learned_error": ORANGE,
              "pred_magnitude": BLUE}
    labels = {"perfect": "perfect refuser", "pred_magnitude": "predicted size (control)",
              "learned_error": "learned error head", "ensemble_spread": "ensemble disagreement",
              "knn_dist": "neighbour distance", "mahalanobis": "distance to mean",
              "random": "random"}
    # the two distance gates sit on top of each other at 90%, so they share one label
    offsets = {
        "0.9": {"perfect": (9, -4, "left"), "pred_magnitude": (0, 10, "center"),
                "learned_error": (9, -4, "left"), "ensemble_spread": (0, -16, "center"),
                "knn_dist": (-9, 6, "right"), "mahalanobis": (None, 0, "right"),
                "random": (9, -3, "left")},
        "0.5": {"perfect": (9, -4, "left"), "pred_magnitude": (0, 10, "center"),
                "learned_error": (9, -4, "left"), "ensemble_spread": (9, -4, "left"),
                "knn_dist": (0, 10, "center"), "mahalanobis": (0, -15, "center"),
                "random": (9, -3, "left")}}
    labels_at = {"0.9": dict(labels, knn_dist="distance to mean · neighbour distance"),
                 "0.5": labels}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200, sharey=False)
    for ax, c, title in [(axes[0], "0.9", "refuse one photo in ten"),
                         (axes[1], "0.5", "refuse half")]:
        ox = g["perfect"]["selectivity"][c]["kept_mean_kcal"]
        ax.axvline(ox, color=GREEN, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
        for name, label in labels_at[c].items():
            r = g[name]
            x, y = r["selectivity"][c]["kept_mean_kcal"], r["mae"][c]
            col = colour.get(name, MUTED)
            big = name in colour
            ax.scatter(x, y, s=130 if big else 60, color=col, zorder=4, linewidths=0)
            dx, dy, ha = offsets[c][name]
            if dx is None:
                continue
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=9.2, fontweight="bold" if big else "normal",
                        color=col if big else INK, zorder=5)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold", pad=8)
        ax.set_xlabel("mean size of the meals it still answers (kcal)", fontsize=10)
        _clean(ax)
    axes[0].set_ylabel("error on those meals (kcal)", fontsize=10)
    axes[0].set_xlim(190, 275)
    axes[0].set_ylim(40, 80)
    axes[1].set_xlim(60, 295)
    axes[1].set_ylim(10, 85)
    axes[0].text(g["perfect"]["selectivity"]["0.9"]["kept_mean_kcal"] + 1.5, 79,
                 "a perfect refuser\nkeeps meals this size", fontsize=9, color=GREEN,
                 fontweight="bold", ha="left", va="top", linespacing=1.35)
    fig.tight_layout(pad=0.5, w_pad=2.0)
    _save(fig, "degeneracy.png")


# ---------------------------------------------------------------- 17 Sep additions
SHORT = {"vit_base_patch16_clip_224.openai": "CLIP",
         "vit_base_patch16_siglip_224.v2_webli": "SigLIP 2",
         "vit_base_patch16_dinov3.lvd1689m": "DINOv3"}


def curves_ci(model=PRIMARY) -> None:
    """Risk–coverage curves on clean photos with cluster-bootstrap bands.

    Bands are drawn for the two curves the test compares, the learned error head and
    refusing at random; the others are lines so the plot stays readable.
    """
    res = json.loads((RESULTS / "gate_probe.json").read_text())[model]
    cov = [100 * c for c in res["coverages"]]
    g = res["clean"]["gates"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
    series = [("perfect", GREEN, "-", 2.2, "perfect refuser", False),
              ("learned_error", ORANGE, "-", 2.4, "learned error head", True),
              ("ensemble_spread", OLIVE, "-", 2.0, "ensemble disagreement", False),
              ("lap_var", BLUE, "-", 1.8, "sharpness", False),
              ("random", MUTED, (0, (4, 3)), 1.8, "random", True)]
    for key, colour, style, width, label, band in series:
        y = [g[key]["mae"][str(c)] for c in res["coverages"]]
        if band:
            lo = [g[key]["mae_ci"][str(c)][0] for c in res["coverages"]]
            hi = [g[key]["mae_ci"][str(c)][1] for c in res["coverages"]]
            ax.fill_between(cov, lo, hi, color=colour, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(cov, y, color=colour, linestyle=style, linewidth=width,
                solid_capstyle="round", zorder=3)
    anchors = {"perfect": (28, 0, -15), "learned_error": (38, 0, -16),
               "ensemble_spread": (58, 0, 12), "lap_var": (40, 0, 9), "random": (18, 0, 9)}
    for key, colour, _, _, label, _ in series:
        y = [g[key]["mae"][str(c)] for c in res["coverages"]]
        ax_x, dx, dy = anchors[key]
        i = min(range(len(cov)), key=lambda j: abs(cov[j] - ax_x))
        ax.annotate(label, (cov[i], y[i]), textcoords="offset points", xytext=(dx, dy),
                    color=colour, fontsize=10, fontweight="bold", ha="center", zorder=5)
    ax.axvline(90, color=INK, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    ax.text(89, 8, "the operating point\nwe report: refuse 1 in 10", fontsize=9.2,
            color=INK, ha="right", va="bottom", linespacing=1.35)
    ax.set_xlabel("share of photos the app still answers", fontsize=10.5)
    ax.set_ylabel("calorie error on those photos (kcal)", fontsize=10.5)
    ax.set_xlim(8, 102)
    ax.set_ylim(0, 100)
    ax.set_xticks([10, 25, 50, 75, 90, 100])
    ax.set_xticklabels([f"{t}%" for t in [10, 25, 50, 75, 90, 100]], fontsize=10)
    _clean(ax)
    fig.tight_layout(pad=0.4)
    _save(fig, "curves_ci.png")


FOREST_ORDER = ["learned_error", "blend", "pred_magnitude", "ensemble_spread", "knn_dist",
                "mahalanobis", "hf_ratio", "blockiness", "tenengrad", "lap_var"]
FOREST_LABELS = {"learned_error": "learned error head", "blend": "blend (four model-aware)",
                 "pred_magnitude": "predicted size (control)",
                 "ensemble_spread": "ensemble disagreement", "knn_dist": "neighbour distance",
                 "mahalanobis": "distance to mean", "hf_ratio": "high-freq. ratio",
                 "blockiness": "blockiness", "tenengrad": "Tenengrad", "lap_var": "sharpness"}


def significance(coverage="0.9") -> None:
    """Paired difference to the random gate, one panel per backbone, with 95% intervals.

    Filled marker: Holm-corrected p < 0.05. Hollow: not distinguishable from random.
    Colour is the gate family; identity is carried by the row label.
    """
    res = json.loads((RESULTS / "gate_probe.json").read_text())
    keys = [k for k in SHORT if k in res]
    family = {g: (ORANGE if g in ("learned_error", "blend", "ensemble_spread") else
                  BLUE if g == "pred_magnitude" else
                  OLIVE if g in ("knn_dist", "mahalanobis") else MUTED)
              for g in FOREST_ORDER}

    fig, axes = plt.subplots(1, len(keys), figsize=(9.8, 3.9), dpi=200, sharey=True)
    for ax, key in zip(axes, keys):
        g = res[key]["clean"]["gates"]
        ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
        for i, name in enumerate(FOREST_ORDER):
            if name not in g:
                continue
            v = g[name]["vs_random"][coverage]
            y = len(FOREST_ORDER) - 1 - i
            lo, hi = v["ci"]
            col = family[name]
            sig = v.get("p_holm", 1.0) < 0.05
            ax.plot([lo, hi], [y, y], color=col, linewidth=2.0, solid_capstyle="round",
                    zorder=3)
            ax.scatter([v["diff"]], [y], s=64, color=col if sig else "white",
                       edgecolors=col, linewidths=1.8, zorder=4)
        ax.set_title(SHORT[key], fontsize=11, color=INK, fontweight="bold", pad=8)
        ax.set_xlim(-22, 12)
        ax.set_xticks([-20, -10, 0, 10])
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(FOREST_ORDER)))
    axes[0].set_yticklabels([FOREST_LABELS[n] for n in FOREST_ORDER[::-1]], fontsize=10)
    fig.text(0.5, 0.075, "change in error against refusing at random, refusing 1 photo in "
             "10  (kcal; left is better)", ha="center", fontsize=10, color=MUTED)
    fig.text(0.5, 0.012, "filled: differs from random after Holm correction, p < 0.05 "
             "(left of zero: better; right: worse)     hollow: not distinguishable     "
             "bars: 95% interval, bootstrap over plate sessions", ha="center",
             fontsize=8.8, color=INK)
    fig.tight_layout(pad=0.5, w_pad=1.2, rect=(0, 0.09, 1, 1))
    _save(fig, "significance.png")


def significance_sized(coverage="0.9") -> None:
    """The forest plot twice: plain, and with every gate refusing inside predicted-size
    bands. Same kcal axis in both rows, so 'how much of the win was size' is read as
    how far each marker moves toward zero between the rows.
    """
    res = json.loads((RESULTS / "gate_probe.json").read_text())
    keys = [k for k in SHORT if k in res]
    family = {g: (ORANGE if g in ("learned_error", "blend", "ensemble_spread") else
                  BLUE if g == "pred_magnitude" else
                  OLIVE if g in ("knn_dist", "mahalanobis") else MUTED)
              for g in FOREST_ORDER}
    rows = [("clean", "refuse 1 in 10 overall"),
            ("clean_sized", "refuse 1 in 10 inside each\npredicted-size band")]

    fig, axes = plt.subplots(2, len(keys), figsize=(10.4, 6.3), dpi=200,
                             sharey=True, sharex=True)
    for r, (block, row_label) in enumerate(rows):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            g = res[key][block]["gates"]
            ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
            for i, name in enumerate(FOREST_ORDER):
                if name not in g:
                    continue
                v = g[name]["vs_random"][coverage]
                y = len(FOREST_ORDER) - 1 - i
                lo, hi = v["ci"]
                col = family[name]
                sig = v.get("p_holm", 1.0) < 0.05
                ax.plot([lo, hi], [y, y], color=col, linewidth=2.0,
                        solid_capstyle="round", zorder=3)
                ax.scatter([v["diff"]], [y], s=60, color=col if sig else "white",
                           edgecolors=col, linewidths=1.8, zorder=4)
            perfect = (g["perfect"]["mae"][coverage] - g["random"]["mae"][coverage])
            ax.axvline(perfect, color=GREEN, linewidth=1.2, linestyle=(0, (3, 2)), zorder=2)
            if r == 0:
                ax.set_title(SHORT[key], fontsize=11, color=INK, fontweight="bold", pad=8)
            if c == 0:
                ax.text(perfect + 0.7, -1.0, "perfect refuser", color=GREEN, fontsize=8.8,
                        ha="left", va="center", fontweight="bold")
            ax.set_ylim(-1.6, len(FOREST_ORDER) - 0.4)
            ax.set_xlim(-22, 12)
            ax.set_xticks([-20, -10, 0, 10])
            ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=9.5, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
        axes[r, 0].set_yticks(range(len(FOREST_ORDER)))
        axes[r, 0].set_yticklabels([FOREST_LABELS[n] for n in FOREST_ORDER[::-1]],
                                   fontsize=9.8)
        axes[r, 0].set_ylabel(row_label, fontsize=10, color=INK, fontweight="bold",
                              labelpad=14, linespacing=1.4)
    fig.text(0.56, 0.055, "change in calorie error against refusing at random, 90% "
             "answered  (kcal; left is better)", ha="center", fontsize=10, color=MUTED)
    fig.text(0.56, 0.012, "filled: differs from random after Holm correction, p < 0.05 "
             "(left of zero: better; right: worse)    hollow: not distinguishable    "
             "bars: 95% interval, bootstrap over plate sessions", ha="center",
             fontsize=8.6, color=INK)
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.2, rect=(0, 0.07, 1, 1))
    _save(fig, "significance_sized.png")


def conformal_chart(model=PRIMARY) -> None:
    """Does the range cover hard dishes as well as easy ones? Plain vs adaptive width."""
    res = json.loads((RESULTS / "conformal.json").read_text())[model]["clean"]
    target = 100 * (1 - json.loads((RESULTS / "conformal.json").read_text())[model]["alpha"])
    plain, adapt = res["constant"], res["ensemble_spread"]
    qs = ["easiest\nquarter", "2nd", "3rd", "hardest\nquarter"]
    x = np.arange(4)
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200)
    ax = axes[0]
    b1 = ax.bar(x - w / 2, [100 * c for c in plain["by_difficulty"]["coverage"]], w,
                color=MUTED, zorder=3, label="same width for every photo")
    b2 = ax.bar(x + w / 2, [100 * c for c in adapt["by_difficulty"]["coverage"]], w,
                color=ORANGE, zorder=3, label="width set by ensemble disagreement")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                    f"{b.get_height():.0f}", ha="center", fontsize=9.5, color=INK,
                    fontweight="bold", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.axhline(target, color=GREEN, linestyle=(0, (4, 3)), linewidth=1.5, zorder=4)
    ax.text(3.55, target + 0.8, f"target {target:.0f}%", fontsize=9.5, color=GREEN,
            fontweight="bold", ha="right", va="bottom", zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(qs, fontsize=9.5)
    ax.set_ylim(60, 104)
    ax.set_ylabel("share of dishes inside the range (%)", fontsize=10)
    ax.set_title("coverage, by how hard the dish looks", fontsize=11, color=INK,
                 fontweight="bold", pad=8)
    _clean(ax)

    ax = axes[1]
    ax.bar(x - w / 2, plain["by_difficulty"]["width"], w, color=MUTED, zorder=3)
    ax.bar(x + w / 2, adapt["by_difficulty"]["width"], w, color=ORANGE, zorder=3)
    for i in range(4):
        ax.text(x[i] - w / 2, plain["by_difficulty"]["width"][i] + 6,
                f"{plain['by_difficulty']['width'][i]:.0f}", ha="center", fontsize=9.5,
                color=INK, fontweight="bold")
        ax.text(x[i] + w / 2, adapt["by_difficulty"]["width"][i] + 6,
                f"{adapt['by_difficulty']['width'][i]:.0f}", ha="center", fontsize=9.5,
                color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(qs, fontsize=9.5)
    ax.set_ylabel("mean range width (kcal)", fontsize=10)
    ax.set_title("width of the range", fontsize=11, color=INK, fontweight="bold", pad=8)
    ax.set_ylim(0, max(adapt["by_difficulty"]["width"]) * 1.25)
    _clean(ax)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.5, w_pad=2.0, rect=(0, 0.07, 1, 1))
    _save(fig, "conformal.png")


# ---------------------------------------------------------------- update deck, slide 7
def reader_chart(model=PRIMARY) -> None:
    """The risk–coverage axes with only the reference and the ceiling drawn.

    Shown before any gate result so the audience learns the axes once: refusing at
    random is flat, a perfect refuser falls, a gate is judged by where it sits between
    them at the operating point, and by the area it leaves.
    """
    res = json.loads((RESULTS / "gate_probe.json").read_text())[model]
    covs = res["coverages"]
    cov = [100 * c for c in covs]
    g = res["clean"]["gates"]
    rnd = [g["random"]["mae"][str(c)] for c in covs]
    prf = [g["perfect"]["mae"][str(c)] for c in covs]

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200)
    ax.fill_between(cov, prf, rnd, color=ORANGE, alpha=0.08, linewidth=0, zorder=1)
    ax.plot(cov, rnd, color=MUTED, linestyle=(0, (4, 3)), linewidth=2.0, zorder=3)
    ax.plot(cov, prf, color=GREEN, linewidth=2.4, solid_capstyle="round", zorder=3)
    ax.text(12, max(rnd) + 4.5, "refusing at random: the reference", color=MUTED,
            fontsize=10.5, fontweight="bold", ha="left", va="bottom", zorder=5)
    ax.text(47, prf[covs.index(0.45)] - 5, "a perfect refuser: the ceiling", color=GREEN,
            fontsize=10.5, fontweight="bold", ha="left", va="top", zorder=5)
    ax.text(34, 0.5 * (rnd[covs.index(0.35)] + prf[covs.index(0.35)]),
            "the room a gate\nhas to win", color=ORANGE, fontsize=10.5,
            fontweight="bold", ha="center", va="center", linespacing=1.4, zorder=5)

    i90 = covs.index(0.9)
    ax.axvline(90, color=INK, linewidth=1.0, linestyle=(0, (2, 3)), zorder=2)
    ax.annotate("", xy=(90, prf[i90]), xytext=(90, rnd[i90]),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3,
                                shrinkA=2, shrinkB=2), zorder=5)
    ax.plot([90, 90], [rnd[i90], prf[i90]], "o", color=INK, markersize=5, zorder=6)
    ax.text(88, 97, f"at 90% answered: random {rnd[i90]:.0f}, perfect {prf[i90]:.0f}.\n"
            f"A gate has {rnd[i90] - prf[i90]:.0f} kcal to win here.",
            fontsize=9.5, color=INK, ha="right", va="top", linespacing=1.4, zorder=6)
    ax.text(90.6, 3, "the operating\npoint we report", fontsize=9.2, color=INK,
            ha="left", va="bottom", linespacing=1.35, zorder=6)
    ax.set_xlabel("share of photos the app still answers", fontsize=10.5)
    ax.set_ylabel("calorie error on those photos (kcal)", fontsize=10.5)
    ax.set_xlim(8, 102)
    ax.set_ylim(0, 100)
    ax.set_xticks([10, 25, 50, 75, 90, 100])
    ax.set_xticklabels([f"{t}%" for t in [10, 25, 50, 75, 90, 100]], fontsize=10)
    _clean(ax)
    fig.tight_layout(pad=0.4)
    _save(fig, "reader.png")


# ---------------------------------------------------------------- update deck, slide 13
def conformal_kcal_chart(model=PRIMARY) -> None:
    """Coverage by true meal size (the honest cut) beside coverage by difficulty.

    The difficulty quartiles are cut on the same signal that sets the adaptive width,
    so the adaptive range is flat there partly by construction. Meal size is not, and
    both ranges fail the same way on the largest quarter.
    """
    blob = json.loads((RESULTS / "conformal.json").read_text())[model]
    res, target = blob["clean"], 100 * (1 - blob["alpha"])
    plain, adapt = res["constant"], res["ensemble_spread"]
    x, w = np.arange(4), 0.36

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200, sharey=True)
    panels = [("by_kcal", ["smallest\nquarter", "2nd", "3rd", "largest\nquarter"],
               "coverage, by true meal size"),
              ("by_difficulty", ["easiest\nquarter", "2nd", "3rd", "hardest\nquarter"],
               "coverage, by difficulty (ensemble-disagreement quartiles)")]
    for ax, (key, qs, title) in zip(axes, panels):
        b1 = ax.bar(x - w / 2, [100 * c for c in plain[key]["coverage"]], w, color=MUTED,
                    zorder=3, label="same width for every photo")
        b2 = ax.bar(x + w / 2, [100 * c for c in adapt[key]["coverage"]], w, color=ORANGE,
                    zorder=3, label="width set by ensemble disagreement")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                        f"{b.get_height():.0f}", ha="center", fontsize=9.5, color=INK,
                        fontweight="bold", zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ax.axhline(target, color=GREEN, linestyle=(0, (4, 3)), linewidth=1.5, zorder=4)
        ax.text(3.5, target + 0.8, f"target {target:.0f}%", fontsize=9.5, color=GREEN,
                fontweight="bold", ha="right", va="bottom", zorder=6)
        ax.set_xticks(x)
        ax.set_xticklabels(qs, fontsize=9.5)
        ax.set_ylim(55, 104)
        ax.set_title(title, fontsize=10.5, color=INK, fontweight="bold", pad=8)
        _clean(ax)
    axes[0].set_ylabel("share of dishes inside the range (%)", fontsize=10)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.5, w_pad=1.6, rect=(0, 0.07, 1, 1))
    _save(fig, "conformal_kcal.png")


REP_GATES = [("learned_error", "learned error head"),
             ("ensemble_spread", "ensemble disagreement"),
             ("pred_magnitude", "predicted-size control")]


def replication_chart() -> None:
    """The within-size-band result three ways (final deck, response to feedback).

    Grey: what the update talk reported (one training carve, scored on the 127 test
    sessions). Blue: the same test sessions, averaged over 20 calibration carves.
    Orange: 367 train sessions no gate had been scored on, cross-fitted by session.
    """
    gp = json.loads((RESULTS / "gate_probe.json").read_text())
    rep = json.loads((RESULTS / "replication.json").read_text())
    keys = [k for k in SHORT if k in rep]
    src = [("update talk: one training carve,\n127 test sessions", MUTED),
           ("averaged over 20 training carves,\n127 test sessions", BLUE),
           ("new data: 367 train sessions,\nnever scored before", ORANGE)]
    fig, axes = plt.subplots(1, len(keys), figsize=(10.4, 3.6), dpi=200, sharey=True)
    for c, key in enumerate(keys):
        ax = axes[c]
        old = gp[key]["clean_sized"]["gates"]
        car = rep[key]["carves"]["averaged"]
        new = rep[key]["clean_sized"]["gates"]
        ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
        for i, (g, _) in enumerate(REP_GATES):
            base = (len(REP_GATES) - 1 - i) * 4
            vals = [(old[g]["vs_random"]["0.9"]["diff"], old[g]["vs_random"]["0.9"]["ci"]),
                    (car[f"{g}_sized"]["mean"], car[f"{g}_sized"]["ci"]),
                    (new[g]["vs_random"]["0.9"]["diff"], new[g]["vs_random"]["0.9"]["ci"])]
            for j, ((d, (lo, hi)), (_, col)) in enumerate(zip(vals, src)):
                y = base + 2 - j
                ax.plot([lo, hi], [y, y], color=col, linewidth=2.0,
                        solid_capstyle="round", zorder=3)
                ax.scatter([d], [y], s=46, color=col, edgecolors="white", linewidths=1.2,
                           zorder=4)
        ax.set_title(SHORT[key], fontsize=11, color=INK, fontweight="bold", pad=8)
        ax.set_xlim(-10, 3)
        ax.set_xticks([-8, -6, -4, -2, 0, 2])
        ax.set_ylim(-1, len(REP_GATES) * 4 - 1)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks([(len(REP_GATES) - 1 - i) * 4 + 1 for i in range(len(REP_GATES))])
    axes[0].set_yticklabels([lab for _, lab in REP_GATES], fontsize=10)
    handles = [plt.Line2D([], [], color=col, marker="o", linewidth=2.0, markersize=6)
               for _, col in src]
    fig.legend(handles, [lab for lab, _ in src], loc="lower center", ncol=3, fontsize=9.2,
               frameon=False, bbox_to_anchor=(0.55, 0.0))
    fig.text(0.55, 0.165, "change in calorie error against refusing at random, inside "
             "predicted-size bands, 90% answered (kcal; left is better; bars: 95% interval)",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(pad=0.5, w_pad=1.2, rect=(0, 0.19, 1, 1))
    _save(fig, "replication.png")


def carves_chart() -> None:
    """Twenty training carves, one dot each; the one the update talk reported is orange."""
    rep = json.loads((RESULTS / "replication.json").read_text())
    keys = [k for k in SHORT if k in rep]
    fig, ax = plt.subplots(figsize=(6.2, 2.9), dpi=200)
    rng = np.random.default_rng(0)
    for i, key in enumerate(keys):
        v = np.array(rep[key]["carves"]["per_carve"]["learned_error_sized"])
        y = len(keys) - 1 - i
        jit = rng.uniform(-0.16, 0.16, len(v))
        ax.scatter(v[1:], y + jit[1:], s=34, color=MUTED, alpha=0.75, edgecolors="white",
                   linewidths=0.8, zorder=3)
        ax.scatter([v[0]], [y], s=70, color=ORANGE, edgecolors="white", linewidths=1.2,
                   zorder=5)
        m = rep[key]["carves"]["averaged"]["learned_error_sized"]["mean"]
        ax.plot([m, m], [y - 0.3, y + 0.3], color=BLUE, linewidth=2.2, zorder=4)
        if i == 0:
            ax.annotate("the training split we\nreported in September", (v[0], y), xytext=(v[0] - 0.2, y + 0.62),
                        fontsize=9, color=ORANGE, fontweight="bold", ha="left", va="center")
            ax.text(m + 0.15, y - 0.5, "average of 20", fontsize=9, color=BLUE,
                    fontweight="bold", ha="left", va="center")
    ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([SHORT[k] for k in keys[::-1]], fontsize=10)
    ax.set_ylim(-0.7, len(keys) - 0.1)
    ax.set_xlim(-8, 1)
    ax.set_xlabel("learned error head vs random, inside size bands, 90% answered (kcal)",
                  fontsize=9.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    _save(fig, "carves.png")


def ranges_size_chart() -> None:
    """Coverage by true meal size: the same range for everyone vs CQR, three backbones."""
    blob = json.loads((RESULTS / "conformal.json").read_text())
    keys = [k for k in SHORT if k in blob]
    x, w = np.arange(4), 0.36
    fig, axes = plt.subplots(1, len(keys), figsize=(10.4, 3.4), dpi=200, sharey=True)
    for ax, key in zip(axes, keys):
        res, target = blob[key]["clean"], 100 * (1 - blob[key]["alpha"])
        plain, cqr = res["constant"]["by_kcal"], res["cqr"]["by_kcal"]
        b1 = ax.bar(x - w / 2 - 0.01, [100 * c for c in plain["coverage"]], w, color=MUTED,
                    zorder=3, label="same range width for every photo")
        b2 = ax.bar(x + w / 2 + 0.01, [100 * c for c in cqr["coverage"]], w, color=ORANGE,
                    zorder=3, label="size-aware range (conformalised quantile regression)")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                        f"{b.get_height():.0f}", ha="center", fontsize=9, color=INK,
                        fontweight="bold", zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        ax.axhline(target, color=GREEN, linestyle=(0, (4, 3)), linewidth=1.5, zorder=4,
                   label=f"target coverage {target:.0f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(["smallest\nquarter", "2nd", "3rd", "largest\nquarter"],
                           fontsize=9.2)
        ax.set_ylim(55, 106)
        ax.set_title(SHORT[key], fontsize=11, color=INK, fontweight="bold", pad=8)
        _clean(ax)
    axes[0].set_ylabel("true calories inside the range (%)", fontsize=9.8)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.115, "test dishes grouped by true meal size", ha="center", fontsize=9.5,
             color=MUTED)
    fig.tight_layout(pad=0.5, w_pad=1.4, rect=(0, 0.15, 1, 1))
    _save(fig, "ranges_size.png")


# Only the gates the final talk speaks to on this slide; the stabilised heads and
# cross-backbone disagreement are on their own slide (stabilised.png), the rest in the JSON.
POOLED_ROWS = [("learned_error", "learned error head", ORANGE),
               ("ensemble_spread", "ensemble disagreement", ORANGE),
               ("pred_magnitude", "predicted-size control", BLUE),
               ("lap_var", "sharpness (Laplacian variance)", MUTED)]


def pooled_chart(bands="10") -> None:
    """The best estimate: every dish out of fold, five refits averaged, inside size bands."""
    rob = json.loads((RESULTS / "robustness.json").read_text())
    keys = [k for k in SHORT if k in rob and "pooled" in rob[k]]
    fig, axes = plt.subplots(1, len(keys), figsize=(10.4, 3.2), dpi=200, sharey=True)
    for ax, key in zip(axes, keys):
        blk = rob[key]["pooled"][bands]
        ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
        for i, (g, _, col) in enumerate(POOLED_ROWS):
            v = blk["gates"][g]["coverage"]["0.9"]
            y = len(POOLED_ROWS) - 1 - i
            sig = v.get("p_holm", 1.0) < 0.05
            ax.plot(v["ci"], [y, y], color=col, linewidth=2.0, solid_capstyle="round", zorder=3)
            ax.scatter([v["diff"]], [y], s=52, color=col if sig else "white", edgecolors=col,
                       linewidths=1.8, zorder=4)
        ax.set_title(SHORT[key], fontsize=11, color=INK, fontweight="bold", pad=8)
        ax.set_xlim(-4.5, 2.5)
        ax.set_xticks([-4, -3, -2, -1, 0, 1, 2])
        ax.set_ylim(-0.7, len(POOLED_ROWS) - 0.3)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        perfect = blk["gates"]["perfect"]["coverage"]["0.9"]["diff"]
        ax.text(-4.4, -0.55, f"perfect refuser: {perfect:.1f}", fontsize=8.8, color=GREEN,
                fontweight="bold", ha="left", va="center")
    axes[0].set_yticks(range(len(POOLED_ROWS)))
    axes[0].set_yticklabels([lab for _, lab, _ in POOLED_ROWS[::-1]], fontsize=9.6)
    fig.text(0.6, 0.09, "change in calorie error against refusing at random, inside predicted-"
             "size bands, 90% answered (kcal; left is better)", ha="center", fontsize=9.6,
             color=MUTED)
    fig.text(0.6, 0.025, "every dish scored by models that never saw its session; five refits "
             "averaged; filled: differs from random after Holm correction; bars: 95% interval",
             ha="center", fontsize=8.6, color=INK)
    fig.tight_layout(pad=0.5, w_pad=1.0, rect=(0, 0.13, 1, 1))
    _save(fig, "pooled.png")


def size_split_chart() -> None:
    """How much of each gate's win is meal size: overall vs inside size bands (pooled)."""
    rob = json.loads((RESULTS / "robustness.json").read_text())
    keys = [k for k in SHORT if k in rob and "pooled" in rob[k]]
    rows = [("pred_magnitude", "predicted size only"), ("learned_error", "learned error head"),
            ("ensemble_spread", "ensemble disagreement"), ("perfect", "perfect refuser")]
    modes = [("none", "refuse 1 in 10 overall", MUTED),
             ("10", "refuse 1 in 10 inside each size band", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=200)
    bh = 0.36
    for i, (g, _) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (mode, lab, col) in enumerate(modes):
            v = [-rob[k]["pooled"][mode]["gates"][g]["coverage"]["0.9"]["diff"] for k in keys]
            yy = y + (0.2 if j == 0 else -0.2)
            ax.barh(yy, np.mean(v), height=bh, color=col, zorder=3,
                    label=lab if i == 0 else None)
            ax.scatter(v, [yy] * len(v), s=10, color=INK, zorder=4, linewidths=0)
            ax.text(max(v) + 0.6, yy, f"{np.mean(v):.1f}", va="center", fontsize=9,
                    color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows[::-1]], fontsize=10)
    ax.set_xlim(0, 22.5)
    ax.set_xlabel("kcal of error removed vs refusing at random, 90% answered", fontsize=9.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=8.8, frameon=False)
    fig.tight_layout(pad=0.4)
    _save(fig, "size_split.png")


def stabilised_chart() -> None:
    """Whole-curve gain inside size bands: the plain error head and the two stabilised ones."""
    rob = json.loads((RESULTS / "robustness.json").read_text())
    keys = [k for k in SHORT if k in rob and "pooled" in rob[k]]
    gates = [("learned_error", "error head", MUTED),
             ("learned_error_bag", "average of 10 heads", ORANGE),
             ("learned_error_oof", "trained on 3x the errors", BLUE)]
    fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=200)
    x, w = np.arange(len(keys)), 0.25
    for j, (g, lab, col) in enumerate(gates):
        vals = [-rob[k]["pooled"]["10"]["gates"][g]["aurc_diff"]["diff"] for k in keys]
        cis = [rob[k]["pooled"]["10"]["gates"][g]["aurc_diff"]["ci"] for k in keys]
        xs = x + (j - 1) * (w + 0.02)
        ax.bar(xs, vals, w, color=col, zorder=3, label=lab)
        for xx, v, (lo, hi) in zip(xs, vals, cis):
            ax.plot([xx, xx], [-hi, -lo], color=INK, linewidth=1.1, zorder=4)
            ax.text(xx, -lo + 0.35, f"{v:.1f}", ha="center", fontsize=8.8, color=INK,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=10.5)
    ax.set_ylim(0, 13.5)
    ax.set_ylabel("kcal removed vs random,\nwhole curve, inside size bands", fontsize=9.3)
    _clean(ax)
    ax.legend(loc="upper left", fontsize=8.8, frameon=False, ncol=3)
    fig.tight_layout(pad=0.4)
    _save(fig, "stabilised.png")


def sessions_chart() -> None:
    """Why the unit of analysis is the plate session: the gap between scans is bimodal."""
    ids = n5k.usable_split("test", local_only=True) + n5k.usable_split("train", local_only=True)
    ts = np.array(sorted(n5k.timestamp(d) for d in ids))
    gaps = np.diff(ts)
    gaps = gaps[gaps > 0]
    fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=200)
    bins = np.logspace(0, 6.5, 40)
    ax.hist(gaps, bins=bins, color=MUTED, zorder=3, linewidth=0)
    ax.axvline(n5k.SESSION_GAP_S, color=ORANGE, linewidth=1.8, zorder=4)
    ax.text(n5k.SESSION_GAP_S * 1.3, ax.get_ylim()[1] * 0.92, "10 minutes", color=ORANGE,
            fontsize=10, fontweight="bold", va="top")
    ax.text(3.2, ax.get_ylim()[1] * 0.92, "next scan of\nthe same session", color=INK,
            fontsize=9.5, ha="center", va="top", linespacing=1.3)
    ax.text(2.2e5, ax.get_ylim()[1] * 0.92, "next session,\nhours or days later", color=INK,
            fontsize=9.5, ha="center", va="top", linespacing=1.3)
    ax.set_xscale("log")
    ax.set_xlabel("seconds between consecutive dish scans", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.set_xticks([1, 60, 600, 3600, 86400, 1e6])
    ax.set_xticklabels(["1 s", "1 min", "10 min", "1 h", "1 day", "12 days"], fontsize=9)
    _clean(ax)
    fig.tight_layout(pad=0.4)
    _save(fig, "sessions.png")


def panel_phone() -> None:
    src = Image.open(n5k.IMAGES / DEMO_DISH / "rgb.png").convert("RGB")
    apply_degradation(src, "phone").resize((720, 540), Image.LANCZOS).save(FIGS / "panel_phone.png")
    print("  panel_phone.png")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    print("figures:")
    photo_panels()
    risk_coverage_chart()
    risk_coverage_pooled()
    dataset_scatter()
    plate_grid()
    ladder()
    backbones()
    gates()
    degeneracy()
    curves_ci()
    significance()
    significance_sized()
    conformal_chart()
    conformal_kcal_chart()
    reader_chart()
    sessions_chart()
    panel_phone()
    ranges_size_chart()
    if (RESULTS / "replication.json").exists():
        replication_chart()
        carves_chart()
    if (RESULTS / "robustness.json").exists():
        pooled_chart()
        size_split_chart()
        stabilised_chart()


if __name__ == "__main__":
    main()
