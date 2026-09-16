"""The gate benchmark — reproduce every number on slides 7, 8 and 9 of the deck.

  ../.venv/bin/python gate_bench.py

Until now the predictor result (73.4 kcal), the quality-metric correlations and the
risk-coverage curves lived only in prose. This script regenerates all of them from
the cached features in data/cache/ plus the images on disk, and writes:

  results/predictions.csv    one row per (dish, damage level): truth, prediction,
                             error, and every gate score. This is the file the whole
                             project groupbys — see the methodology slide.
  results/gate_bench.json    the summary numbers the deck quotes.

Everything is fit on the 1,929 fit dishes, never on calibration and never on test.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import n5k
from degrade import apply_degradation
from quality import METRICS

MODEL = "vit_base_patch16_clip_224"
RESULTS = Path(__file__).resolve().parent.parent / "results"

# The cached test conditions. Labels are the plain-English ones the deck uses.
# `phone` (added 17 Sep 2026) is the compound imitation of a hand-held phone photo.
RUNGS = [
    ("clean", "clean", "clean"),
    ("blur3", "blur:3", "blurred ×3"),
    ("blur6", "blur:6", "blurred ×6"),
    ("crop0p4", "crop:0.4", "cropped to 40%"),
    ("jpeg10", "jpeg:10", "heavy JPEG"),
    ("downscale8", "downscale:8", "downscaled ×8"),
    ("noise0p16", "noise:0.16", "heavy noise"),
    ("phone", "phone", "phone imitation"),
]

PIXEL_GATES = ["lap_var", "tenengrad", "blockiness", "hf_ratio"]


# ---------------------------------------------------------------- feature cache
def load_cache(split: str, tag: str) -> tuple[np.ndarray, list[str]]:
    z = np.load(n5k.CACHE / f"{MODEL}__{split}__{tag}.npz", allow_pickle=True)
    return z["features"], [str(d) for d in z["dish_ids"]]


def rows_for(feats: np.ndarray, ids: list[str], want: list[str]) -> np.ndarray:
    pos = {d: i for i, d in enumerate(ids)}
    return feats[[pos[d] for d in want]]


# ---------------------------------------------------------------- ridge, by hand
def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    """Closed-form ridge with an unpenalised intercept (features are centred first)."""
    mu, ymu = X.mean(0), y.mean()
    Xc = X - mu
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1], dtype=np.float64)
    w = np.linalg.solve(A, Xc.T @ (y - ymu))
    return mu, ymu, w


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    mu, ymu, w = model
    return (X - mu) @ w + ymu


ALPHA_GRID = tuple(10 ** (x / 4) for x in range(0, 25))  # 1 … 1e6, quarter-decade steps


def pick_alpha(Xf, yf, Xc, yc, grid=ALPHA_GRID):
    """Alpha chosen on the calibration split, which the head is never fit on."""
    best, best_mae = None, np.inf
    for a in grid:
        mae = float(np.abs(ridge_predict(ridge_fit(Xf, yf, a), Xc) - yc).mean())
        if mae < best_mae:
            best, best_mae = a, mae
    return best, best_mae


def standardiser(Xfit: np.ndarray):
    """Per-dimension z-scoring, learned on the fit split only.

    The backbone's 2,304 dimensions (CLS, patch mean, patch std) differ in scale by
    orders of magnitude, so an unstandardised ridge penalty is effectively arbitrary
    across blocks. Standardising first is worth ~7 kcal and ~3 g.
    """
    mu, sd = Xfit.mean(0), Xfit.std(0) + 1e-8
    return lambda X: (X - mu) / sd


# ---------------------------------------------------------------- gate signals
def mahalanobis_scorer(Xfit: np.ndarray, k: int = 64):
    """Distance from the fit-set mean in feature space, PCA-whitened to k dims.

    2,304 dimensions on 1,929 samples is rank-deficient, so a full covariance is not
    invertible. Projecting onto the leading k principal directions of the fit set and
    whitening there is the stable version, and it costs no extra inference — the
    features are already cached.
    """
    mu = Xfit.mean(0)
    Xc = Xfit - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V, var = Vt[:k].T, (S[:k] ** 2) / (len(Xfit) - 1)

    def score(X: np.ndarray) -> np.ndarray:
        Z = (X - mu) @ V
        return np.sqrt(((Z ** 2) / np.maximum(var, 1e-12)).sum(1))

    return score


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v):
        order = v.argsort()
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def risk_coverage(score: np.ndarray, err: np.ndarray, fracs=(1.0, 0.75, 0.5, 0.25)):
    """Keep the `frac` of photos the gate is most confident about; report their MAE.

    Higher score = the gate thinks this photo is riskier, so we keep the LOWEST scores.
    """
    order = np.asarray(score, float).argsort()
    return {f: float(err[order[: max(1, int(round(f * len(err))))]].mean()) for f in fracs}


# ---------------------------------------------------------------- quality metrics
def measure_rungs(dish_ids: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    """Per-image pixel statistics for every (dish, damage level). Reads the images."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    total = len(dish_ids) * len(RUNGS)
    done = 0
    for did in dish_ids:
        base_img = Image.open(n5k.IMAGES / did / "rgb.png").convert("RGB")
        for tag, spec, _ in RUNGS:
            img = base_img if tag == "clean" else apply_degradation(base_img, spec)
            out[(did, tag)] = {m: fn(img) for m, fn in METRICS.items() if m in PIXEL_GATES}
            done += 1
        if done % 350 < len(RUNGS):
            print(f"  quality {done}/{total}", end="\r")
    print(f"  quality {total}/{total}   ")
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    dishes = n5k.load_dishes()
    fit_ids, calib_ids = n5k.fit_calib_split(local_only=True)
    test_ids = n5k.usable_split("test", local_only=True)

    Xtr, tr_ids = load_cache("train", "clean")
    fit_ids = [d for d in fit_ids if d in set(tr_ids)]
    calib_ids = [d for d in calib_ids if d in set(tr_ids)]
    Xfit = rows_for(Xtr, tr_ids, fit_ids).astype(np.float64)
    Xcal = rows_for(Xtr, tr_ids, calib_ids).astype(np.float64)
    zscore = standardiser(Xfit)
    Xfit, Xcal = zscore(Xfit), zscore(Xcal)

    print(f"fit {len(fit_ids)} | calib {len(calib_ids)} | test {len(test_ids)}")

    heads, alphas = {}, {}
    for target in ("cal", "mass"):
        yf = np.array([getattr(dishes[d], target) for d in fit_ids], float)
        yc = np.array([getattr(dishes[d], target) for d in calib_ids], float)
        a, cal_mae = pick_alpha(Xfit, yf, Xcal, yc)
        heads[target] = ridge_fit(Xfit, yf, a)
        alphas[target] = a
        print(f"  {target:5s} alpha={a:g}  calib MAE {cal_mae:.1f}")

    maha = mahalanobis_scorer(Xfit)
    qual = measure_rungs(test_ids)

    # ---- one row per (dish, damage level): the predictions file
    recs = []
    for tag, spec, label in RUNGS:
        Xt, ids = load_cache("test", tag)
        keep = [d for d in test_ids if d in set(ids)]
        Xk = zscore(rows_for(Xt, ids, keep).astype(np.float64))
        pred_cal, pred_mass = ridge_predict(heads["cal"], Xk), ridge_predict(heads["mass"], Xk)
        mscore = maha(Xk)
        for i, did in enumerate(keep):
            d = dishes[did]
            recs.append({
                "dish_id": did, "rung": tag, "damage": label,
                "cal_true": d.cal, "cal_pred": float(pred_cal[i]),
                "cal_abs_err": abs(float(pred_cal[i]) - d.cal),
                "mass_true": d.mass, "mass_pred": float(pred_mass[i]),
                "mass_abs_err": abs(float(pred_mass[i]) - d.mass),
                "mahalanobis": float(mscore[i]),
                **qual[(did, tag)],
            })

    with open(RESULTS / "predictions.csv", "w", encoding="utf-8") as fh:
        cols = list(recs[0])
        fh.write(",".join(cols) + "\n")
        for r in recs:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    print(f"wrote {RESULTS / 'predictions.csv'} ({len(recs)} rows)")

    # ---- per-damage-level table (slide 8 left)
    by_rung = defaultdict(list)
    for r in recs:
        by_rung[r["rung"]].append(r)
    per_rung = {}
    for tag, _, label in RUNGS:
        rs = by_rung[tag]
        per_rung[tag] = {
            "label": label,
            "cal_mae": st.mean(r["cal_abs_err"] for r in rs),
            "mass_mae": st.mean(r["mass_abs_err"] for r in rs),
            "lap_var": st.mean(r["lap_var"] for r in rs),
            "n": len(rs),
        }

    # ---- correlations and risk-coverage over the pooled ladder (slide 8 right)
    err = np.array([r["cal_abs_err"] for r in recs])
    rho = {g: spearman(np.array([r[g] for r in recs]), err) for g in PIXEL_GATES}
    rho["mahalanobis"] = spearman(np.array([r["mahalanobis"] for r in recs]), err)

    rng = np.random.default_rng(760)
    # lap_var is negated: high sharpness is meant to mean LOW risk.
    scores = {
        "lap_var": -np.array([r["lap_var"] for r in recs]),
        "mahalanobis": np.array([r["mahalanobis"] for r in recs]),
        "random": rng.random(len(recs)),
        "oracle": err,
    }
    curves = {k: risk_coverage(v, err) for k, v in scores.items()}
    dense = [round(0.10 + 0.02 * i, 2) for i in range(46)]           # 10% … 100%
    curves_dense = {k: risk_coverage(v, err, dense) for k, v in scores.items()}

    summary = {
        "n_fit": len(fit_ids), "n_calib": len(calib_ids), "n_test": len(test_ids),
        "n_rows": len(recs), "alphas": alphas,
        "clean_cal_mae": per_rung["clean"]["cal_mae"],
        "clean_mass_mae": per_rung["clean"]["mass_mae"],
        "clean_mass_rel": per_rung["clean"]["mass_mae"] / st.mean(dishes[d].mass for d in test_ids),
        "per_rung": per_rung, "spearman": rho, "risk_coverage": curves,
        "risk_coverage_dense": curves_dense,
    }
    with open(RESULTS / "gate_bench.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- report
    print(f"\nclean: {per_rung['clean']['cal_mae']:.1f} kcal MAE | "
          f"{per_rung['clean']['mass_mae']:.1f} g mass MAE "
          f"({100 * summary['clean_mass_rel']:.1f}% of mean mass)")
    print(f"\n{'damage':<16}{'kcal MAE':>10}{'×':>7}{'mass MAE':>10}{'×':>7}{'lap_var':>11}")
    base_c, base_m = per_rung["clean"]["cal_mae"], per_rung["clean"]["mass_mae"]
    for tag, _, label in RUNGS:
        p = per_rung[tag]
        print(f"{label:<16}{p['cal_mae']:>10.1f}{p['cal_mae'] / base_c:>7.2f}"
              f"{p['mass_mae']:>10.1f}{p['mass_mae'] / base_m:>7.2f}{p['lap_var']:>11.1f}")

    print("\nrank correlation with |calorie error| over "
          f"{len(recs)} dish × damage-level pairs:")
    for g, v in rho.items():
        print(f"  {g:<14}{v:+.3f}")

    print(f"\n{'coverage kept':<16}" + "".join(f"{k:>14}" for k in curves))
    for f in (1.0, 0.75, 0.5, 0.25):
        print(f"{int(f * 100):>3}%{'':<12}" + "".join(f"{curves[k][f]:>14.1f}" for k in curves))


if __name__ == "__main__":
    main()
