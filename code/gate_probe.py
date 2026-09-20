"""Probe the model-aware gates, on any backbone, and say which ones are honest.

  ../.venv/bin/python gate_probe.py [--model NAME] [--all] [--json] [--boot 2000]

gate_bench.py established the negative result: cheap pixel statistics rank nutrition
error no better than chance. This script scores every gate, pixel and model-aware, on
one protocol, and since 17 Sep 2026 it also says how sure we are: every number carries
a cluster-bootstrap confidence interval and every gate is tested against refusing at
random, with a multiple-comparison correction.

Gates, all of which reuse cached features and cost no extra inference:
  mahalanobis     PCA-whitened distance from the fit-set mean
  knn_dist        mean distance to the k nearest fit-set dishes
  ensemble_spread disagreement across B ridge heads fit on bootstrap resamples
  learned_error   a ridge trained to predict |error| directly, from features
  blend           a ridge over the four signals above
  pred_magnitude  CONTROL, and the important one. See below.
  lap_var, tenengrad, blockiness, hf_ratio
                  the four pixel metrics, read from results/predictions.csv and
                  signed so that "risky" is high (sharpness metrics are negated)

THE DEGENERACY CONTROL. |error| scales with dish size, so a gate can post a good MAE
at fixed coverage simply by refusing large dishes. `pred_magnitude` uses the predicted
calorie value as the risk score and does exactly that: it scores near the best learned
gate while refusing ~99% of dishes over 400 kcal, answering only for small meals. Any
gate must therefore be read against the ORACLE's selectivity, not against the row above
it, which is why every table below reports the kept-set mean calories and the share of
large dishes refused alongside the MAE.

SIGNIFICANCE. The 507 test dishes are not 507 independent draws: Nutrition5k scans a
plate several times in one session and |error| has an intraclass correlation of ~0.34
within a session. So the unit of resampling is the plate session (127 in test), not
the dish, and the pooled ladder keeps every damage level of a dish inside its session.
For each of `--boot` resamples we recompute every gate's whole risk-coverage curve on
the same resample, which makes the comparisons paired. Reported per gate:
  * MAE at each coverage on a 5% grid, with a 95% percentile interval;
  * AURC, the mean error across coverages 10%..100% (kcal), with an interval;
  * gap share, the fraction of the random-to-perfect AURC gap the gate captures;
  * the paired difference to the random gate at each coverage, its interval, a
    two-sided bootstrap p-value, and that p-value Holm-corrected across the family
    of candidate gates at the same coverage.

LEAK DISCIPLINE, which is the whole ballgame for `learned_error`:
  heads are fit on the FIT dishes;
  their errors are measured on the CALIBRATION dishes, which the heads never saw
  (and, since 17 Sep, share no plate session with them);
  the error predictor is trained on those calibration errors;
  everything is reported on the 507 TEST dishes, which nothing was fit on.

CONTROLLING FOR SIZE (added 20 Sep). The control above ties the learned head at 90%
answered, and the head's score has rank correlation ~0.77 with the predicted calorie
count, so a plain MAE-at-coverage cannot say whether a gate reads difficulty or just
size. Two further blocks answer that:
  clean_sized   the same gates, same kcal loss, but every gate must refuse the same
                share of photos WITHIN each predicted-size band (SIZE_BANDS bands, edges
                from the calibration split's predictions, so nothing on test defines
                them). Size cannot be the lever across bands; the control collapses to
                ~0 by construction, and whatever a gate keeps is difficulty signal.
  clean_rel     the same gates on relative error, |err| / max(true, REL_FLOOR_KCAL).
                This loss has the opposite bias (it rewards refusing small meals, and
                16% of test dishes are under 50 kcal), so it is a secondary check.
Each gate also carries `rho_size` (Spearman with the predicted calorie count) and
`rho_within_size` (mean Spearman with |error| inside predicted-size quartiles).

THE RANDOM REFERENCE is analytic: refusing at random has expected selective risk equal
to the answer-everything risk at every coverage, so that is the curve every gate is
tested against, on the same resample. Before 20 Sep it was one fixed random draw,
whose seed noise (±1–2 kcal) leaked into every paired difference.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field

import numpy as np

import n5k
from gate_bench import (MODEL, PIXEL_GATES, RESULTS, RUNGS, pick_alpha, ridge_fit,
                        ridge_predict, rows_for, spearman, standardiser)

SEED = 760
N_BOOTSTRAP = 32
KNN_K = 10
PCA_K = 64
BIG_DISH_KCAL = 400
SMALL_DISH_KCAL = 100      # "small meal" for the selectivity report
REL_FLOOR_KCAL = 50        # relative error is |err| / max(true, floor); 83 test dishes sit under it
SIZE_BANDS = 10            # predicted-size bands for the size-controlled block
BACKBONES = ["vit_base_patch16_clip_224.openai",
             "vit_base_patch16_siglip_224.v2_webli",
             "vit_base_patch16_dinov3.lvd1689m"]

COVERAGES = [round(1.0 - 0.05 * i, 2) for i in range(19)]      # 1.00 … 0.10
REPORT_COVERAGES = ("0.9", "0.75", "0.5", "0.25")
MODEL_GATES = ["mahalanobis", "knn_dist", "ensemble_spread", "learned_error", "blend",
               "pred_magnitude"]
# Sharpness-type metrics say "sharp = good", so their risk score is the negation.
PIXEL_SIGN = {"lap_var": -1.0, "tenengrad": -1.0, "blockiness": 1.0, "hf_ratio": -1.0}
CANDIDATES = MODEL_GATES + PIXEL_GATES           # the family for the Holm correction


def load_cache_for(model, split, tag):
    z = np.load(n5k.CACHE / f"{model.split('.')[0]}__{split}__{tag}.npz", allow_pickle=True)
    return z["features"], [str(d) for d in z["dish_ids"]]


def mahalanobis_scorer(Xfit, k=PCA_K):
    mu = Xfit.mean(0)
    _, S, Vt = np.linalg.svd(Xfit - mu, full_matrices=False)
    V, var = Vt[:k].T, (S[:k] ** 2) / (len(Xfit) - 1)
    return lambda X: np.sqrt((((X - mu) @ V) ** 2 / np.maximum(var, 1e-12)).sum(1))


def knn_scorer(Xfit, k=KNN_K):
    """Mean distance to the k nearest fit dishes.

    Mahalanobis asks whether a photo is far from the centre of the training cloud.
    This asks whether it has close neighbours at all, which is the better question
    when the cloud is lumpy rather than ellipsoidal.
    """
    fit_sq = (Xfit ** 2).sum(1)

    def score(X):
        d2 = fit_sq[None, :] - 2 * X @ Xfit.T + (X ** 2).sum(1)[:, None]
        return np.sqrt(np.maximum(np.sort(d2, axis=1)[:, :k], 0)).mean(1)

    return score


def ensemble_scorer(Xfit, yfit, alpha, b=N_BOOTSTRAP, seed=SEED):
    """Spread across heads fit on bootstrap resamples of the fit split.

    Cheap only because the head is a closed-form ridge on frozen features: B heads
    cost B matrix solves, and the backbone is never run again.
    """
    rng = np.random.default_rng(seed)
    heads = [ridge_fit(Xfit[idx], yfit[idx], alpha)
             for idx in (rng.integers(0, len(Xfit), len(Xfit)) for _ in range(b))]
    return lambda X: np.std([ridge_predict(h, X) for h in heads], axis=0)


# ---------------------------------------------------------------- fitted objects
@dataclass
class Fitted:
    """Everything fit on one backbone: the standardiser, the heads and the gates."""
    model: str
    fit_ids: list[str]
    calib_ids: list[str]
    test_ids: list[str]
    zscore: object
    heads: dict
    alphas: dict
    calib_err: np.ndarray                     # |error| of the calorie head on calibration
    scorers: dict = field(default_factory=dict)
    band_edges: np.ndarray = None             # predicted-kcal band edges, from calibration

    def size_band(self, pred_kcal: np.ndarray) -> np.ndarray:
        """Which predicted-size band a photo falls in. Edges come from calibration."""
        return np.searchsorted(self.band_edges, pred_kcal, side="right")

    def features(self, split: str, tag: str) -> tuple[np.ndarray, list[str]]:
        """Standardised features for the usable dishes of one split and rung.

        `fit` and `calib` are carved from train, so their clean features live in the
        train cache; degraded calibration rungs are cached under `calib` by
        `features.py --split calib`.
        """
        want = {"test": self.test_ids, "calib": self.calib_ids, "fit": self.fit_ids}[split]
        try:
            X, ids = load_cache_for(self.model, split, tag)
        except FileNotFoundError:
            if split == "test":
                raise
            X, ids = load_cache_for(self.model, "train", tag)
        have = set(ids)
        keep = [d for d in want if d in have]
        return self.zscore(rows_for(X, ids, keep).astype(np.float64)), keep

    def scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        s = {k: f(X) for k, f in self.scorers.items()}
        s["blend"] = ridge_predict(self.blend_head, np.column_stack(
            [s[k] for k in ("mahalanobis", "knn_dist", "ensemble_spread", "learned_error")]))
        s["pred_magnitude"] = ridge_predict(self.heads["cal"], X)
        return s


def fit_gates(model: str) -> Fitted:
    dishes = n5k.load_dishes()
    fit_ids, calib_ids = n5k.fit_calib_split(local_only=True)
    test_ids = n5k.usable_split("test", local_only=True)

    Xtr, tr_ids = load_cache_for(model, "train", "clean")
    known = set(tr_ids)
    fit_ids = [d for d in fit_ids if d in known]
    calib_ids = [d for d in calib_ids if d in known]

    zscore = standardiser(rows_for(Xtr, tr_ids, fit_ids).astype(np.float64))
    Xfit = zscore(rows_for(Xtr, tr_ids, fit_ids).astype(np.float64))
    Xcal = zscore(rows_for(Xtr, tr_ids, calib_ids).astype(np.float64))

    heads, alphas = {}, {}
    for target in ("cal", "mass"):
        yf = np.array([getattr(dishes[d], target) for d in fit_ids], float)
        yc = np.array([getattr(dishes[d], target) for d in calib_ids], float)
        a, _ = pick_alpha(Xfit, yf, Xcal, yc)
        heads[target], alphas[target] = ridge_fit(Xfit, yf, a), a

    ycal = np.array([dishes[d].cal for d in calib_ids], float)
    cal_err = np.abs(ridge_predict(heads["cal"], Xcal) - ycal)

    f = Fitted(model, fit_ids, calib_ids, test_ids, zscore, heads, alphas, cal_err)
    f.scorers = {
        "mahalanobis": mahalanobis_scorer(Xfit),
        "knn_dist": knn_scorer(Xfit),
        "ensemble_spread": ensemble_scorer(
            Xfit, np.array([dishes[d].cal for d in fit_ids], float), alphas["cal"]),
    }
    err_head = ridge_fit(Xcal, cal_err, alphas["cal"])
    f.scorers["learned_error"] = lambda X: ridge_predict(err_head, X)
    base = np.column_stack([f.scorers[k](Xcal) for k in
                            ("mahalanobis", "knn_dist", "ensemble_spread", "learned_error")])
    f.blend_head = ridge_fit(base, cal_err, 1.0)
    # size bands for the size-controlled block: deciles of the predicted calorie count
    # on the calibration dishes, so the test split plays no part in defining them
    cal_pred = ridge_predict(heads["cal"], Xcal)
    f.band_edges = np.quantile(cal_pred, np.linspace(0, 1, SIZE_BANDS + 1)[1:-1])
    return f


# ---------------------------------------------------------------- pixel metrics
def load_pixel_metrics() -> dict[tuple[str, str], dict[str, float]]:
    """(dish_id, rung) -> the four pixel metrics, from gate_bench's predictions file."""
    path = RESULTS / "predictions.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["dish_id"], r["rung"])] = {m: float(r[m]) for m in PIXEL_GATES}
    return out


# ---------------------------------------------------------------- the curve, fast
def curve(score: np.ndarray, err: np.ndarray, coverages=COVERAGES) -> np.ndarray:
    """MAE on the kept set at each coverage. Keep the LOWEST scores (least risky)."""
    order = np.argsort(score, kind="stable")
    csum = np.cumsum(err[order])
    n = len(err)
    ks = np.array([max(1, int(round(c * n))) for c in coverages])
    return csum[ks - 1] / ks


def curve_banded(score: np.ndarray, err: np.ndarray, band: np.ndarray,
                 coverages=COVERAGES) -> np.ndarray:
    """MAE on the kept set when the gate must keep the same share INSIDE every band.

    Refusing 10% overall by refusing 10% of each predicted-size band takes size off
    the table as a lever between bands; what is left is how well the gate ranks
    photos of about the same size.
    """
    ncov = len(coverages)
    tot, cnt = np.zeros(ncov), np.zeros(ncov)
    for b in np.unique(band):
        idx = np.flatnonzero(band == b)
        csum = np.cumsum(err[idx][np.argsort(score[idx], kind="stable")])
        ks = np.array([max(1, int(round(c * len(idx)))) for c in coverages])
        tot += csum[ks - 1]
        cnt += ks
    return tot / cnt


def aurc(vals: np.ndarray, coverages=COVERAGES) -> float:
    """Mean error across coverages 10%..100%, by the trapezoid rule. In kcal."""
    c = np.array(coverages)[::-1]
    v = np.asarray(vals)[::-1]
    return float(np.trapezoid(v, c) / (c[-1] - c[0]))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over one family of tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- one block
def evaluate_block(gates: dict[str, np.ndarray], err: np.ndarray, true: np.ndarray,
                   session: np.ndarray, n_boot: int, seed: int = SEED,
                   band: np.ndarray = None, size: np.ndarray = None) -> dict:
    """Score every gate on one set of rows, with the cluster bootstrap.

    `gates` holds the candidate gates only; the perfect refuser (score = the loss) and
    the random reference (analytic: the answer-everything loss at every coverage) are
    added here. With `band`, every gate keeps the same share inside each band. `size`
    is the predicted calorie count, for the two size diagnostics.
    """
    n = len(err)
    names = list(gates) + ["random", "perfect"]
    big, small = true > BIG_DISH_KCAL, true < SMALL_DISH_KCAL
    risk = (lambda s, e: curve_banded(s, e, band)) if band is not None else curve
    flat = lambda e: np.full(len(COVERAGES), e.mean())        # the random gate's expectation

    def refuse_sets(score, c):
        """Kept and refused rows at coverage c, honouring the bands if any."""
        if band is None:
            order = np.argsort(score, kind="stable")
            k = max(1, int(round(c * n)))
            return order[:k], order[k:]
        kept, refused = [], []
        for b in np.unique(band):
            idx = np.flatnonzero(band == b)
            order = idx[np.argsort(score[idx], kind="stable")]
            k = max(1, int(round(c * len(idx))))
            kept.append(order[:k]), refused.append(order[k:])
        return np.concatenate(kept), np.concatenate(refused)

    # point estimates
    point = {g: risk(gates[g], err) for g in gates}
    point["perfect"] = risk(err, err)
    point["random"] = flat(err)
    rows = {}
    for g in names:
        r = {"mae": {str(c): float(v) for c, v in zip(COVERAGES, point[g])},
             "aurc": aurc(point[g]), "selectivity": {}}
        if g == "random":
            r["rho"] = 0.0
            for c in (0.9, 0.5):
                r["selectivity"][str(c)] = {"kept_mean_kcal": float(true.mean()),
                                            "big_refused": 1 - c, "small_refused": 1 - c}
        else:
            s = err if g == "perfect" else gates[g]
            r["rho"] = spearman(s, err)
            if size is not None:
                r["rho_size"] = spearman(s, size)
                q = np.searchsorted(np.quantile(size, [0.25, 0.5, 0.75]), size, side="right")
                r["rho_within_size"] = float(np.mean(
                    [spearman(s[q == j], err[q == j]) for j in range(4)]))
            for c in (0.9, 0.5):
                kept, refused = refuse_sets(s, c)
                r["selectivity"][str(c)] = {
                    "kept_mean_kcal": float(true[kept].mean()),
                    "big_refused": float(big[refused].sum() / max(big.sum(), 1)),
                    "small_refused": float(small[refused].sum() / max(small.sum(), 1))}
        rows[g] = r
    gap = rows["random"]["aurc"] - rows["perfect"]["aurc"]
    for g in names:
        rows[g]["gap_share"] = float((rows["random"]["aurc"] - rows[g]["aurc"]) / gap)

    # cluster bootstrap over plate sessions, paired across gates
    rng = np.random.default_rng(seed)
    sess_ids = np.unique(session)
    members = {s: np.flatnonzero(session == s) for s in sess_ids}
    boot_curve = {g: np.empty((n_boot, len(COVERAGES))) for g in names}
    boot_rho = {g: np.empty(n_boot) for g in names}
    boot_all = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(sess_ids, size=len(sess_ids), replace=True)
        idx = np.concatenate([members[s] for s in draw])
        e = err[idx]
        bb = band[idx] if band is not None else None
        boot_all[b] = e.mean()
        boot_curve["random"][b], boot_rho["random"][b] = flat(e), 0.0
        boot_curve["perfect"][b] = curve_banded(e, e, bb) if bb is not None else curve(e, e)
        boot_rho["perfect"][b] = 1.0
        for g in gates:
            s = gates[g][idx]
            boot_curve[g][b] = curve_banded(s, e, bb) if bb is not None else curve(s, e)
            boot_rho[g][b] = spearman(s, e)

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5], axis=0)
        return lo, hi

    for g in names:
        lo, hi = ci(boot_curve[g])
        rows[g]["mae_ci"] = {str(c): [float(l), float(h)]
                             for c, l, h in zip(COVERAGES, lo, hi)}
        ba = np.array([aurc(v) for v in boot_curve[g]])
        rows[g]["aurc_ci"] = [float(x) for x in ci(ba)]
        rows[g]["rho_ci"] = [float(x) for x in ci(boot_rho[g])]
        rows[g]["gap_share_ci"] = [float(x) for x in ci(
            (np.array([aurc(v) for v in boot_curve["random"]]) - ba)
            / (np.array([aurc(v) for v in boot_curve["random"]])
               - np.array([aurc(v) for v in boot_curve["perfect"]])))]

    # paired tests against the random gate, Holm-corrected within each coverage
    for ci_idx, c in enumerate(COVERAGES):
        pv = {}
        for g in names:
            if g in ("random", "perfect"):
                continue
            d = boot_curve[g][:, ci_idx] - boot_curve["random"][:, ci_idx]
            point_d = float(point[g][ci_idx] - point["random"][ci_idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            p = float(min(1.0, max(p, 1.0 / n_boot)))
            rows[g].setdefault("vs_random", {})[str(c)] = {
                "diff": point_d, "ci": [float(lo), float(hi)], "p": p}
            if g in CANDIDATES:
                pv[g] = p
        for g, p_adj in holm(pv).items():
            rows[g]["vs_random"][str(c)]["p_holm"] = float(p_adj)

    # head-to-head between model-aware gates at the two reported operating points,
    # so "is A better than B" has a number and not just a ranking
    pairwise = {}
    for c in ("0.9", "0.5"):
        ci_idx = COVERAGES.index(float(c))
        pairwise[c] = {}
        for i, a in enumerate(MODEL_GATES):
            for b_ in MODEL_GATES[i + 1:]:
                if a not in names or b_ not in names:
                    continue
                d = boot_curve[a][:, ci_idx] - boot_curve[b_][:, ci_idx]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                pairwise[c][f"{a} - {b_}"] = {
                    "diff": float(point[a][ci_idx] - point[b_][ci_idx]),
                    "ci": [float(lo), float(hi)],
                    "p": float(min(1.0, max(p, 1.0 / n_boot)))}

    return {"n": int(n), "n_sessions": int(len(sess_ids)), "n_boot": int(n_boot),
            "answer_all": float(err.mean()),
            "answer_all_ci": [float(x) for x in ci(boot_all)],
            "gates": rows, "pairwise": pairwise}


# ---------------------------------------------------------------- probe
def probe(model: str, n_boot: int, pixel: dict, score_rows: list) -> dict:
    """Fit heads and gates on one backbone; report every gate on clean and pooled test."""
    dishes = n5k.load_dishes()
    f = fit_gates(model)
    session_of = n5k.session_ids(f.test_ids)

    out = {"model": model, "alpha": f.alphas, "n_fit": len(f.fit_ids),
           "n_calib": len(f.calib_ids), "n_test": len(f.test_ids),
           "calib_mae": float(f.calib_err.mean()), "coverages": COVERAGES,
           "rungs": [r[0] for r in RUNGS]}

    blocks = {}
    for tag, spec, label in RUNGS:
        try:
            X, ids = f.features("test", tag)
        except FileNotFoundError:
            print(f"  (no {tag} cache for {model}; skipped)")
            continue
        y = np.array([dishes[d].cal for d in ids], float)
        m = np.array([dishes[d].mass for d in ids], float)
        pred = ridge_predict(f.heads["cal"], X)
        pred_m = ridge_predict(f.heads["mass"], X)
        s = f.scores(X)
        px = {g: np.array([pixel.get((d, tag), {}).get(g, np.nan) for d in ids])
              for g in PIXEL_GATES}
        blocks[tag] = dict(ids=ids, y=y, err=np.abs(pred - y), scores=s, pixel=px,
                           session=np.array([session_of[d] for d in ids]))
        for i, d in enumerate(ids):
            score_rows.append({
                "backbone": model, "dish_id": d, "session": int(session_of[d]),
                "rung": tag, "damage": label,
                "cal_true": y[i], "cal_pred": float(pred[i]),
                "cal_abs_err": float(abs(pred[i] - y[i])),
                "mass_true": m[i], "mass_pred": float(pred_m[i]),
                "mass_abs_err": float(abs(pred_m[i] - m[i])),
                **{g: float(s[g][i]) for g in MODEL_GATES},
                **{g: float(px[g][i]) for g in PIXEL_GATES},
            })

    for label, tags in [("clean", ["clean"]), ("pooled", list(blocks)),
                        ("clean_sized", ["clean"]), ("clean_rel", ["clean"])]:
        err = np.concatenate([blocks[t]["err"] for t in tags])
        true = np.concatenate([blocks[t]["y"] for t in tags])
        session = np.concatenate([blocks[t]["session"] for t in tags])
        gates = {g: np.concatenate([blocks[t]["scores"][g] for t in tags])
                 for g in MODEL_GATES}
        for g in PIXEL_GATES:
            v = np.concatenate([blocks[t]["pixel"][g] for t in tags])
            if np.isfinite(v).all():
                gates[g] = PIXEL_SIGN[g] * v
        size = gates["pred_magnitude"]
        if label == "clean_sized":
            band = f.size_band(size)
            out[label] = evaluate_block(gates, err, true, session, n_boot,
                                        band=band, size=size)
            out[label]["bands"] = int(SIZE_BANDS)
            out[label]["band_edges"] = [float(x) for x in f.band_edges]
            out[label]["band_sizes"] = [int((band == b).sum()) for b in range(SIZE_BANDS)]
        elif label == "clean_rel":
            rel = err / np.maximum(true, REL_FLOOR_KCAL)
            out[label] = evaluate_block(gates, rel, true, session, n_boot, size=size)
            out[label]["floor_kcal"] = REL_FLOOR_KCAL
            out[label]["n_under_floor"] = int((true < REL_FLOOR_KCAL).sum())
        else:
            out[label] = evaluate_block(gates, err, true, session, n_boot, size=size)
        out[label]["rungs"] = tags

    # per-rung error, for the ladder table
    out["per_rung"] = {t: {"cal_mae": float(b["err"].mean()), "n": len(b["ids"])}
                       for t, b in blocks.items()}

    # predictor headline, clean test only
    X, ids = f.features("test", "clean")
    mass_true = np.array([dishes[d].mass for d in ids], float)
    out["predictor"] = {
        "cal_mae": float(blocks["clean"]["err"].mean()),
        "cal_mae_ci": out["clean"]["answer_all_ci"],
        "mass_mae": float(np.abs(ridge_predict(f.heads["mass"], X) - mass_true).mean()),
        "mass_rel": float(np.abs(ridge_predict(f.heads["mass"], X) - mass_true).mean()
                          / mass_true.mean()),
    }
    return out


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def report(res: dict) -> None:
    print(f"model {res['model']}")
    print(f"fit {res['n_fit']} | calib {res['n_calib']} | test {res['n_test']} "
          f"| calibration MAE {res['calib_mae']:.1f} kcal | alpha {res['alpha']}\n")
    for label in ("clean", "pooled"):
        blk = res[label]
        title = ("CLEAN PHOTOS ONLY, dish difficulty" if label == "clean"
                 else f"POOLED over {len(blk['rungs'])} damage levels")
        lo, hi = blk["answer_all_ci"]
        print(f"=== {title}  (n={blk['n']}, {blk['n_sessions']} sessions, "
              f"answer everything {blk['answer_all']:.1f} [{lo:.1f}, {hi:.1f}] kcal)")
        print(f"{'gate':<16}{'rho':>7}{'keep 90%':>19}{'keep 50%':>19}{'AURC':>17}"
              f"{'gap':>6}{'big refused 90/50':>19}")
        for g, r in sorted(blk["gates"].items(), key=lambda kv: kv[1]["aurc"]):
            def cell(c):
                v, (l, h) = r["mae"][c], r["mae_ci"][c]
                st = stars(r["vs_random"][c]["p_holm"]) if "vs_random" in r and "p_holm" in r["vs_random"][c] else ""
                return f"{v:6.1f} [{l:5.1f},{h:5.1f}]{st:<3}"
            al, ah = r["aurc_ci"]
            sel = r["selectivity"]
            print(f"{g:<16}{r['rho']:>+7.3f} {cell('0.9')}{cell('0.5')}"
                  f"{r['aurc']:>6.1f} [{al:4.1f},{ah:5.1f}]{100 * r['gap_share']:>5.0f}%"
                  f"{100 * sel['0.9']['big_refused']:>10.0f}%{100 * sel['0.5']['big_refused']:>8.0f}%")
        print("   stars: Holm-corrected paired bootstrap test against the random gate "
              "at that coverage (* .05, ** .01, *** .001)\n")

    # the two size-controlled views, at the operating point
    sz, rl = res["clean_sized"], res["clean_rel"]
    print(f"=== CONTROLLING FOR SIZE, clean photos, 90% answered  "
          f"({sz['bands']} predicted-size bands, {sz['band_sizes']} test dishes each; "
          f"relative error floor {rl['floor_kcal']} kcal, {rl['n_under_floor']} dishes under it)")
    print(f"{'gate':<16}{'rho size':>9}{'rho in-band':>12}{'plain kcal':>17}"
          f"{'within bands':>17}{'relative %':>17}")
    plain = res["clean"]["gates"]
    for g in MODEL_GATES + PIXEL_GATES + ["perfect"]:
        if g not in plain:
            continue
        def d(blk, scale=1.0):
            r = blk["gates"][g]
            if g == "perfect":
                v = r["mae"]["0.9"] - blk["gates"]["random"]["mae"]["0.9"]
                return f"{scale * v:+8.1f}       "
            v = r["vs_random"]["0.9"]
            return f"{scale * v['diff']:+8.1f}{stars(v['p_holm']):<3}     "
        print(f"{g:<16}{plain[g].get('rho_size', 0):>+9.2f}{plain[g].get('rho_within_size', 0):>+12.2f}"
              f"  {d(res['clean'])}{d(sz)}{d(rl, 100)}")
    print("   each column: change against refusing at random at 90% answered, paired, "
          "Holm-corrected over the ten candidates\n")
    p = res["predictor"]
    lo, hi = p["cal_mae_ci"]
    print(f"predictor on clean test: {p['cal_mae']:.1f} [{lo:.1f}, {hi:.1f}] kcal | "
          f"{p['mass_mae']:.1f} g ({100 * p['mass_rel']:.1f}% of mean mass)")
    print("per rung: " + "  ".join(f"{t} {v['cal_mae']:.1f}" for t, v in res["per_rung"].items()))


def write_scores(rows: list) -> None:
    path = RESULTS / "gate_scores.csv"
    cols = list(rows[0])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--all", action="store_true", help="probe every backbone in BACKBONES")
    ap.add_argument("--json", action="store_true", help="write results/gate_probe.json")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    pixel = load_pixel_metrics()
    if not pixel:
        print("results/predictions.csv missing: pixel gates will be skipped")
    models = BACKBONES if args.all else [args.model]
    results, score_rows = {}, []
    for m in models:
        res = probe(m, args.boot, pixel, score_rows)
        results[m] = res
        report(res)
        print("-" * 100)
    if args.json:
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / "gate_probe.json", "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"wrote {RESULTS / 'gate_probe.json'}")
        write_scores(score_rows)


if __name__ == "__main__":
    main()
