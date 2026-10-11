"""Robustness runs for the final talk (11 Oct 2026): strengthen what survived replicate.py.

  ../.venv/bin/python robustness.py [--repeats 5] [--boot 2000] [--json]
                                    [--skip-gates] [--skip-ranges] [--carves 20]

replicate.py showed that the update talk's headline came from a favourable calibration
carve and that a smaller within-size-band signal survives on new sessions. The runs
below make the surviving claims as solid as the data allow, check the range result the
same way, and try two fixes for the error head's instability. Results go to
results/robustness.json; the experiment log is in final-presentation-plan.md.

EXPERIMENT 1, ONE POOLED HEADLINE. All 3,262 usable dishes (2,755 train + 507 test) are
scored out of fold. Sessions are recomputed on the pooled timeline, where train and test
scans interleave: the 600 s rule then chains neighbouring plates into 366 clusters (430 of
the 507 test dishes share a cluster with train dishes), coarser than the 367 + 127
per-split sessions, so folds and the bootstrap use the coarser, more conservative unit.
Scoring is R repeats of K-fold cross-fitting by session, each
repeat with its own fold assignment and its own fit/calibration carves. Inside every
fold every head, gate and size band is refit on the other folds exactly as
gate_probe.fit_gates does on the official split, and each fold's models refuse the same
share inside their own fold (x size band). Each gate's effect is averaged over the R
repeats, which averages out the training-carve noise a single fit carries (sd about
2 kcal for the error head); the session bootstrap recomputes every repeat's curve on
the same resample, so its interval is for that averaged estimate. The between-repeat
sd is reported beside it.

EXPERIMENT 3, SENSITIVITY. The within-band effect at 95/90/80/70/50% answered and as
AURC over the whole curve, and with 5, 10 and 20 predicted-size bands.

EXPERIMENT 4, PLACEBO. In every repeat the error head is also retrained N_PLACEBO
times on its calibration errors shuffled (a) across all calibration dishes and (b) only
within each dish's predicted-size band. (a) knows nothing; (b) knows each meal's size
band and nothing inside it. Inside size bands both should sit on random, and the real
head is read against their spread, like with like: the real head averaged over the
repeats against each placebo averaged over the same repeats (and the first repeat alone).

EXPERIMENT 5, STABILISING THE ERROR HEAD (exploratory). The head moves by ~2 kcal from
one calibration carve to the next, so two variants:
  learned_error_bag  the mean of BAG_CARVES error heads, each trained on a different
                     calibration carve of the fold's pool
  learned_error_oof  one error head trained on cross-fitted errors of the WHOLE pool
                     (each pool dish's error measured by a calorie head that did not see
                     its session): about three times the calibration split's data

EXPERIMENT 7, DISAGREEMENT ACROSS BACKBONES (exploratory). backbone_spread is the
standard deviation of the three backbones' calorie predictions for a photo, all fit on
the same carve: a small ensemble of genuinely different models, free because all three
feature sets are cached.

EXPERIMENT 2, THE RANGES CHECKED THE SAME WAY. Same-width and CQR ranges (as in
conformal.py) refit over CARVES calibration carves and scored on the official test
split, and cross-fitted on the 367 train sessions; coverage overall and by true-meal-
size quartile.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import n5k
from conformal import conformal_quantile
from gate_bench import PIXEL_GATES, RESULTS, pick_alpha, ridge_fit, ridge_predict, standardiser
from gate_probe import (BACKBONES, COVERAGES, MODEL_GATES, PIXEL_SIGN, SEED, aurc,
                        ensemble_scorer, holm, knn_scorer, load_cache_for,
                        load_pixel_metrics, mahalanobis_scorer)
from replicate import carve_calib, session_folds, train_pixel_metrics

K_FOLDS = 5
N_REPEATS = 5
BAG_CARVES = 10
N_PLACEBO = 200
BAND_COUNTS = (5, 10, 20)
MAIN_BANDS = 10
N_CARVES = 20
ALPHA_COV = 0.1
SENS = ("0.95", "0.9", "0.8", "0.7", "0.5")
NEW_GATES = ["learned_error_bag", "learned_error_oof", "backbone_spread"]
FAMILY = MODEL_GATES + NEW_GATES + PIXEL_GATES          # Holm family at each coverage
PAIRS = [("learned_error", "pred_magnitude"), ("ensemble_spread", "learned_error"),
         ("learned_error_bag", "learned_error"), ("learned_error_oof", "learned_error"),
         ("backbone_spread", "ensemble_spread")]
OP = "0.9"
COV = np.array(COVERAGES)


# ---------------------------------------------------------------- data
def load_all():
    """Clean features of every usable train and test dish, rows aligned across backbones."""
    dishes = n5k.load_dishes()
    feats = {m: [load_cache_for(m, s, "clean") for s in ("train", "test")] for m in BACKBONES}
    known = set.intersection(*(set(a[1] + b[1]) for a, b in feats.values()))
    train = [d for d in n5k.usable_split("train", local_only=True) if d in known]
    test = [d for d in n5k.usable_split("test", local_only=True) if d in known]
    ids = train + test
    X = {}
    for m, ((Xa, ia), (Xb, ib)) in feats.items():
        pos = {d: i for i, d in enumerate(ia + ib)}
        X[m] = np.vstack([Xa, Xb])[[pos[d] for d in ids]].astype(np.float64)
    y = np.array([dishes[d].cal for d in ids], float)
    return ids, X, y, len(train)


def pixel_scores(ids, n_train):
    """The four pixel metrics on every clean photo, signed so that high = risky."""
    tr = train_pixel_metrics(ids[:n_train])
    te = load_pixel_metrics()
    out = {}
    for g in PIXEL_GATES:
        v = [tr[d][g] if i < n_train else te[(d, "clean")][g] for i, d in enumerate(ids)]
        out[g] = PIXEL_SIGN[g] * np.array(v)
    return out


# ---------------------------------------------------------------- fast curves
def curve_strata(score, err, strata):
    """MAE on the kept set at every coverage when each stratum keeps the same share.

    Same numbers as gate_probe.curve_banded (stable order inside a stratum, Python
    rounding of c * n), vectorised so the bootstrap can afford five repeats.
    """
    order = np.lexsort((score, strata))
    st = strata[order]
    cs = np.concatenate([[0.0], np.cumsum(err[order])])
    starts = np.flatnonzero(np.r_[True, st[1:] != st[:-1]])
    sizes = np.diff(np.r_[starts, len(st)])
    ks = np.maximum(1, np.rint(np.outer(sizes, COV)).astype(int))
    return (cs[starts[:, None] + ks] - cs[starts[:, None]]).sum(0) / ks.sum(0)


def kept_mean(score, strata, true, c=0.9):
    order = np.lexsort((score, strata))
    st = strata[order]
    starts = np.flatnonzero(np.r_[True, st[1:] != st[:-1]])
    sizes = np.diff(np.r_[starts, len(st)])
    keep = np.concatenate([order[s:s + max(1, int(round(c * n)))] for s, n in zip(starts, sizes)])
    return float(true[keep].mean())


def ridge_multi(X, Y, alpha):
    """Ridge for many targets at once (one solve): the placebo heads."""
    mu, ymu = X.mean(0), Y.mean(0)
    Xc = X - mu
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (Y - ymu))
    return mu, ymu, W


# ---------------------------------------------------------------- one fold, one backbone
def fit_fold(X, y, ids, pool, held, seed, placebo, rng):
    pos = {d: i for i, d in enumerate(ids)}
    pool_ids = [ids[i] for i in pool]
    fit_ids, cal_ids = carve_calib(pool_ids, seed)
    fi = np.array([pos[d] for d in fit_ids])
    ci = np.array([pos[d] for d in cal_ids])
    z = standardiser(X[fi])
    Xf, Xc, Xh = z(X[fi]), z(X[ci]), z(X[held])
    a, _ = pick_alpha(Xf, y[fi], Xc, y[ci])
    head = ridge_fit(Xf, y[fi], a)
    cal_pred = ridge_predict(head, Xc)
    cal_err = np.abs(cal_pred - y[ci])

    sc = {"mahalanobis": mahalanobis_scorer(Xf), "knn_dist": knn_scorer(Xf),
          "ensemble_spread": ensemble_scorer(Xf, y[fi], a)}
    err_head = ridge_fit(Xc, cal_err, a)
    sc["learned_error"] = lambda Z: ridge_predict(err_head, Z)
    four = ("mahalanobis", "knn_dist", "ensemble_spread", "learned_error")
    blend = ridge_fit(np.column_stack([sc[k](Xc) for k in four]), cal_err, 1.0)
    s = {k: f(Xh) for k, f in sc.items()}
    s["blend"] = ridge_predict(blend, np.column_stack([s[k] for k in four]))
    pred = ridge_predict(head, Xh)
    s["pred_magnitude"] = pred

    # experiment 5a: error heads over BAG_CARVES calibration carves, averaged
    bag = s["learned_error"].copy()
    for c in range(1, BAG_CARVES):
        f2, c2 = carve_calib(pool_ids, seed + 1000 * c)
        fi2 = np.array([pos[d] for d in f2])
        ci2 = np.array([pos[d] for d in c2])
        z2 = standardiser(X[fi2])
        h2 = ridge_fit(z2(X[fi2]), y[fi2], a)
        e2 = np.abs(ridge_predict(h2, z2(X[ci2])) - y[ci2])
        bag += ridge_predict(ridge_fit(z2(X[ci2]), e2, a), z2(X[held]))
    s["learned_error_bag"] = bag / BAG_CARVES

    # experiment 5b: one error head on cross-fitted errors of the whole pool
    inner = session_folds(pool_ids, K_FOLDS, seed)
    fold_in = np.array([inner[d] for d in pool_ids])
    Xp = z(X[pool])
    oof = np.empty(len(pool))
    for j in range(K_FOLDS):
        h = ridge_fit(Xp[fold_in != j], y[pool][fold_in != j], a)
        oof[fold_in == j] = ridge_predict(h, Xp[fold_in == j])
    s["learned_error_oof"] = ridge_predict(ridge_fit(Xp, np.abs(oof - y[pool]), a), Xh)

    out = dict(pred=pred, scores=s, cal_pred=cal_pred, alpha=a)
    if placebo:
        # experiment 4: error heads trained on shuffled calibration errors
        edges = np.quantile(cal_pred, np.linspace(0, 1, MAIN_BANDS + 1)[1:-1])
        band = np.searchsorted(edges, cal_pred, side="right")
        Yg = np.column_stack([rng.permutation(cal_err) for _ in range(N_PLACEBO)])
        Yb = np.empty((len(cal_err), N_PLACEBO))
        for p in range(N_PLACEBO):
            col = cal_err.copy()
            for b in np.unique(band):
                idx = np.flatnonzero(band == b)
                col[idx] = cal_err[rng.permutation(idx)]
            Yb[:, p] = col
        out["placebo_global"] = ridge_predict(ridge_multi(Xc, Yg, a), Xh)
        out["placebo_band"] = ridge_predict(ridge_multi(Xc, Yb, a), Xh)
    return out


# ---------------------------------------------------------------- experiments 1, 3, 4, 5, 7
def run_pooled(ids, X, y, session, n_repeats, placebo_repeats=N_REPEATS):
    """Out-of-fold scores for every dish, every repeat, every backbone."""
    n = len(ids)
    runs = {m: [] for m in BACKBONES}
    placebo = {m: {} for m in BACKBONES}
    rng = np.random.default_rng(SEED)
    for r in range(n_repeats):
        fold_of = session_folds(ids, K_FOLDS, seed=SEED + r)
        fold = np.array([fold_of[d] for d in ids])
        per = {m: dict(pred=np.empty(n), scores={}, band={nb: np.empty(n, int)
                                                          for nb in BAND_COUNTS})
               for m in BACKBONES}
        for k in range(K_FOLDS):
            held, pool = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
            for m in BACKBONES:
                res = fit_fold(X[m], y, ids, pool, held, SEED + 100 * r + k,
                               r < placebo_repeats, rng)
                per[m]["pred"][held] = res["pred"]
                for g, v in res["scores"].items():
                    per[m]["scores"].setdefault(g, np.empty(n))[held] = v
                for nb in BAND_COUNTS:
                    e = np.quantile(res["cal_pred"], np.linspace(0, 1, nb + 1)[1:-1])
                    per[m]["band"][nb][held] = np.searchsorted(e, res["pred"], side="right")
                if r < placebo_repeats:
                    for key in ("placebo_global", "placebo_band"):
                        reps = placebo[m].setdefault(key, [])
                        if len(reps) == r:
                            reps.append(np.empty((n, N_PLACEBO)))
                        reps[r][held] = res[key]
                print(f"  repeat {r} fold {k} {m.split('.')[0]:<28} alpha {res['alpha']:.0f}")
            spread = np.std([per[m]["pred"][held] for m in BACKBONES], axis=0)
            for m in BACKBONES:
                per[m]["scores"].setdefault("backbone_spread", np.empty(n))[held] = spread
        for m in BACKBONES:
            runs[m].append(dict(err=np.abs(per[m]["pred"] - y), scores=per[m]["scores"],
                                fold=fold, band=per[m]["band"]))
    return runs, placebo


def evaluate(runs, y, session, pix, n_boot, nb):
    """Effects averaged over repeats, with a session bootstrap; nb=None means no bands."""
    R = len(runs)
    strata = [run["fold"] * (nb or 1) + (run["band"][nb] if nb else 0) for run in runs]
    gates = FAMILY + ["perfect"]

    def score_of(run, g, idx):
        if g == "perfect":
            return run["err"][idx]
        return pix[g][idx] if g in PIXEL_GATES else run["scores"][g][idx]

    def diffs(idx):
        """Per-repeat change vs random at every coverage, shape (R, gates, coverages)."""
        out = np.empty((R, len(gates), len(COV)))
        for r, (run, st) in enumerate(zip(runs, strata)):
            e = run["err"][idx]
            for j, g in enumerate(gates):
                out[r, j] = curve_strata(score_of(run, g, idx), e, st[idx]) - e.mean()
        return out

    allidx = np.arange(len(y))
    point_r = diffs(allidx)
    point = point_r.mean(0)
    sids = np.unique(session)
    members = {s: np.flatnonzero(session == s) for s in sids}
    rng = np.random.default_rng(SEED)
    boot = np.empty((n_boot, len(gates), len(COV)))
    for b in range(n_boot):
        idx = np.concatenate([members[s] for s in rng.choice(sids, len(sids), replace=True)])
        boot[b] = diffs(idx).mean(0)

    def summarise(d_boot, d_point):
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        p = 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())
        return dict(diff=float(d_point), ci=[float(lo), float(hi)],
                    p=float(min(1.0, max(p, 1.0 / n_boot))))

    res = {"bands": nb, "n": int(len(y)), "n_sessions": int(len(sids)), "repeats": R,
           "n_boot": n_boot, "answer_all": float(np.mean([r["err"].mean() for r in runs])),
           "gates": {}}
    rev = COV[::-1]
    a_points, a_boots = {}, {}
    for j, g in enumerate(gates):
        a_point = float(np.trapezoid(point[j][::-1], rev) / (rev[-1] - rev[0]))
        a_boot = np.trapezoid(boot[:, j, ::-1], rev, axis=1) / (rev[-1] - rev[0])
        a_points[g], a_boots[g] = a_point, a_boot
        row = {"coverage": {}, "aurc_diff": summarise(a_boot, a_point),
               "repeat_sd_90": float(point_r[:, j, list(COVERAGES).index(0.9)].std(ddof=1))
               if R > 1 else 0.0,
               "per_repeat_90": [float(v) for v in point_r[:, j, list(COVERAGES).index(0.9)]]}
        for c in SENS:
            ci = list(COVERAGES).index(float(c))
            row["coverage"][c] = summarise(boot[:, j, ci], point[j, ci])
        row["kept_mean_kcal_90"] = float(np.mean(
            [kept_mean(score_of(run, g, allidx), st, y) for run, st in zip(runs, strata)]))
        res["gates"][g] = row
    for c in SENS:
        adj = holm({g: res["gates"][g]["coverage"][c]["p"] for g in FAMILY})
        for g, p in adj.items():
            res["gates"][g]["coverage"][c]["p_holm"] = float(p)
    res["pairs_90"], res["pairs_50"], res["pairs_aurc"] = {}, {}, {}
    for a, b in PAIRS:
        ja, jb = gates.index(a), gates.index(b)
        for key, c in (("pairs_90", 0.9), ("pairs_50", 0.5)):
            ci = list(COVERAGES).index(c)
            res[key][f"{a} - {b}"] = summarise(boot[:, ja, ci] - boot[:, jb, ci],
                                               point[ja, ci] - point[jb, ci])
        res["pairs_aurc"][f"{a} - {b}"] = summarise(a_boots[a] - a_boots[b],
                                                    a_points[a] - a_points[b])
    gap = -res["gates"]["perfect"]["coverage"][OP]["diff"]
    res["share_of_perfect_90"] = {g: -res["gates"][g]["coverage"][OP]["diff"] / gap
                                  for g in FAMILY}
    return res


def evaluate_placebo(runs, placebo, y, nb=MAIN_BANDS):
    """The real head against heads trained on shuffled errors, inside bands, at 90%.

    Compared like with like: the real head's effect averaged over the repeats that have
    placebos, against each placebo index's effect averaged over the same repeats (the
    placebo heads are fresh shuffles in every repeat). The first repeat alone is also
    reported.
    """
    ci90 = list(COVERAGES).index(0.9)
    n_rep = len(placebo["placebo_global"])

    def eff(score, run):
        st = run["fold"] * nb + run["band"][nb]
        return curve_strata(score, run["err"], st)[ci90] - run["err"].mean()

    real = np.array([eff(runs[r]["scores"]["learned_error"], runs[r]) for r in range(n_rep)])
    out = {"repeats": n_rep, "n_placebo": N_PLACEBO,
           "real_head_first_repeat": float(real[0]), "real_head_averaged": float(real.mean())}
    for key in ("placebo_global", "placebo_band"):
        v = np.array([[eff(placebo[key][r][:, p], runs[r]) for p in range(N_PLACEBO)]
                      for r in range(n_rep)])                   # repeats x placebos
        blk = {}
        for label, vals, ref in (("first_repeat", v[0], real[0]),
                                 ("averaged", v.mean(0), real.mean())):
            blk[label] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                          "pct": [float(x) for x in np.percentile(vals, [2.5, 50, 97.5])],
                          "real": float(ref),
                          "z": float((ref - vals.mean()) / vals.std(ddof=1)),
                          "p": float(((vals <= ref).sum() + 1) / (len(vals) + 1))}
        out[key] = blk
    return out


# ---------------------------------------------------------------- experiment 2, ranges
def score_ranges(y, lo, hi, session, n_boot, rng):
    lo = np.maximum(0.0, lo)
    cov = ((y >= lo) & (y <= hi)).astype(float)
    q = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
    miss = np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
    out = {"coverage": float(cov.mean()), "width": float((hi - lo).mean()),
           "interval_score": float((hi - lo + (2 / ALPHA_COV) * miss).mean()),
           "by_kcal": [float(cov[q == b].mean()) for b in range(4)]}
    if n_boot:
        sids = np.unique(session)
        members = {s: np.flatnonzero(session == s) for s in sids}
        allc, big = np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            idx = np.concatenate([members[s] for s in rng.choice(sids, len(sids), replace=True)])
            allc[b] = cov[idx].mean()
            big[b] = cov[idx][q[idx] == 3].mean()
        out["coverage_ci"] = [float(x) for x in np.percentile(allc, [2.5, 97.5])]
        out["largest_quarter_ci"] = [float(x) for x in np.percentile(big, [2.5, 97.5])]
    return out


def fit_ranges(X, y, ids, pool, target, seed):
    """Same-width and CQR ranges for `target`, everything fit on `pool` (as conformal.py)."""
    from sklearn.linear_model import QuantileRegressor
    pos = {d: i for i, d in enumerate(ids)}
    fit_ids, cal_ids = carve_calib([ids[i] for i in pool], seed)
    fi = np.array([pos[d] for d in fit_ids])
    ci = np.array([pos[d] for d in cal_ids])
    z = standardiser(X[fi])
    Xf, Xc, Xt = z(X[fi]), z(X[ci]), z(X[target])
    a, _ = pick_alpha(Xf, y[fi], Xc, y[ci])
    head = ridge_fit(Xf, y[fi], a)
    yhat_c, yhat_t = ridge_predict(head, Xc), ridge_predict(head, Xt)
    q = conformal_quantile(np.abs(y[ci] - yhat_c), ALPHA_COV)
    const = (yhat_t - q, yhat_t + q)
    # out-of-fold predictions over the fit split, folds as in conformal.oof_predictions
    sess = n5k.session_ids(fit_ids)
    order = sorted(set(sess.values()))
    np.random.default_rng(SEED).shuffle(order)
    fs = {s: i % K_FOLDS for i, s in enumerate(order)}
    ff = np.array([fs[sess[d]] for d in fit_ids])
    oof = np.empty(len(fi))
    for j in range(K_FOLDS):
        h = ridge_fit(Xf[ff != j], y[fi][ff != j], a)
        oof[ff == j] = ridge_predict(h, Xf[ff == j])
    qs = (ALPHA_COV / 2, 1 - ALPHA_COV / 2)
    qr = {t: QuantileRegressor(quantile=t, alpha=0.0, solver="highs").fit(oof[:, None], y[fi])
          for t in qs}
    lo_c, hi_c = (qr[t].predict(yhat_c[:, None]) for t in qs)
    Q = conformal_quantile(np.maximum(lo_c - y[ci], y[ci] - hi_c), ALPHA_COV)
    lo_t, hi_t = (qr[t].predict(yhat_t[:, None]) for t in qs)
    return const, (lo_t - Q, hi_t + Q)


def run_ranges(ids, X, y, session, n_train, n_carves, n_boot):
    train, test = np.arange(n_train), np.arange(n_train, len(ids))
    out = {}
    for m in BACKBONES:
        rng = np.random.default_rng(SEED)
        seeds = [SEED] + [s for s in range(1, n_carves + 5) if s != SEED][:n_carves - 1]
        carves = {"constant": [], "cqr": []}
        for s in seeds:
            const, cqr = fit_ranges(X[m], y, ids, train, test, s)
            for name, (lo, hi) in (("constant", const), ("cqr", cqr)):
                carves[name].append(score_ranges(y[test], lo, hi, session[test], 0, rng))
        # new sessions: cross-fitted on the train split only
        fold_of = session_folds([ids[i] for i in train], K_FOLDS, SEED)
        fold = np.array([fold_of[ids[i]] for i in train])
        lo = {"constant": np.empty(n_train), "cqr": np.empty(n_train)}
        hi = {"constant": np.empty(n_train), "cqr": np.empty(n_train)}
        for k in range(K_FOLDS):
            held, pool = train[fold == k], train[fold != k]
            const, cqr = fit_ranges(X[m], y, ids, pool, held, SEED + k)
            for name, (l, h) in (("constant", const), ("cqr", cqr)):
                lo[name][held], hi[name][held] = l, h
        new = {name: score_ranges(y[train], lo[name], hi[name], session[train], n_boot, rng)
               for name in ("constant", "cqr")}
        summ = {}
        for name, rows in carves.items():
            big = np.array([r["by_kcal"][3] for r in rows])
            small = np.array([r["by_kcal"][0] for r in rows])
            cov = np.array([r["coverage"] for r in rows])
            isc = np.array([r["interval_score"] for r in rows])
            summ[name] = {"coverage_mean": float(cov.mean()), "coverage_range": [float(cov.min()), float(cov.max())],
                          "largest_mean": float(big.mean()), "largest_range": [float(big.min()), float(big.max())],
                          "smallest_mean": float(small.mean()), "smallest_range": [float(small.min()), float(small.max())],
                          "interval_score_mean": float(isc.mean())}
        out[m] = {"seeds": seeds, "carves": carves, "carve_summary": summ, "new_sessions": new}
        print(f"  ranges {m.split('.')[0]}: largest quarter, carves const "
              f"{100 * summ['constant']['largest_mean']:.0f}% vs CQR {100 * summ['cqr']['largest_mean']:.0f}% "
              f"(CQR range {100 * summ['cqr']['largest_range'][0]:.0f}-{100 * summ['cqr']['largest_range'][1]:.0f}); "
              f"new sessions const {100 * new['constant']['by_kcal'][3]:.0f}% vs CQR {100 * new['cqr']['by_kcal'][3]:.0f}%")
    return out


# ---------------------------------------------------------------- report
def report(m, blocks, plc):
    print(f"\n=== {m}")
    b = blocks[str(MAIN_BANDS)]
    print(f"pooled, {b['n']} dishes, {b['n_sessions']} sessions, {b['repeats']} repeats; "
          f"answer everything {b['answer_all']:.1f} kcal; inside {MAIN_BANDS} size bands, 90% answered:")
    for g in FAMILY + ["perfect"]:
        r = b["gates"][g]["coverage"][OP]
        ph = r.get("p_holm", float("nan"))
        print(f"  {g:<20}{r['diff']:+6.2f} [{r['ci'][0]:+5.2f}, {r['ci'][1]:+5.2f}]  Holm p {ph:.3f}"
              f"  repeat sd {b['gates'][g]['repeat_sd_90']:.2f}  kept {b['gates'][g]['kept_mean_kcal_90']:.0f} kcal")
    for k in b["pairs_90"]:
        cells = []
        for key, lab in (("pairs_90", "90%"), ("pairs_50", "50%"), ("pairs_aurc", "AURC")):
            v = b[key][k]
            cells.append(f"{lab} {v['diff']:+5.2f} [{v['ci'][0]:+5.2f}, {v['ci'][1]:+5.2f}] p {v['p']:.3f}")
        print(f"  pair {k:<38}" + " | ".join(cells))
    print("  AURC vs random: " + "  ".join(
        f"{g} {b['gates'][g]['aurc_diff']['diff']:+.1f}" for g in
        ("learned_error", "learned_error_bag", "learned_error_oof", "ensemble_spread",
         "backbone_spread", "pred_magnitude", "perfect")))
    print("  sensitivity (head / ensemble / control / bag / oof / backbone_spread, kcal vs random):")
    for key, blk in blocks.items():
        cells = []
        for c in SENS:
            cells.append(c + ": " + " ".join(f"{blk['gates'][g]['coverage'][c]['diff']:+.1f}" for g in
                         ("learned_error", "ensemble_spread", "pred_magnitude", "learned_error_bag",
                          "learned_error_oof", "backbone_spread")))
        print(f"   bands {key:>4}: " + " | ".join(cells))
    print_placebo(plc)


def print_placebo(plc):
    for label in ("first_repeat", "averaged"):
        g, b = plc["placebo_global"][label], plc["placebo_band"][label]
        print(f"  placebo, {label:<12} (in bands, 90%): real head {g['real']:+.2f} | shuffled "
              f"{g['mean']:+.2f} ± {g['sd']:.2f}, z {g['z']:+.1f}, p {g['p']:.3f} | shuffled "
              f"within band {b['mean']:+.2f} ± {b['sd']:.2f}, z {b['z']:+.1f}, p {b['p']:.3f}")


def main() -> None:
    global BAG_CARVES
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--carves", type=int, default=N_CARVES)
    ap.add_argument("--json", action="store_true", help="write results/robustness.json")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--skip-ranges", action="store_true")
    ap.add_argument("--bag", type=int, default=BAG_CARVES, help="carves in the bagged head")
    ap.add_argument("--placebo-repeats", type=int, default=N_REPEATS,
                    help="repeats that also fit placebo heads")
    ap.add_argument("--placebo-only", action="store_true",
                    help="refit, evaluate only the placebo, merge into robustness.json")
    args = ap.parse_args()
    BAG_CARVES = args.bag

    ids, X, y, n_train = load_all()
    sess = n5k.session_ids(ids)
    session = np.array([sess[d] for d in ids])
    print(f"{len(ids)} dishes ({n_train} train + {len(ids) - n_train} test), "
          f"{len(np.unique(session))} sessions")
    out = {"meta": {"k_folds": K_FOLDS, "repeats": args.repeats, "bag_carves": BAG_CARVES,
                    "n_placebo": N_PLACEBO, "band_counts": list(BAND_COUNTS),
                    "n_boot": args.boot, "carves": args.carves, "alpha_cov": ALPHA_COV}}
    if args.placebo_only:
        path = RESULTS / "robustness.json"
        out = json.loads(path.read_text()) if path.exists() else out
        runs, placebo = run_pooled(ids, X, y, session, args.repeats, args.placebo_repeats)
        for m in BACKBONES:
            out.setdefault(m, {})["placebo"] = evaluate_placebo(runs[m], placebo[m], y)
            print(f"\n=== {m}")
            print_placebo(out[m]["placebo"])
        out["meta"]["placebo_repeats"] = args.placebo_repeats
        if args.json:
            path.write_text(json.dumps(out, indent=1))
            print(f"\nwrote {path}")
        return
    if not args.skip_gates:
        pix = pixel_scores(ids, n_train)
        runs, placebo = run_pooled(ids, X, y, session, args.repeats, args.placebo_repeats)
        for m in BACKBONES:
            blocks = {"none": evaluate(runs[m], y, session, pix, args.boot, None)}
            for nb in BAND_COUNTS:
                blocks[str(nb)] = evaluate(runs[m], y, session, pix,
                                           args.boot if nb == MAIN_BANDS else args.boot // 2, nb)
            plc = evaluate_placebo(runs[m], placebo[m], y)
            out.setdefault(m, {})["pooled"] = blocks
            out[m]["placebo"] = plc
            report(m, blocks, plc)
    if not args.skip_ranges:
        rng_out = run_ranges(ids, X, y, session, n_train, args.carves, args.boot)
        for m, v in rng_out.items():
            out.setdefault(m, {})["ranges"] = v
    if args.json:
        with open(RESULTS / "robustness.json", "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {RESULTS / 'robustness.json'}")


if __name__ == "__main__":
    main()
