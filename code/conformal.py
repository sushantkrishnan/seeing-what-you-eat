"""Confidence ranges on top of the frozen predictor: split conformal, first cut.

  ../.venv/bin/python conformal.py [--model NAME] [--all] [--json] [--alpha 0.1] [--boot 2000]

Objective O3 of the proposal asks whether the best gate signal gives ranges worth
having. This is the first measurement. Nothing here is trained: the calorie head is the
frozen ridge from gate_bench.py, and the range is split conformal prediction on the
831 calibration dishes the head never saw.

THE DESIGN POINT. Plain split conformal gives every photo the same width, the
calibration quantile of |y - yhat|, so it cannot say "this photo is harder". The
normalised score |y - yhat| / sigma(x) lets a gate signal set the width, and then
"which gate" and "how wide" are one question. We try sigma = 1 (plain), the ensemble
spread, the k-NN distance and the Mahalanobis distance. learned_error is excluded as a
width model because it is trained on the calibration errors, which would make the
calibration quantile optimistic.

WHAT IS REPORTED, per width model, on the 507 clean test dishes:
  coverage        share of dishes whose true calories fall inside the range, with a
                  cluster-bootstrap interval over plate sessions (target 1 - alpha)
  width           mean and median range width in kcal
  usable share    share of ranges narrower than +/-100 kcal
  honesty         coverage inside each quartile of the width model, and the largest
                  shortfall below the target across those quartiles
  interval score  Winkler's proper score: width plus 2/alpha times the miss distance
  range-as-gate   answer only when the range is narrower than W: share answered,
                  coverage and error on the answered set, and what got refused

EXCHANGEABILITY. Coverage claims need the calibration and test photos to come from the
same distribution. For damaged photos that means one damage level per dish, drawn from
a stated deployment mixture, for calibration and test alike. The mixture block draws
three mixtures, calibrates once on clean photos and once on the mixture, and reports
coverage both ways. Calibrating on clean and deploying on damaged photos is expected to
under-cover; measuring by how much is the point, and it is a check, not a finding.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import n5k
from gate_bench import MODEL, RESULTS, RUNGS, ridge_fit, ridge_predict
from gate_probe import BACKBONES, BIG_DISH_KCAL, SEED, fit_gates

WIDTH_MODELS = ["constant", "ensemble_spread", "knn_dist", "mahalanobis"]
# Size-aware ranges (added 9 Oct), the O3 fix for the largest meals:
#   size_scaled    sigma = the predicted calorie count (floored at REL_FLOOR_KCAL), so the
#                  range grows in proportion to the meal
#   mondrian_size  a separate conformal quantile inside each predicted-size quartile
#                  (edges from calibration): coverage holds within each quartile by
#                  construction, which is what "honest for large meals" asks for
#   cqr            conformalised quantile regression: linear 5% and 95% quantiles of the
#                  true count given the predicted count, fit on out-of-fold predictions
#                  over the fit split, then widened on calibration. The only asymmetric
#                  range: big meals are under-predicted, so it can reach further up.
SIZE_WIDTHS = ["size_scaled", "mondrian_size", "cqr"]
REL_FLOOR_KCAL = 50.0
SIZE_QUARTILES = 4
GATE_ANSWER = 0.90            # the gate for the answered-only block refuses 1 photo in 10
USABLE_HALF_WIDTH = 100.0
GATE_WIDTHS = (150.0, 200.0, 300.0)

# Deployment mixtures: probability of each cached damage level for a submitted photo.
# Named so the sensitivity table reads in plain English.
MIXTURES = {
    "mostly clean": {"clean": 0.70, "phone": 0.15, "blur3": 0.05, "jpeg10": 0.05,
                     "crop0p4": 0.05},
    "phone heavy": {"clean": 0.30, "phone": 0.40, "blur3": 0.10, "downscale8": 0.10,
                    "crop0p4": 0.10},
    "uniform": {r[0]: 1.0 / len(RUNGS) for r in RUNGS},
}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


def sigma_of(f, X: np.ndarray) -> dict[str, np.ndarray]:
    s = f.scores(X)
    return {"constant": np.ones(len(X)),
            "ensemble_spread": s["ensemble_spread"],
            "knn_dist": s["knn_dist"],
            "mahalanobis": s["mahalanobis"]}


def cluster_ci(values: np.ndarray, session: np.ndarray, n_boot: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    ids = np.unique(session)
    members = {s: np.flatnonzero(session == s) for s in ids}
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[s] for s in rng.choice(ids, len(ids), replace=True)])
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(y, yhat, sigma, q, alpha, session, n_boot, err=None, difficulty=None) -> dict:
    """Score one set of symmetric ranges yhat +/- q * sigma."""
    return evaluate_ranges(y, np.maximum(0.0, yhat - q * sigma), yhat + q * sigma, alpha,
                           session, n_boot, err, sigma if difficulty is None else difficulty,
                           q=q)


def evaluate_ranges(y, lo, hi, alpha, session, n_boot, err=None, difficulty=None,
                    q=float("nan")) -> dict:
    """Score one set of ranges. `difficulty` is the axis for the honesty check, shared
    by every width model so that they are compared on the same quartiles."""
    lo = np.maximum(0.0, lo)
    covered = ((y >= lo) & (y <= hi)).astype(float)
    width = hi - lo
    miss = np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
    score = width + (2.0 / alpha) * miss
    out = {"q": float(q),
           "coverage": float(covered.mean()),
           "coverage_ci": cluster_ci(covered, session, n_boot),
           "width_mean": float(width.mean()), "width_median": float(np.median(width)),
           "usable_share": float((width <= 2 * USABLE_HALF_WIDTH).mean()),
           "interval_score": float(score.mean())}
    # honesty across width-model quartiles and across meal-size quartiles
    for name, key in (("by_difficulty", difficulty), ("by_kcal", y)):
        edges = np.quantile(key, [0.25, 0.5, 0.75])
        bins = np.digitize(key, edges)
        cov = [float(covered[bins == b].mean()) for b in range(4)]
        wid = [float(width[bins == b].mean()) for b in range(4)]
        out[name] = {"coverage": cov, "width": wid}
    out["honesty_gap"] = float(max(0.0, (1 - alpha) - min(out["by_difficulty"]["coverage"])))
    if err is not None:
        big = y > BIG_DISH_KCAL
        out["as_gate"] = {}
        for W in GATE_WIDTHS:
            ans = width <= W
            if ans.sum() == 0:
                continue
            out["as_gate"][str(int(W))] = {
                "answered": float(ans.mean()),
                "coverage": float(covered[ans].mean()),
                "mae": float(err[ans].mean()),
                "kept_mean_kcal": float(y[ans].mean()),
                "big_refused": float(big[~ans].sum() / max(big.sum(), 1))}
    return out


def draw_mixture(ids, mixture, rng):
    tags = list(mixture)
    p = np.array([mixture[t] for t in tags], float)
    p /= p.sum()
    return {d: tags[i] for d, i in zip(ids, rng.choice(len(tags), len(ids), p=p))}


def oof_predictions(f, dishes, folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold calorie predictions over the fit split, folds by plate session.

    The quantile regression inside CQR must learn how truth spreads around a
    prediction the head did not train on; in-sample fit-set predictions would make
    that spread look too tight.
    """
    X, ids = f.features("fit", "clean")
    y = np.array([dishes[d].cal for d in ids], float)
    sess = n5k.session_ids(ids)
    order = sorted(set(sess.values()))
    np.random.default_rng(SEED).shuffle(order)
    fold_of_sess = {s: i % folds for i, s in enumerate(order)}
    fold = np.array([fold_of_sess[sess[d]] for d in ids])
    oof = np.empty(len(ids))
    for k in range(folds):
        head = ridge_fit(X[fold != k], y[fold != k], f.alphas["cal"])
        oof[fold == k] = ridge_predict(head, X[fold == k])
    return oof, y


def size_aware_ranges(f, dishes, alpha, yhat_c, yc, yhat_t) -> dict[str, tuple]:
    """(lo, hi) on test for each size-aware width model, all calibrated on `yc`."""
    out = {}
    sig_c, sig_t = np.maximum(yhat_c, REL_FLOOR_KCAL), np.maximum(yhat_t, REL_FLOOR_KCAL)
    q = conformal_quantile(np.abs(yc - yhat_c) / sig_c, alpha)
    out["size_scaled"] = (yhat_t - q * sig_t, yhat_t + q * sig_t)

    edges = np.quantile(yhat_c, np.linspace(0, 1, SIZE_QUARTILES + 1)[1:-1])
    bc, bt = np.digitize(yhat_c, edges), np.digitize(yhat_t, edges)
    qb = np.array([conformal_quantile(np.abs(yc - yhat_c)[bc == b], alpha)
                   for b in range(SIZE_QUARTILES)])
    out["mondrian_size"] = (yhat_t - qb[bt], yhat_t + qb[bt])

    from sklearn.linear_model import QuantileRegressor
    oof, yf = oof_predictions(f, dishes)
    qr = {t: QuantileRegressor(quantile=t, alpha=0.0, solver="highs").fit(oof[:, None], yf)
          for t in (alpha / 2, 1 - alpha / 2)}
    lo_c, hi_c = (qr[t].predict(yhat_c[:, None]) for t in (alpha / 2, 1 - alpha / 2))
    Q = conformal_quantile(np.maximum(lo_c - yc, yc - hi_c), alpha)
    lo_t, hi_t = (qr[t].predict(yhat_t[:, None]) for t in (alpha / 2, 1 - alpha / 2))
    out["cqr"] = (lo_t - Q, hi_t + Q)
    return out


def answered_only(f, dishes, alpha, n_boot, Xc, yc, yhat_c, Xt, yt, yhat_t, session) -> dict:
    """Ranges shown only on the photos a gate answers.

    The split-conformal guarantee is marginal over ALL photos. Shown after a gate, the
    claim a user reads is coverage on the ANSWERED photos, which the marginal guarantee
    does not give. The gate here is ensemble disagreement (the one gate whose
    within-size-band win replicated on all three backbones), refusing the riskiest
    10%, with the threshold set on calibration and then applied unchanged to test (so
    the share actually answered on test is measured, not imposed). The learned error
    head cannot be used: it is trained on the calibration errors, so its calibration
    scores are in-sample.

      all_calib       ranges calibrated on every calibration dish, scored on answered test
      answered_calib  calibrated on the answered calibration dishes only, the gate fixed
                      first; this is the version with a guarantee on answered photos
    """
    g_c = f.scores(Xc)["ensemble_spread"]
    g_t = f.scores(Xt)["ensemble_spread"]
    thr = float(np.quantile(g_c, GATE_ANSWER))
    ac, at = g_c <= thr, g_t <= thr
    err_t = np.abs(yhat_t - yt)
    out = {"gate": "ensemble_spread", "threshold": thr,
           "answered_calib": float(ac.mean()), "answered_test": float(at.mean()),
           "answered_test_ci": cluster_ci(at.astype(float), session, n_boot),
           "mae_answered": float(err_t[at].mean()), "mae_all": float(err_t.mean()),
           "kept_mean_kcal": float(yt[at].mean()), "ranges": {}}
    for w in ("constant", "mondrian_size", "cqr"):
        blk = {}
        for name, mask in (("all_calib", np.ones_like(ac)), ("answered_calib", ac)):
            if w == "constant":
                q = conformal_quantile(np.abs(yc - yhat_c)[mask], alpha)
                lo, hi = yhat_t - q, yhat_t + q
            else:
                lo, hi = size_aware_ranges(f, dishes, alpha, yhat_c[mask], yc[mask],
                                           yhat_t)[w]
            blk[name] = evaluate_ranges(yt[at], lo[at], hi[at], alpha, session[at], n_boot,
                                        err_t[at], yhat_t[at])
        out["ranges"][w] = blk
    return out


def run(model: str, alpha: float, n_boot: int) -> dict:
    dishes = n5k.load_dishes()
    f = fit_gates(model)
    Xc, cids = f.features("calib", "clean")
    Xt, tids = f.features("test", "clean")
    yc = np.array([dishes[d].cal for d in cids], float)
    yt = np.array([dishes[d].cal for d in tids], float)
    sess_t = n5k.session_ids(tids)
    session = np.array([sess_t[d] for d in tids])

    yhat_c, yhat_t = ridge_predict(f.heads["cal"], Xc), ridge_predict(f.heads["cal"], Xt)
    sig_c, sig_t = sigma_of(f, Xc), sigma_of(f, Xt)
    err_t = np.abs(yhat_t - yt)

    out = {"model": model, "alpha": alpha, "n_calib": len(cids), "n_test": len(tids),
           "n_boot": n_boot, "clean": {}, "mixtures": {}}
    # Difficulty quartiles for the honesty check are the ensemble spread's, for every
    # width model, so "does the range cover hard dishes" is asked the same way of each.
    difficulty = sig_t["ensemble_spread"]
    for w in WIDTH_MODELS:
        q = conformal_quantile(np.abs(yc - yhat_c) / sig_c[w], alpha)
        out["clean"][w] = evaluate(yt, yhat_t, sig_t[w], q, alpha, session, n_boot, err_t,
                                   difficulty)
    for w, (lo, hi) in size_aware_ranges(f, dishes, alpha, yhat_c, yc, yhat_t).items():
        out["clean"][w] = evaluate_ranges(yt, lo, hi, alpha, session, n_boot, err_t,
                                          difficulty)
    out["answered_only"] = answered_only(f, dishes, alpha, n_boot, Xc, yc, yhat_c,
                                         Xt, yt, yhat_t, session)

    # deployment mixtures: one damage level per dish, for calibration and test alike
    rung_feats_c, rung_feats_t = {}, {}
    for tag, _, _ in RUNGS:
        try:
            rung_feats_c[tag] = f.features("calib", tag)
            rung_feats_t[tag] = f.features("test", tag)
        except FileNotFoundError:
            continue
    for name, mix in MIXTURES.items():
        if any(t not in rung_feats_c for t in mix):
            out["mixtures"][name] = {"skipped": "calibration features missing for a rung"}
            continue
        rng = np.random.default_rng(SEED + 1)
        pick_c, pick_t = draw_mixture(cids, mix, rng), draw_mixture(tids, mix, rng)

        def assemble(ids, pick, feats):
            rows = []
            for d in ids:
                X, keep = feats[pick[d]]
                rows.append(X[keep.index(d)])
            return np.vstack(rows)

        Xmc, Xmt = assemble(cids, pick_c, rung_feats_c), assemble(tids, pick_t, rung_feats_t)
        yhat_mc, yhat_mt = ridge_predict(f.heads["cal"], Xmc), ridge_predict(f.heads["cal"], Xmt)
        sig_mc, sig_mt = sigma_of(f, Xmc), sigma_of(f, Xmt)
        err_mt = np.abs(yhat_mt - yt)
        blk = {"answer_all_mae": float(err_mt.mean()), "levels": {}}
        for w in ("constant", "ensemble_spread"):
            q_clean = conformal_quantile(np.abs(yc - yhat_c) / sig_c[w], alpha)
            q_mix = conformal_quantile(np.abs(yc - yhat_mc) / sig_mc[w], alpha)
            blk[w] = {
                "calibrated_on_clean": evaluate(yt, yhat_mt, sig_mt[w], q_clean, alpha,
                                                session, n_boot, None,
                                                sig_mt["ensemble_spread"]),
                "calibrated_on_mixture": evaluate(yt, yhat_mt, sig_mt[w], q_mix, alpha,
                                                  session, n_boot, err_mt,
                                                  sig_mt["ensemble_spread"])}
        out["mixtures"][name] = blk
    return out


def report(res: dict) -> None:
    a = res["alpha"]
    print(f"model {res['model']} | alpha {a} (target coverage {100 * (1 - a):.0f}%) | "
          f"calib {res['n_calib']} | test {res['n_test']}\n")
    print("=== CLEAN TEST PHOTOS")
    print(f"{'width model':<17}{'coverage':>19}{'mean w':>8}{'median':>8}{'usable':>8}"
          f"{'honesty gap':>13}{'int. score':>12}{'  cov. by difficulty q':<24}"
          f"{'cov. by kcal q':<18}{'width by kcal q':<18}")
    for w, r in res["clean"].items():
        lo, hi = r["coverage_ci"]
        qd = " ".join(f"{100 * c:3.0f}" for c in r["by_difficulty"]["coverage"])
        qk = " ".join(f"{100 * c:3.0f}" for c in r["by_kcal"]["coverage"])
        wk = " ".join(f"{c:3.0f}" for c in r["by_kcal"]["width"])
        print(f"{w:<17}{100 * r['coverage']:6.1f}% [{100 * lo:4.1f},{100 * hi:5.1f}]"
              f"{r['width_mean']:8.0f}{r['width_median']:8.0f}{100 * r['usable_share']:7.0f}%"
              f"{100 * r['honesty_gap']:12.1f}pt{r['interval_score']:12.0f}   {qd:<21}"
              f"  {qk:<16}  {wk}")
    print("\n--- the range as a gate: answer only when the range is narrower than W")
    print(f"{'width model':<17}{'W':>5}{'answered':>10}{'coverage':>10}{'MAE':>7}"
          f"{'kept kcal':>11}{'big refused':>13}")
    for w, r in res["clean"].items():
        for W, g in r.get("as_gate", {}).items():
            print(f"{w:<17}{W:>5}{100 * g['answered']:9.0f}%{100 * g['coverage']:9.1f}%"
                  f"{g['mae']:7.1f}{g['kept_mean_kcal']:11.0f}{100 * g['big_refused']:12.0f}%")
    a = res.get("answered_only")
    if a:
        lo, hi = a["answered_test_ci"]
        print(f"\n=== ANSWERED PHOTOS ONLY: gate {a['gate']}, threshold set on calibration "
              f"to answer {100 * a['answered_calib']:.0f}%; answers "
              f"{100 * a['answered_test']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] of test; "
              f"MAE {a['mae_answered']:.1f} vs {a['mae_all']:.1f} answering all")
        print(f"{'width model':<17}{'calibrated on':<16}{'coverage':>19}{'mean w':>8}"
              f"{'  cov. by kcal q':<18}")
        for w, blk in a["ranges"].items():
            for on, r in blk.items():
                lo, hi = r["coverage_ci"]
                qk = " ".join(f"{100 * c:3.0f}" for c in r["by_kcal"]["coverage"])
                print(f"{w:<17}{on:<16}{100 * r['coverage']:6.1f}% [{100 * lo:4.1f},"
                      f"{100 * hi:5.1f}]{r['width_mean']:8.0f}   {qk}")
    if any("levels" in v for v in res["mixtures"].values()):
        print("\n=== DEPLOYMENT MIXTURES, one damage level per dish")
        print(f"{'mixture':<14}{'width model':<17}{'calibrated on':<15}{'coverage':>19}"
              f"{'mean w':>8}{'usable':>8}")
        for name, blk in res["mixtures"].items():
            if "levels" not in blk:
                print(f"{name:<14}{blk['skipped']}")
                continue
            for w in ("constant", "ensemble_spread"):
                for on in ("clean", "mixture"):
                    r = blk[w][f"calibrated_on_{on}"]
                    lo, hi = r["coverage_ci"]
                    print(f"{name:<14}{w:<17}{on:<15}{100 * r['coverage']:6.1f}% "
                          f"[{100 * lo:4.1f},{100 * hi:5.1f}]{r['width_mean']:8.0f}"
                          f"{100 * r['usable_share']:7.0f}%")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true", help="write results/conformal.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    models = BACKBONES if args.all else [args.model]
    results = {}
    for m in models:
        results[m] = run(m, args.alpha, args.boot)
        report(results[m])
    if args.json:
        with open(RESULTS / "conformal.json", "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"wrote {RESULTS / 'conformal.json'}")


if __name__ == "__main__":
    main()
