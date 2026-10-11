"""Replicate the gate benchmark on the 367 train sessions no gate was ever scored on.

  ../.venv/bin/python replicate.py [--all] [--model NAME] [--json] [--boot 2000] [--folds 5]

WHY. Feedback on the update talk asked whether the results are inconclusive: with
p near 0.05, more data could tip them either way. The test split has 127 plate
sessions, and every analysis so far was chosen while looking at it. This script runs
the same analysis on data that played no part in any of those choices: the official
TRAIN split (2,755 dishes, 367 sessions). Until now train was only used to fit heads,
never to score a gate.

HOW (cross-fitting by plate session). The train sessions are dealt into K folds. For
each fold, everything is refit from scratch on the other K-1 folds exactly as
gate_probe.fit_gates does it on the official split: that pool is carved by session
into fit and calibration (30%), the standardiser and calorie head are fit on fit,
alpha is picked on calibration, the error head is trained on calibration errors, and
the size-band edges come from calibration predictions. The held-out fold is then
scored. Every train dish is scored exactly once, by models that never saw its
session. The official test split is not touched.

Each fold's models score their own fold, so a threshold must not compare scores from
different fold models. Every gate therefore refuses the same share inside each fold
(clean) or inside each fold x size band (clean_sized). The random reference is the
same analytic flat line as in gate_probe.

HYPOTHESES, fixed from the test-split results before this script was first run
(9 Oct 2026). All at 90% answered, kcal loss, session bootstrap, per backbone:
  H1  inside size bands, the learned error head beats refusing at random
  H2  inside size bands, the learned error head beats the predicted-size control
  H3  without size bands, the predicted-size control beats random (the size confound
      is real), and inside size bands it does not (the design removes it)
  H4  inside size bands, no pixel gate's benefit can be as large as a quarter of what
      a perfect refuser saves: the 95% interval's best end stays short of that
  H5  (secondary) inside size bands, ensemble disagreement beats random
Pixel metrics for the train photos are measured here (clean photos only) and cached
in results/train_pixel.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict

import numpy as np
from PIL import Image

import n5k
from gate_bench import (MODEL, PIXEL_GATES, RESULTS, pick_alpha, ridge_fit, ridge_predict,
                        rows_for, standardiser)
from gate_probe import (BACKBONES, MODEL_GATES, PIXEL_SIGN, SEED, SIZE_BANDS, curve,
                        curve_banded, ensemble_scorer, evaluate_block, knn_scorer,
                        load_cache_for, mahalanobis_scorer)
from quality import METRICS

K_FOLDS = 5
EQUIV_SHARE = 0.25          # H4: a quarter of the perfect refuser's saving
N_CARVES = 20               # calibration carves for the training-variance check
OP = "0.9"


# ---------------------------------------------------------------- data
def train_pixel_metrics(ids: list[str]) -> dict[str, dict[str, float]]:
    """The four pixel metrics on the clean train photos, cached after the first run."""
    path = RESULTS / "train_pixel.csv"
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            out = {r["dish_id"]: {m: float(r[m]) for m in PIXEL_GATES}
                   for r in csv.DictReader(fh)}
    todo = [d for d in ids if d not in out]
    for i, d in enumerate(todo):
        img = Image.open(n5k.IMAGES / d / "rgb.png").convert("RGB")
        out[d] = {m: METRICS[m](img) for m in PIXEL_GATES}
        if i % 250 == 0:
            print(f"  pixel metrics {i}/{len(todo)}", end="\r")
    if todo:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("dish_id," + ",".join(PIXEL_GATES) + "\n")
            for d in sorted(out):
                fh.write(d + "," + ",".join(f"{out[d][m]:.6g}" for m in PIXEL_GATES) + "\n")
        print(f"  wrote {path} ({len(out)} photos)      ")
    return out


def session_folds(ids: list[str], k: int, seed: int = SEED) -> dict[str, int]:
    """Deal plate sessions into k folds of about equal dish count."""
    sess = n5k.session_ids(ids)
    groups: dict[int, list[str]] = defaultdict(list)
    for d in ids:
        groups[sess[d]].append(d)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    load, fold_of = [0] * k, {}
    for g in sorted(order, key=lambda g: -len(groups[g])):     # largest sessions first
        f = int(np.argmin(load))
        load[f] += len(groups[g])
        for d in groups[g]:
            fold_of[d] = f
    return fold_of


def carve_calib(ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Split a pool into fit and calibration by session, as n5k.fit_calib_split does."""
    sess = n5k.session_ids(ids)
    groups: dict[int, list[str]] = defaultdict(list)
    for d in ids:
        groups[sess[d]].append(d)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target, calib = n5k.CALIB_FRACTION * len(ids), []
    for g in order:
        if len(calib) >= target:
            break
        calib.extend(groups[g])
    cs = set(calib)
    return sorted(d for d in ids if d not in cs), sorted(calib)


# ---------------------------------------------------------------- one fold
def fit_and_score(Xall, ids_all, pool, held, dishes, seed: int = SEED):
    """Refit every head and gate on `pool`; return scores for `held`.

    `seed` picks which sessions of the pool become calibration.
    """
    fit_ids, cal_ids = carve_calib(pool, seed)
    z = standardiser(rows_for(Xall, ids_all, fit_ids).astype(np.float64))
    Xf = z(rows_for(Xall, ids_all, fit_ids).astype(np.float64))
    Xc = z(rows_for(Xall, ids_all, cal_ids).astype(np.float64))
    Xh = z(rows_for(Xall, ids_all, held).astype(np.float64))
    yf = np.array([dishes[d].cal for d in fit_ids], float)
    yc = np.array([dishes[d].cal for d in cal_ids], float)

    a, _ = pick_alpha(Xf, yf, Xc, yc)
    head = ridge_fit(Xf, yf, a)
    cal_pred = ridge_predict(head, Xc)
    cal_err = np.abs(cal_pred - yc)
    scorers = {"mahalanobis": mahalanobis_scorer(Xf), "knn_dist": knn_scorer(Xf),
               "ensemble_spread": ensemble_scorer(Xf, yf, a)}
    err_head = ridge_fit(Xc, cal_err, a)
    scorers["learned_error"] = lambda X: ridge_predict(err_head, X)
    four = ("mahalanobis", "knn_dist", "ensemble_spread", "learned_error")
    blend = ridge_fit(np.column_stack([scorers[k](Xc) for k in four]), cal_err, 1.0)
    edges = np.quantile(cal_pred, np.linspace(0, 1, SIZE_BANDS + 1)[1:-1])

    s = {k: f(Xh) for k, f in scorers.items()}
    s["blend"] = ridge_predict(blend, np.column_stack([s[k] for k in four]))
    pred = ridge_predict(head, Xh)
    s["pred_magnitude"] = pred
    return dict(scores=s, pred=pred, band=np.searchsorted(edges, pred, side="right"),
                alpha=a, n_fit=len(fit_ids), n_calib=len(cal_ids))


# ---------------------------------------------------------------- hypotheses
def verdicts(clean: dict, sized: dict) -> dict:
    """Read H1-H5 off the two blocks. p-values are the Holm-corrected paired tests."""
    g, gs = clean["gates"], sized["gates"]
    gap = gs["random"]["mae"][OP] - gs["perfect"]["mae"][OP]     # kcal a perfect refuser saves
    v = {}
    h1 = gs["learned_error"]["vs_random"][OP]
    v["H1"] = dict(diff=h1["diff"], ci=h1["ci"], p_holm=h1["p_holm"],
                   holds=h1["diff"] < 0 and h1["ci"][1] < 0)
    h2 = sized["pairwise"][OP]["learned_error - pred_magnitude"]
    v["H2"] = dict(diff=h2["diff"], ci=h2["ci"], p=h2["p"], holds=h2["ci"][1] < 0)
    a, b = g["pred_magnitude"]["vs_random"][OP], gs["pred_magnitude"]["vs_random"][OP]
    v["H3"] = dict(plain=dict(diff=a["diff"], ci=a["ci"], p_holm=a["p_holm"]),
                   sized=dict(diff=b["diff"], ci=b["ci"], p_holm=b["p_holm"]),
                   holds=a["ci"][1] < 0 and b["ci"][0] <= 0 <= b["ci"][1])
    margin = EQUIV_SHARE * gap
    v["H4"] = {"margin_kcal": margin, "perfect_saving_kcal": gap, "gates": {}}
    for p in PIXEL_GATES:
        if p in gs:
            r = gs[p]["vs_random"][OP]
            v["H4"]["gates"][p] = dict(diff=r["diff"], ci=r["ci"],
                                       best_case_share=-r["ci"][0] / gap,
                                       holds=r["ci"][0] > -margin)
    v["H4"]["holds"] = all(x["holds"] for x in v["H4"]["gates"].values())
    h5 = gs["ensemble_spread"]["vs_random"][OP]
    v["H5"] = dict(diff=h5["diff"], ci=h5["ci"], p_holm=h5["p_holm"],
                   holds=h5["ci"][1] < 0)
    v["share_of_perfect"] = {k: gs[k]["vs_random"][OP]["diff"] / -gap
                             for k in MODEL_GATES + PIXEL_GATES if k in gs}
    return v


# ---------------------------------------------------------------- one backbone
def replicate(model: str, n_boot: int, k: int, pixel: dict) -> dict:
    dishes = n5k.load_dishes()
    Xall, ids_all = load_cache_for(model, "train", "clean")
    known = set(ids_all)
    ids = [d for d in n5k.usable_split("train", local_only=True) if d in known]
    fold_of = session_folds(ids, k)
    sess = n5k.session_ids(ids)

    parts, folds_meta = [], []
    for f in range(k):
        held = sorted(d for d in ids if fold_of[d] == f)
        pool = sorted(d for d in ids if fold_of[d] != f)
        r = fit_and_score(Xall, ids_all, pool, held, dishes)
        y = np.array([dishes[d].cal for d in held], float)
        parts.append(dict(ids=held, y=y, err=np.abs(r["pred"] - y), scores=r["scores"],
                          band=r["band"], fold=np.full(len(held), f)))
        folds_meta.append(dict(n_held=len(held), n_fit=r["n_fit"], n_calib=r["n_calib"],
                               alpha=r["alpha"], mae=float(np.abs(r["pred"] - y).mean())))
        print(f"  fold {f}: held {len(held)} | fit {r['n_fit']} | calib {r['n_calib']} "
              f"| alpha {r['alpha']:.0f} | MAE {folds_meta[-1]['mae']:.1f}")

    cat = lambda key: np.concatenate([p[key] for p in parts])
    ids_cat = [d for p in parts for d in p["ids"]]
    err, true, fold, band = cat("err"), cat("y"), cat("fold"), cat("band")
    session = np.array([sess[d] for d in ids_cat])
    gates = {g: np.concatenate([p["scores"][g] for p in parts]) for g in MODEL_GATES}
    for g in PIXEL_GATES:
        gates[g] = PIXEL_SIGN[g] * np.array([pixel[d][g] for d in ids_cat])
    size = gates["pred_magnitude"]

    out = {"model": model, "folds": folds_meta, "n": len(ids_cat),
           "n_sessions": int(len(np.unique(session))),
           "predictor_mae": float(err.mean())}
    out["clean"] = evaluate_block(gates, err, true, session, n_boot, band=fold, size=size)
    out["clean_sized"] = evaluate_block(gates, err, true, session, n_boot,
                                        band=fold * SIZE_BANDS + band, size=size)
    out["verdicts"] = verdicts(out["clean"], out["clean_sized"])
    return out


# ---------------------------------------------------------------- training variance
CARVE_GATES = ("learned_error", "ensemble_spread", "pred_magnitude")


def carve_spread(model: str, n_carves: int, n_boot: int) -> dict:
    """How much does the test-split result depend on which sessions became calibration?

    gate_probe fits once, with the calibration carve fixed by seed 760, and its session
    bootstrap resamples the TEST split only: the fitted heads are held fixed. That
    interval leaves out a second source of variance, the training carve. Here the full
    train split is re-carved with `n_carves` seeds (760 first, then 1, 2, ...), every
    head and gate is refit, and the official test split is scored each time.

    Reported: each carve's effect at 90% answered (inside size bands and without), the
    official carve's rank among them, and the carve-averaged effect with a session
    bootstrap that recomputes every carve's curve on the same resample.
    """
    dishes = n5k.load_dishes()
    Xtr, ids_tr = load_cache_for(model, "train", "clean")
    Xte, ids_te = load_cache_for(model, "test", "clean")
    X, ids = np.vstack([Xtr, Xte]), ids_tr + ids_te
    train = [d for d in n5k.usable_split("train", local_only=True) if d in set(ids_tr)]
    test = [d for d in n5k.usable_split("test", local_only=True) if d in set(ids_te)]
    y = np.array([dishes[d].cal for d in test], float)
    sess = n5k.session_ids(test)
    session = np.array([sess[d] for d in test])
    c90 = [float(OP)]

    seeds = [SEED] + [s for s in range(1, n_carves + 5) if s != SEED][:n_carves - 1]
    fits = []
    for s in seeds:
        r = fit_and_score(X, ids, train, test, dishes, seed=s)
        fits.append(dict(err=np.abs(r["pred"] - y), scores=r["scores"], band=r["band"]))

    def effects(idx=None):
        """Per-carve change vs random at 90%, banded and plain, on rows `idx`."""
        out = {f"{g}_{k}": [] for g in CARVE_GATES + ("perfect",) for k in ("sized", "plain")}
        out["mae"] = []
        for f in fits:
            e = f["err"] if idx is None else f["err"][idx]
            b = f["band"] if idx is None else f["band"][idx]
            out["mae"].append(e.mean())
            for g in CARVE_GATES + ("perfect",):
                s = e if g == "perfect" else (f["scores"][g] if idx is None else f["scores"][g][idx])
                out[f"{g}_sized"].append(curve_banded(s, e, b, c90)[0] - e.mean())
                out[f"{g}_plain"].append(curve(s, e, c90)[0] - e.mean())
        return {k: np.array(v) for k, v in out.items()}

    point = effects()
    rng = np.random.default_rng(SEED)
    sids = np.unique(session)
    members = {s: np.flatnonzero(session == s) for s in sids}
    boot = {k: np.empty(n_boot) for k in point}
    boot["head_minus_control_sized"] = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[s] for s in rng.choice(sids, len(sids), replace=True)])
        e = effects(idx)
        for k in point:
            boot[k][b] = e[k].mean()
        boot["head_minus_control_sized"][b] = (e["learned_error_sized"]
                                               - e["pred_magnitude_sized"]).mean()

    def summary(k, pt):
        d = boot[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / n_boot)))
        return dict(mean=float(pt), ci=[float(lo), float(hi)], p=p)

    res = {"seeds": seeds, "n_test": len(test), "n_sessions": int(len(sids)),
           "per_carve": {k: [float(x) for x in v] for k, v in point.items()},
           "official": {k: float(v[0]) for k, v in point.items()},
           "official_rank": {k: int((v < v[0]).sum()) + 1 for k, v in point.items()},
           "spread_sd": {k: float(v.std(ddof=1)) for k, v in point.items()},
           "averaged": {k: summary(k, v.mean()) for k, v in point.items()}}
    res["averaged"]["head_minus_control_sized"] = summary(
        "head_minus_control_sized",
        (point["learned_error_sized"] - point["pred_magnitude_sized"]).mean())
    return res


def report_carves(model: str, c: dict) -> None:
    print(f"\n{model}: {len(c['seeds'])} calibration carves, official test split "
          f"({c['n_test']} dishes, {c['n_sessions']} sessions), 90% answered")
    for k in ("mae", "learned_error_sized", "ensemble_spread_sized", "pred_magnitude_sized",
              "perfect_sized", "learned_error_plain", "pred_magnitude_plain"):
        a, v = c["averaged"][k], np.array(c["per_carve"][k])
        print(f"  {k:<24} official {c['official'][k]:+6.1f} (rank {c['official_rank'][k]:>2}) | "
              f"carves {v.min():+6.1f} .. {v.max():+6.1f}, sd {c['spread_sd'][k]:.1f} | "
              f"averaged {a['mean']:+6.1f} [{a['ci'][0]:+5.1f}, {a['ci'][1]:+5.1f}] p {a['p']:.3f}")
    a = c["averaged"]["head_minus_control_sized"]
    print(f"  head - control, in bands  averaged {a['mean']:+6.1f} "
          f"[{a['ci'][0]:+5.1f}, {a['ci'][1]:+5.1f}] p {a['p']:.3f}")


def report(res: dict) -> None:
    v = res["verdicts"]
    lo, hi = res["clean"]["answer_all_ci"]
    print(f"\n{res['model']}: {res['n']} dishes, {res['n_sessions']} sessions, "
          f"answer everything {res['predictor_mae']:.1f} [{lo:.1f}, {hi:.1f}] kcal")
    fmt = lambda d: f"{d['diff']:+6.1f} [{d['ci'][0]:+5.1f}, {d['ci'][1]:+5.1f}]"
    print(f"  H1 head vs random, in bands     {fmt(v['H1'])}  Holm p {v['H1']['p_holm']:.3f}"
          f"  {'HOLDS' if v['H1']['holds'] else 'fails'}")
    print(f"  H2 head vs size control, bands  {fmt(v['H2'])}  p {v['H2']['p']:.3f}"
          f"  {'HOLDS' if v['H2']['holds'] else 'fails'}")
    print(f"  H3 control vs random, plain     {fmt(v['H3']['plain'])}; in bands "
          f"{fmt(v['H3']['sized'])}  {'HOLDS' if v['H3']['holds'] else 'fails'}")
    print(f"  H4 pixel gates, margin {v['H4']['margin_kcal']:.1f} kcal "
          f"(a quarter of {v['H4']['perfect_saving_kcal']:.1f}):  "
          f"{'HOLDS' if v['H4']['holds'] else 'fails'}")
    for p, r in v["H4"]["gates"].items():
        print(f"     {p:<11}{fmt(r)}  best case {100 * r['best_case_share']:.0f}% of perfect")
    print(f"  H5 ensemble vs random, bands    {fmt(v['H5'])}  Holm p {v['H5']['p_holm']:.3f}"
          f"  {'HOLDS' if v['H5']['holds'] else 'fails'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true", help="write results/replication.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=K_FOLDS)
    ap.add_argument("--carves", type=int, default=N_CARVES,
                    help="calibration carves for the training-variance check (0 = skip)")
    ap.add_argument("--no-replication", action="store_true",
                    help="run only the training-variance check")
    args = ap.parse_args()

    pixel = train_pixel_metrics(n5k.usable_split("train", local_only=True))
    results = {}
    for m in (BACKBONES if args.all else [args.model]):
        print(f"\n=== {m}")
        results[m] = {"model": m}
        if not args.no_replication:
            results[m] = replicate(m, args.boot, args.folds, pixel)
            report(results[m])
        if args.carves:
            results[m]["carves"] = carve_spread(m, args.carves, args.boot)
            report_carves(m, results[m]["carves"])
    if args.json:
        with open(RESULTS / "replication.json", "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {RESULTS / 'replication.json'}")


if __name__ == "__main__":
    main()
