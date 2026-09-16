"""Where does the error come from, and which part of it is gateable?

  ../.venv/bin/python error_sources.py      (reads results/predictions.csv)

gate_bench.py answers "does gate X work?" on the pooled damage ladder. That pooling
hides the question that decides what we actually ship: a photo can be wrong because
it is a BAD PHOTO, or because it is a HARD DISH photographed perfectly. Only the
first is simulated by the ladder, and only the second survives into a deployment
where users mostly take ordinary photos.

This script separates them. Three findings, all reproducible from the CSV:

  1. Detection is not the bottleneck. Cheap pixel metrics separate degraded photos
     from clean ones at AUC ~= 1.00 — near-perfectly. Detecting damage is solved and
     it still does not help, because detecting damage is not the same as ranking harm.
  2. Damage level explains ~7% of the variance in |calorie error|; which dish it is
     explains ~56%. The photo-quality framing is aimed at the small half.
  3. On clean photos alone — no damage anywhere — a perfect gate still takes 73.4
     kcal down to 25.6 at 50% coverage. That headroom is dish difficulty, and it is
     larger than everything the whole degradation ladder moves.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
RUNGS = ["clean", "blur3", "blur6", "crop0p4", "jpeg10", "downscale8", "noise0p16", "phone"]
PIXEL = ["lap_var", "tenengrad", "blockiness", "hf_ratio"]
NUMERIC = PIXEL + ["cal_abs_err", "mass_abs_err", "mahalanobis", "cal_true"]
SEED = 760


def load():
    rows = list(csv.DictReader(open(RESULTS / "predictions.csv")))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k])
    return rows


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random pos scores above random neg), via ranks. 0.5 = cannot tell them apart."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    rank = np.empty(len(allv))
    rank[order] = np.arange(1, len(allv) + 1)
    return (rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def reject_curve(score, err, fracs=(1.0, 0.5, 0.25)):
    """Keep the frac the gate is most confident about (lowest score); report their MAE."""
    order = np.asarray(score, float).argsort()
    e = np.asarray(err, float)
    return {f: float(e[order[: max(1, int(round(f * len(e))))]].mean()) for f in fracs}


def gates_for(subset, rng):
    e = np.array([r["cal_abs_err"] for r in subset])
    return e, {
        "sharpness": -np.array([r["lap_var"] for r in subset]),   # high sharpness = low risk
        "mahalanobis": np.array([r["mahalanobis"] for r in subset]),
        "random": rng.random(len(subset)),
        "perfect": e,
    }


def variance_decomposition(rows) -> dict[str, float]:
    """Two-way additive split of |error| into dish effect, damage effect, remainder."""
    err = np.array([r["cal_abs_err"] for r in rows])
    grand = err.mean()
    by_dish, by_rung = defaultdict(list), defaultdict(list)
    for r in rows:
        by_dish[r["dish_id"]].append(r["cal_abs_err"])
        by_rung[r["rung"]].append(r["cal_abs_err"])
    dish_eff = {d: np.mean(v) - grand for d, v in by_dish.items()}
    rung_eff = {g: np.mean(v) - grand for g, v in by_rung.items()}
    resid = [r["cal_abs_err"] - grand - dish_eff[r["dish_id"]] - rung_eff[r["rung"]]
             for r in rows]
    v = {"dish": np.var([dish_eff[r["dish_id"]] for r in rows]),
         "damage": np.var([rung_eff[r["rung"]] for r in rows]),
         "interaction": np.var(resid)}
    total = sum(v.values())
    return {k: 100 * x / total for k, x in v.items()}


def main() -> None:
    rows = load()
    rng = np.random.default_rng(SEED)
    clean_by_dish = {r["dish_id"]: r for r in rows if r["rung"] == "clean"}
    clean = list(clean_by_dish.values())

    share = variance_decomposition(rows)
    print("=== 1. where does |calorie error| variance live? "
          f"({len(clean)} dishes x {len(RUNGS)} damage levels)")
    print(f"  which DISH it is           {share['dish']:5.1f}%")
    print(f"  which DAMAGE LEVEL it is   {share['damage']:5.1f}%")
    print(f"  dish x damage interaction  {share['interaction']:5.1f}%")
    print("  -> the photograph is the small half. Even counting the interaction")
    print("     generously, most of the error is decided by which dish it is.")

    print("\n=== 2. can a cheap metric even SEE the damage? (AUC vs the same dish clean)")
    print(f"{'damage level':<14}" + "".join(f"{m:>13}" for m in PIXEL + ["mahalanobis"]))
    for g in RUNGS[1:]:
        sub = [r for r in rows if r["rung"] == g]
        line = f"{g:<14}"
        for m in PIXEL + ["mahalanobis"]:
            a = auc(np.array([r[m] for r in sub]),
                    np.array([clean_by_dish[r["dish_id"]][m] for r in sub]))
            line += f"{max(a, 1 - a):>13.2f}"
        print(line)
    print("  -> 1.00 means perfect separation. Detection is SOLVED and it does not help:")
    print("     the metrics see the damage, they just cannot rank how much it costs.")

    print("\n=== 3. reject curves, computed three ways")
    for label, subset in [
        ("A. POOLED over all damage levels — what the deck reports", rows),
        ("B. CLEAN PHOTOS ONLY — pure dish difficulty, no damage at all", clean),
    ]:
        e, gates = gates_for(subset, rng)
        print(f"\n{label}  (n={len(subset)}, mean {e.mean():.1f} kcal)")
        print(f"{'keep':<8}" + "".join(f"{k:>14}" for k in gates))
        for f in (1.0, 0.5, 0.25):
            print(f"{int(f * 100):>3}%{'':<4}"
                  + "".join(f"{reject_curve(v, e)[f]:>14.1f}" for v in gates.values()))

    print("\nC. WITHIN each damage level separately — error at 50% coverage")
    print(f"{'damage level':<14}" + "".join(f"{k:>14}" for k in
                                            ("sharpness", "mahalanobis", "random", "perfect")))
    for g in RUNGS:
        e, gates = gates_for([r for r in rows if r["rung"] == g], rng)
        print(f"{g:<14}" + "".join(f"{reject_curve(v, e)[0.5]:>14.1f}" for v in gates.values()))

    e_clean = np.array([r["cal_abs_err"] for r in clean])
    e_pool = np.array([r["cal_abs_err"] for r in rows])
    perfect_clean = reject_curve(e_clean, e_clean)[0.5]
    print(f"\n=== 4. the prize, in one comparison")
    print(f"  everything the whole damage ladder does :  {e_clean.mean():.1f} -> "
          f"{e_pool.mean():.1f} kcal  ({e_pool.mean() - e_clean.mean():+.1f})")
    print(f"  what a perfect gate buys on CLEAN photos:  {e_clean.mean():.1f} -> "
          f"{perfect_clean:.1f} kcal  ({perfect_clean - e_clean.mean():+.1f})")
    print("  -> dish difficulty offers more gateable range than all the damage combined.")


if __name__ == "__main__":
    main()
