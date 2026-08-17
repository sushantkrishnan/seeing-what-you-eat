"""The oracle ladder — how much is each clue worth?

Refactored from the root-level oracle_ladder.py to go through n5k.py, and extended
to answer the question the original could not: does the headline result survive when
the test set is restricted to the 507 dishes that actually have overhead imagery?

  python code/oracle_ladder.py

Each rung hands the calculator one more piece of ground truth and reports the error
that remains. It is a ceiling analysis, not a model — no image is ever read.
"""
from __future__ import annotations

import random
import statistics as st
from collections import defaultdict

import n5k

PUBLISHED = [
    ("2D direct prediction, RGB only", 70.6, 26.1),
    ("RGB-D (depth as 4th channel)", 47.6, 18.8),
    ("volume scalar (best reported)", 41.3, 16.5),
]


def learn_priors(train: list[n5k.Dish]):
    """Median kcal/g and median grams per ingredient, learned from TRAIN only."""
    dens, gram = defaultdict(list), defaultdict(list)
    for d in train:
        for name, grams, kcal in d.ingredients:
            if grams > 0:
                dens[name].append(kcal / grams)
            gram[name].append(grams)
    return (
        {k: st.median(v) for k, v in dens.items()},
        {k: st.median(v) for k, v in gram.items()},
        st.median([d.cal / d.mass for d in train]),   # global kcal/g
        st.mean([d.cal for d in train]),              # global mean kcal
    )


def build_rungs(dens, gram, gdens, gcal):
    def o3(d: n5k.Dish) -> float:
        """Identity + total mass: share the true total mass out by typical proportions."""
        total = sum(gram.get(n, 0) for n in d.names)
        if total <= 0:
            return gcal
        return sum((gram.get(n, 0) / total) * d.mass * dens.get(n, 0) for n in d.names)

    return [
        ("O0  nothing (predict train mean kcal)", lambda d: gcal),
        ("O1  total MASS only", lambda d: d.mass * gdens),
        ("O2  ingredient IDENTITY only", lambda d: sum(gram.get(n, 0) * dens.get(n, 0) for n in d.names)),
        ("O3  IDENTITY + total MASS", o3),
        ("O4  IDENTITY + exact GRAMS (lookup floor)",
         lambda d: sum(g * dens.get(n, 0) for n, g, _ in d.ingredients)),
    ]


def perturbed_o3(dens, gram, gcal, test, mass_err, ident_err, seed=760, reps=20):
    """O3 (identity + total mass) when the two stages are WRONG by a stated amount.

    The clean O3 rung is an error budget, not a headroom claim: it assumes a perfect
    namer and a perfect scale. This asks the deployable question instead — if the mass
    head is off by `mass_err` and a fraction `ident_err` of ingredients are misnamed,
    where does the two-stage route actually land? Averaged over `reps` draws.

    `mass_err` is a MEAN relative error, so it is directly comparable to the 22.2%
    our own mass head measures: drawing the relative error uniformly on ±2·mass_err
    makes the mean absolute relative error exactly mass_err.
    """
    pool = sorted(gram)
    rng = random.Random(seed)
    totals = []
    for _ in range(reps):
        errs = []
        for d in test:
            mass = d.mass * (1.0 + rng.uniform(-2 * mass_err, 2 * mass_err))
            names = [rng.choice(pool) if rng.random() < ident_err else n for n in d.names]
            tot = sum(gram.get(n, 0) for n in names)
            pred = (gcal if tot <= 0 else
                    sum((gram.get(n, 0) / tot) * mass * dens.get(n, 0) for n in names))
            errs.append(abs(pred - d.cal))
        totals.append(sum(errs) / len(errs))
    return sum(totals) / len(totals)


def evaluate(rungs, test: list[n5k.Dish]) -> list[tuple[str, float, float]]:
    rows = []
    for name, fn in rungs:
        errs = [abs(fn(d) - d.cal) for d in test]
        pcts = [abs(fn(d) - d.cal) / d.cal for d in test if d.cal > 1]
        rows.append((name, sum(errs) / len(errs), 100 * sum(pcts) / len(pcts)))
    return rows


def main() -> None:
    dishes = n5k.load_dishes()
    train = [dishes[i] for i in n5k.load_split("train") if i in dishes]
    have = n5k.available_image_ids()

    test_all = [dishes[i] for i in n5k.load_split("test") if i in dishes]
    test_img = [d for d in test_all if d.dish_id in have]

    priors = learn_priors(train)
    rungs = build_rungs(*priors)
    rows_all = evaluate(rungs, test_all)
    rows_img = evaluate(rungs, test_img)

    print(f"priors learned from {len(train)} train dishes\n")
    print(f"{'condition':<44}{'ALL n=' + str(len(test_all)):>16}{'IMAGERY n=' + str(len(test_img)):>18}")
    print(f"{'':<44}{'MAE':>9}{'MAPE':>7}{'MAE':>11}{'MAPE':>7}")
    print("-" * 78)
    for (name, m_a, p_a), (_, m_i, p_i) in zip(rows_all, rows_img):
        print(f"{name:<44}{m_a:>9.1f}{p_a:>6.0f}%{m_i:>11.1f}{p_i:>6.0f}%")
    print("-" * 78)
    delta = rows_img[3][1] - rows_all[3][1]
    print(f"O3 shift from restricting to dishes with imagery: {delta:+.1f} kcal\n")
    print("published baselines on this dataset (Thames et al., CVPR 2021):")
    for label, mae, mape in PUBLISHED:
        print(f"  {label:<36} {mae:>5.1f} kcal MAE / {mape:.1f}% MAPE")
    print(f"\nO3 ({rows_img[3][1]:.1f}) vs best published ({PUBLISHED[-1][1]}): "
          f"{'still beats it' if rows_img[3][1] < PUBLISHED[-1][1] else 'NO LONGER BEATS IT'}")

    dens, gram, _, gcal = priors
    print("\nO3 with REALISTIC stage error (n=%d, 20 draws) — this is the honest column."
          % len(test_img))
    print("Our own mass head measures 22.2%% relative error, so the 20%% row is the")
    print("one that describes the pipeline we would actually build.\n")
    print(f"{'mass error':<14}{'ingredients misnamed':<24}{'kcal MAE':>10}")
    print("-" * 48)
    for mass_err, ident_err in [(0.0, 0.0), (0.10, 0.0), (0.0, 0.20),
                                (0.10, 0.20), (0.20, 0.20), (0.20, 0.30)]:
        mae = perturbed_o3(dens, gram, gcal, test_img, mass_err, ident_err)
        star = "  <- our measured mass error" if mass_err == 0.20 and ident_err == 0.20 else ""
        print(f"{mass_err:<14.0%}{ident_err:<24.0%}{mae:>10.1f}{star}")


if __name__ == "__main__":
    main()
