"""Nutrition5k data access — the single source of truth for the project.

Everything downstream (features, degradation, conformal) imports from here so that
"which dishes are we using" is defined in exactly one place.

Gotchas this module hides (see CLAUDE.md):
  * dish_metadata_cafe{1,2}.csv are ragged and headerless; pd.read_csv chokes.
  * The file is ingredients_metadata.csv (plural).
  * The official split lists 4,059/709 dish ids, but only 3,490 dishes actually have
    overhead RGB imagery -> the usable set is smaller. Always go through usable_split().
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "n5k_metadata"
DATA = ROOT / "data"
IMAGES = DATA / "images" / "overhead"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"

GCS = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"

# Train is sub-split into fit/calibration with this seed. Conformal needs a
# calibration set that the heads never saw; we carve it out of TRAIN so the
# official test split stays untouched.
CALIB_SEED = 760
CALIB_FRACTION = 0.30

# Dish ids are unix timestamps. Scans of one plate session (single ingredients, then
# the combined plate) arrive ~40s apart; sessions are separated by hours or days, so
# the gap distribution is bimodal and a 10-minute threshold splits it cleanly.
# |calorie error| has an intraclass correlation of ~0.34 within a session, so sessions
# are the unit for resampling and for the fit/calibration carve. Splitting them at
# dish level put 98% of calibration dishes in a session with a fit dish.
SESSION_GAP_S = 600


@dataclass
class Dish:
    dish_id: str
    cal: float
    mass: float
    fat: float
    carb: float
    protein: float
    ingredients: list[tuple[str, float, float]] = field(default_factory=list)  # (name, grams, kcal)

    @property
    def names(self) -> list[str]:
        return [n for n, _, _ in self.ingredients]

    def rgb_path(self) -> Path:
        return IMAGES / self.dish_id / "rgb.png"

    def rgb_url(self) -> str:
        return f"{GCS}/imagery/realsense_overhead/{self.dish_id}/rgb.png"


def _plausible(d: Dish) -> bool:
    """Drop physically impossible rows (known Nutrition5k noise)."""
    return 0 < d.mass < 5000 and 0 <= d.cal < 5000


def parse_dish_metadata(path: Path) -> dict[str, Dish]:
    """Parse one ragged, headerless dish_metadata_cafeN.csv.

    Layout: dish_id, cal, mass, fat, carb, protein, then repeating 7-tuples that
    start with an `ingr_` token. Ingredient names may contain commas *and* spaces,
    so we anchor on the `ingr_` tokens and take the last 5 fields of each segment
    as numbers, joining whatever is left over as the name.
    """
    out: dict[str, Dish] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 6 or not row[0].startswith("dish_"):
                continue
            try:
                cal, mass, fat, carb, protein = (float(x) for x in row[1:6])
            except ValueError:
                continue
            anchors = [i for i, tok in enumerate(row) if tok.startswith("ingr_")]
            ings: list[tuple[str, float, float]] = []
            for k, start in enumerate(anchors):
                end = anchors[k + 1] if k + 1 < len(anchors) else len(row)
                seg = row[start:end]
                if len(seg) < 7:
                    continue
                try:
                    grams, kcal = float(seg[-5]), float(seg[-4])
                except ValueError:
                    continue
                ings.append((" ".join(seg[1:-5]).strip(), grams, kcal))
            out[row[0]] = Dish(row[0], cal, mass, fat, carb, protein, ings)
    return out


def load_dishes(drop_implausible: bool = True) -> dict[str, Dish]:
    dishes: dict[str, Dish] = {}
    for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        dishes.update(parse_dish_metadata(META / name))
    if drop_implausible:
        dishes = {k: v for k, v in dishes.items() if _plausible(v)}
    return dishes


def load_ingredient_macros() -> dict[str, dict[str, float]]:
    """ingredients_metadata.csv -> {name: {cal_g, fat_g, carb_g, protein_g}}.

    This is the published USDA-style lookup table. The oracle ladder instead learns
    densities from TRAIN, which sidesteps name-matching; keep both available so the
    two can be cross-checked.
    """
    out = {}
    with open(META / "ingredients_metadata.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["ingr"].strip()] = {
                    "cal_g": float(r["cal/g"]),
                    "fat_g": float(r["fat(g)"]),
                    "carb_g": float(r["carb(g)"]),
                    "protein_g": float(r["protein(g)"]),
                }
            except (ValueError, KeyError):
                continue
    return out


def load_split(which: str) -> list[str]:
    """Official RGB split ids, as published. Use usable_split() for real work."""
    assert which in ("train", "test")
    with open(META / f"rgb_{which}_ids.txt", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def available_image_ids(local_only: bool = False) -> set[str]:
    """Dish ids that have an overhead rgb.png.

    local_only=False consults data/n5k_overhead_manifest.json (what exists in the
    bucket); local_only=True checks the filesystem (what we have downloaded).
    """
    if local_only:
        return {p.parent.name for p in IMAGES.glob("*/rgb.png")}
    manifest = DATA / "n5k_overhead_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing — run code/fetch_images.py --manifest to rebuild it."
        )
    with open(manifest) as fh:
        return {k for k, v in json.load(fh).items() if "rgb.png" in v}


def usable_split(which: str, local_only: bool = False) -> list[str]:
    """Official split ids that survive both the plausibility filter and image availability."""
    dishes = load_dishes()
    have = available_image_ids(local_only=local_only)
    return [i for i in load_split(which) if i in dishes and i in have]


def timestamp(dish_id: str) -> int:
    return int(dish_id.split("_", 1)[1])


def session_ids(ids: list[str], gap_s: int = SESSION_GAP_S) -> dict[str, int]:
    """Map each dish id to a plate-session index.

    Consecutive dishes (by timestamp) closer than `gap_s` share a session. Computed
    over whatever ids are passed, so pass a whole split to get that split's sessions.
    """
    ordered = sorted(ids, key=timestamp)
    out, sid = {}, 0
    for prev, cur in zip([None] + ordered[:-1], ordered):
        if prev is not None and timestamp(cur) - timestamp(prev) > gap_s:
            sid += 1
        out[cur] = sid
    return out


def fit_calib_split(local_only: bool = False,
                    by_session: bool = True) -> tuple[list[str], list[str]]:
    """Deterministically carve TRAIN into (fit, calibration).

    The heads are fit on `fit`; conformal quantiles are computed on `calibration`.
    The official test split is never touched by either.

    by_session=True (default since 17 Sep 2026) keeps every plate session on one side,
    so calibration errors are not optimistic copies of fit errors. by_session=False is
    the dish-level shuffle the proposal numbers were computed with; keep it only to
    reproduce those.
    """
    ids = sorted(usable_split("train", local_only=local_only))
    rng = random.Random(CALIB_SEED)
    if not by_session:
        rng.shuffle(ids)
        n_cal = int(round(CALIB_FRACTION * len(ids)))
        return sorted(ids[n_cal:]), sorted(ids[:n_cal])
    sess = session_ids(ids)
    groups: dict[int, list[str]] = defaultdict(list)
    for d in ids:
        groups[sess[d]].append(d)
    order = sorted(groups)
    rng.shuffle(order)
    target = CALIB_FRACTION * len(ids)
    calib: list[str] = []
    for g in order:
        if len(calib) >= target:
            break
        calib.extend(groups[g])
    calib_set = set(calib)
    return sorted(d for d in ids if d not in calib_set), sorted(calib)


def summary() -> None:
    dishes = load_dishes()
    macros = load_ingredient_macros()
    print(f"dishes parsed (plausible)   : {len(dishes)}")
    print(f"ingredient macro table      : {len(macros)} ingredients")
    for which in ("train", "test"):
        official = load_split(which)
        usable = usable_split(which)
        local = usable_split(which, local_only=True)
        print(
            f"{which:5s}: official {len(official):5d} | with imagery {len(usable):5d} "
            f"({100 * len(usable) / len(official):.1f}%) | downloaded {len(local):5d}"
        )
    fit, calib = fit_calib_split()
    print(f"train sub-split             : {len(fit)} fit / {len(calib)} calibration "
          f"(seed={CALIB_SEED}, by plate session)")
    for which in ("train", "test"):
        ids = usable_split(which)
        n_sess = max(session_ids(ids).values()) + 1
        print(f"{which:5s} plate sessions          : {n_sess} (gap > {SESSION_GAP_S}s)")
    n_ing = [len(d.ingredients) for d in dishes.values()]
    print(f"ingredients per dish        : median {st.median(n_ing):.0f}, max {max(n_ing)}")
    vocab = defaultdict(int)
    for d in dishes.values():
        for n in d.names:
            vocab[n] += 1
    print(f"ingredient vocabulary       : {len(vocab)} distinct, "
          f"{sum(1 for v in vocab.values() if v >= 20)} appear in >=20 dishes")


if __name__ == "__main__":
    summary()
