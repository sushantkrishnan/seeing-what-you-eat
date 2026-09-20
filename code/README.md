# Pipeline code — setup and reproduction

Everything is inference-only and runs on an M-series MacBook. No training, no GPU cluster.

## One-time setup

```bash
cd /Users/sushi/AdvML
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install torch torchvision timm opencv-python \
    numpy pandas scipy scikit-learn matplotlib tqdm pillow

# data (~1.4 GB, ~8 min)
.venv/bin/python code/fetch_images.py --manifest
.venv/bin/python code/fetch_images.py --all
```

Verify:

```bash
.venv/bin/python code/n5k.py          # split sizes
.venv/bin/python code/oracle_ladder.py # the preliminary result
```

Run scripts from inside `code/` (they import each other as flat modules):

```bash
cd code && ../.venv/bin/python features.py --split test
```

## Architecture

The design constraint is that **one person's inference pass is the only shared
dependency**; every research question downstream is a `groupby` on the cached
features. A stalled member costs one column, not the project.

```
n5k.py         data access — dishes, splits, plate sessions, image paths.  ← single source of truth
degrade.py     the perturbation ladder (8 families x 5 severities + screenshot + phone)
quality.py     no-model image-quality metrics
features.py    frozen backbone -> cached embeddings.        ← THE inference pass
fetch_images.py resumable HTTPS downloader
oracle_ladder.py the ceiling analysis (metadata only, reads no images)
gate_bench.py  predictor + pixel metrics per (dish, rung) -> results/predictions.csv
gate_probe.py  every gate on one protocol, three backbones, cluster bootstrap, paired
               tests, pairwise tests -> results/gate_probe.json, results/gate_scores.csv
conformal.py   split conformal ranges, plain and adaptive width -> results/conformal.json
error_sources.py  where the error comes from, and which part of it is gateable
figures.py     every deck figure -> results/figures/
build_deck.py  the proposal deck (Aug 2026) -> 01_Proposal_v6.pptx
build_update_deck.py  the update deck (Sep 2026) -> 01_MethodResults.pptx
build_update_deck_v2.py  the size-controlled update deck (20 Sep) -> 01_MethodResults_v2.pptx
build_update_deck_v3.py  the same argument in plain sentences, notes read from the v3 script
                         -> 01_MethodResults_v3.pptx   (the deck to submit)
md2pdf.py, wordcount.py  talk-script tooling
```

Order matters at the bottom of that list: `gate_bench.py` writes `results/predictions.csv`
(which `gate_probe.py` reads for the pixel metrics), `figures.py` reads all three JSONs, and
the deck builders refuse to run if the figures are missing.

```bash
cd code
../.venv/bin/python gate_bench.py               # ~4.5 min: heads, errors, pixel metrics
../.venv/bin/python gate_probe.py --all --json  # ~2 min: every gate, 4 blocks, 2,000 resamples
../.venv/bin/python conformal.py --all --json   # ~25 s
../.venv/bin/python error_sources.py            # ~2 s
../.venv/bin/python figures.py                  # ~30 s
../.venv/bin/python build_update_deck_v3.py     # ~2 s (v2 and v1 builders still run)
```

**Standardise the features before the ridge.** The 2,304 dims are `[CLS | patch-mean |
patch-std]` and differ in scale by orders of magnitude, so an unstandardised penalty is
arbitrary across blocks. Z-scoring on the fit split is what reproduces 73.4 kcal / 44.1 g;
without it the identical code gives 78.3 / 46.8.

## Splits

The official split lists 4,059 train / 709 test dish ids, but **only 3,490 dishes have
overhead RGB imagery**, so the usable set is smaller. Always go through
`n5k.usable_split()`, never the raw id files.

| | official | with imagery |
|---|---|---|
| train | 4,059 | 2,755 (67.9%) |
| test | 709 | 507 (71.5%) |

Conformal needs a calibration set the heads never saw. `n5k.fit_calib_split()` carves
train into **1,924 fit / 831 calibration** with a fixed seed (760), **by plate session**
(since 17 Sep 2026; `by_session=False` reproduces the proposal's dish-level 1,929 / 826
carve). The official test split is never touched by either — do not recalibrate on test.

## Plate sessions

Dish ids are unix timestamps. Scans of one plate session arrive ~40 s apart; sessions are
hours or days apart, so the gap distribution is bimodal and `n5k.session_ids()` splits it
at 600 s: **367 sessions in train, 127 in test**. |calorie error| has an intraclass
correlation of 0.34 within a session, so the session is the unit for the fit/calibration
carve and for every bootstrap (`gate_probe.py`, `conformal.py`). The dish-level carve had
put 98% of calibration dishes in a session with a fit dish; fixing it changed the ridge
penalty picked on calibration and moved the predictor from 73.4 to 70.9 kcal, inside its
own interval [62.0, 80.4].

The official train/test split does not straddle sessions in a way that matters: test
dishes with a same-session train neighbour score 72.9 kcal against 75.9 without.

## Significance

`gate_probe.py` resamples the 127 test sessions 2,000 times and recomputes every gate's
whole risk-coverage curve on the same resample, so comparisons are paired. It reports, per
gate and backbone: MAE at every 5% of coverage with a 95% percentile interval; AURC (mean
error over coverages 10..100%); the share of the random-to-perfect AURC gap captured; the
paired difference to the random gate with interval and two-sided bootstrap p, Holm-corrected
across the ten candidate gates at each coverage; and pairwise tests between model-aware
gates at 90% and 50%.

The random reference is analytic (added 20 Sep): refusing at random has expected selective
risk equal to the answer-everything risk at every coverage, so that is the curve every gate
is tested against, on the same resample. Before that it was one fixed random draw, whose
seed noise (±1–2 kcal) leaked into every paired difference; ensemble disagreement clears
the Holm correction at 90% against the analytic reference and did not against the draw.

## Controlling for size

|error| scales with meal size, so a gate can post a good MAE at fixed coverage by refusing
large meals; the `pred_magnitude` control ties the learned error head at 90% answered, and
the head's score has Spearman 0.77 with the predicted calorie count. `gate_probe.py`
therefore writes two further blocks per backbone:

- `clean_sized`: the same gates, same kcal loss, but every gate must refuse the same share
  **inside each predicted-size band** (`SIZE_BANDS = 10`, edges from the calibration
  split's predictions). Size cannot be the lever across bands. The control collapses to
  −1.0 / +0.4 / −0.4 kcal at 90% (CLIP / SigLIP 2 / DINOv3), which validates the design;
  the learned head keeps −6.5 / −6.5 / −4.7 (Holm p < 0.01 on all three, ahead of the
  control at paired p ≤ 0.02); the perfect refuser keeps −14.5 / −14.9 / −12.9 of its
  −20.4 / −19.9 / −17.9. About half the head's win, and a third of the ceiling, was size.
- `clean_rel`: the same gates on relative error `|err| / max(true, 50)`. This loss has the
  opposite bias (its oracle refuses 0% of 400+ kcal meals and 27% of sub-100 kcal ones; 83
  test dishes are under 50 kcal), so it is a secondary check: at 90% no gate beats random,
  the control is significantly worse (+1.6 points, Holm p < 0.01, all three), the head is
  null, ensemble disagreement leans the right way (−2.2 to −2.9, n.s. after Holm).

Each gate also carries `rho_size` (Spearman with the predicted calorie count) and
`rho_within_size` (mean Spearman with |error| inside predicted-size quartiles), and every
`selectivity` entry now has `small_refused` beside `big_refused`.

## Degradation ladder

`degrade.py` defines 8 families — `blur, motion, jpeg, downscale, temp, dark, crop,
noise` — at 5 severities each, plus two compound rungs: `screenshot` and `phone` (blur 1 px,
JPEG 75, warm shift 0.1, crop to 85%; an imitation of a hand-held phone photo, stated as
such, since Nutrition5k has no phone photos). Specs are strings: `blur:3`, `jpeg:30`,
`blur:3+jpeg:40`, `clean`, `screenshot`, `phone`.

Degradations are applied on the fly and are deterministic given `(image, spec)`, so
they are exactly reproducible without storing 1.45 GB per rung.

`crop` is the rung to watch: it removes the plate rim and table edge, which are the
only scale references in a fixed-camera overhead shot, so it should hurt the mass head
far more than the naming head.

`noise` is not a realistic phone artifact — it is the control that makes the
adversarial argument concrete (see below).

Regenerate the visual check with `../.venv/bin/python degrade.py`, which writes 41
example images to `results/degradation_examples/`.

## Quality metrics — measured behaviour

`quality.py --sample 40` produces the table below (means over 40 test dishes). This
already settles a design question: **no single metric covers the space.**

| rung | lap_var | tenengrad | blockiness | hf_ratio |
|---|---|---|---|---|
| clean | 183.1 | 297.7 | 0.01 | 0.017 |
| blur:6.0 | 1.5 | 25.2 | 0.00 | 0.004 |
| jpeg:10.0 | **151.6** | 294.1 | **1.52** | 0.017 |
| crop:0.4 | 6.8 | 46.0 | 0.01 | 0.005 |
| noise:0.16 | **13288** | 2888 | 0.00 | 0.073 |

Three consequences:

1. **Sharpness metrics are blind to compression.** `lap_var` moves only 183 → 152
   across the entire JPEG sweep (0.83×). `blockiness` is the metric that catches it,
   and it is near-zero for every other family. Both are required.
2. **`lap_var` confounds darkness with blur** — `dark:0.3` crushes it 183 → 17.6 purely
   from the brightness scaling, with no real information lost. `hf_ratio` is
   brightness-invariant (0.0172 vs 0.0169 clean) and does not have this problem.
3. **Noise raises `lap_var` by up to 72×.** A sharpness-based quality gate scores a
   noise-destroyed image as dramatically *better* than the original. This is the
   adversarial failure mode demonstrated for free, with no attack: adversarial
   perturbation is high-frequency, so it would be routed into the *narrowest,
   most-confident* conformal bin while carrying the largest error. Any quality gate we
   ship has to be tested against this, which is what motivates adding a model-aware
   signal (TTA disagreement) alongside the no-model metrics.

`colour_rb` tracks `temp` cleanly (1.10 → 3.17) but also drifts under `crop`
(1.10 → 1.40) because cropping removes the neutral dark frame — a confound to note
when binning.

## Backbone

Default `vit_base_patch16_clip_224.openai` via `timm`, frozen, `num_classes=0`.
Features are `[CLS ; mean(patch tokens) ; std(patch tokens)]` = 2,304-d. The patch
statistics are included because the mass head needs spatial extent, which a pooled CLS
embedding represents poorly.

Measured throughput on MPS: **~68 img/s** sustained, so one rung over all 3,490 dishes is
~48s and the full 42-rung ladder is about **34 minutes of laptop compute**. (A short run
reports ~40 img/s — that is MPS warm-up, not the steady-state rate.)

Swap backbones with `--model` (e.g. `vit_base_patch14_dinov2.lvd142m`). Keeping the
backbone public and frozen is deliberate: it is what makes a white-box attack a
realistic threat model rather than a contrived one, since the deployed encoder is not
secret.

Input gradients through the frozen backbone are verified working on MPS, which is what
the adversarial stress test needs.

## Preliminary result

`oracle_ladder.py` reports both the full test split and the imagery-restricted one:

| oracle knows | n=707 | n=507 (has imagery) |
|---|---|---|
| nothing | 160.3 | 167.1 |
| total mass only | 96.8 | 107.1 |
| ingredient identity only | 88.3 | 102.4 |
| **identity + total mass** | **29.5** | **32.7** |
| identity + exact grams | 3.0 | 4.2 |

The headline survives the restriction: 32.7 still beats the best published result on
this dataset (41.3, Thames et al. CVPR 2021). Identity alone and mass alone remain
nearly useless separately — the information is complementary, not additive.

**Use 32.7 / n=507 in the report**, not 29.5 / n=707. The n=707 figure is computed over
dishes we cannot actually run a model on.
