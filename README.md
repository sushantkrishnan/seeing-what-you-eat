# Seeing what you eat: when should a nutrition app refuse to answer?

COMPSCI 760 (Advanced Machine Learning), University of Auckland, Group 1, semester 2 2026.
Sushant Krishnan · Keanu De Cleene · Nisarg Patel · Shreya Tigga · Kevin Zou.

Photo-based calorie estimators answer every photo with one number. This project freezes the
predictor and benchmarks **the gate**: the signals an app could use to decide whether to answer
at all, scored on the error-versus-reject protocol between refusing at random (the reference)
and a perfect refuser (the ceiling), on Nutrition5k.

## Results so far (20 Sep 2026)

All numbers are on the official Nutrition5k test split restricted to dishes with overhead RGB
(507 dishes, 127 plate sessions). Intervals are 95%, cluster bootstrap over plate sessions,
2,000 resamples; every gate is tested against refusing at random on the same resample and
Holm-corrected across the ten candidate gates.

| | kcal MAE |
|---|---|
| Published RGB baseline, Inception V2, trained (Thames et al., CVPR 2021) | 70.6 |
| Ours: frozen CLIP ViT-B/16 + ridge, no training | **70.9 [62.0, 80.4]** |
| Ours: frozen SigLIP 2 / DINOv3 | 67.5 / 63.2 |

**Gates, clean photos, answering 90% (refusing 1 in 10), CLIP.** Paired difference to the
random reference (which has the answer-everything error at every coverage):

| gate | kcal | vs random | Holm p |
|---|---|---|---|
| perfect refuser (ceiling) | 50.4 | −20.4 | |
| learned error head | 59.7 | −11.2 [−16.1, −6.4] | 0.005 |
| predicted size (degeneracy control) | 60.6 | −10.2 [−14.5, −6.8] | 0.005 |
| ensemble disagreement | 64.9 | −6.0 [−9.8, −2.6] | 0.007 |
| distance to mean / neighbour distance | 68.2 / 68.6 | n.s. | 0.77 |
| random (reference) | 70.9 | | |
| sharpness (Laplacian variance) | 73.6 | +2.8 [+0.5, +4.1] | 0.05 |

The learned error head and the predicted-size control are statistically tied (paired
p = 0.76), and the head's score has Spearman 0.77 with the predicted calorie count, so a plain
MAE-at-coverage cannot say whether a gate reads the dish or just its size.

**Controlling for size.** Every gate is scored again with the same kcal loss but refusing the
same share *inside each of ten predicted-size bands* (edges from the calibration split). At 90%
answered, CLIP / SigLIP 2 / DINOv3, change against random:

| gate | overall | inside size bands |
|---|---|---|
| predicted-size control | −10 / −8 / −8 | **−1.0 / +0.4 / −0.4** (zero by construction) |
| learned error head | −11 / −13 / −12 | **−6.5 / −6.5 / −4.7**, Holm p < 0.01, ahead of the control (paired p ≤ 0.024) |
| ensemble disagreement | −6 / −6 / −6 | −5.2 / −1.5 / −4.0 |
| perfect refuser | −20 / −20 / −18 | −14.5 / −14.9 / −12.9 |

Half of the best gate's win is meal size; the other half survives on all three backbones. On
relative error (|err| / max(true, 50 kcal)) no gate beats random at 90% and the control is
significantly worse: the two losses reward refusing big and small meals respectively, so
neither judges a gate alone.

Pixel gates: on clean photos none beats random at any coverage on any backbone; on the pooled
damage ladder every one sits within 2 kcal of random at 90% answered (sharpness significantly
worse on all three), and the most any captures over all coverages is 9% of the gap, by dropping
whole damage levels. Which dish it is explains 62% of the error variance; the damage level 9%.

**Confidence ranges (split conformal, 90% target).** Plain ranges cover 89.5% overall but only
80% of the hardest quarter of dishes; letting ensemble disagreement set the width moves that to
85% (SigLIP 2: 90%) and improves the interval score. Both under-cover the largest quarter of
meals (~66%), the same size effect, which is the next problem.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision timm opencv-python numpy pandas scipy scikit-learn matplotlib tqdm pillow python-pptx
.venv/bin/python code/fetch_images.py --manifest && .venv/bin/python code/fetch_images.py --all   # 1.4 GB
cd code
../.venv/bin/python features.py --split train                     # cached features, ~1 min per rung
../.venv/bin/python features.py --split test --degradation blur:3   # one rung of the ladder
../.venv/bin/python gate_bench.py               # predictor, pixel gates, results/predictions.csv
../.venv/bin/python gate_probe.py --all --json  # every gate, three backbones, cluster bootstrap
../.venv/bin/python conformal.py --all --json   # confidence ranges
../.venv/bin/python error_sources.py            # where the error comes from
../.venv/bin/python figures.py                  # results/figures/
../.venv/bin/python build_update_deck_v3.py     # 01_MethodResults_v3.pptx from results/*.json
```

`code/README.md` documents each module. The whole pipeline reruns on an M-series laptop in
under 15 minutes once features are cached.

## Layout

```
code/           the pipeline (see code/README.md)
results/        gate_probe.json, gate_bench.json, conformal.json, predictions.csv,
                gate_scores.csv (12,168 rows: 3 backbones × 507 dishes × 8 damage levels),
                figures/
n5k_metadata/   Nutrition5k metadata and official split ids, pulled locally
data/           image manifest (images and feature caches are not committed)
01_Proposal.pdf         proposal deck (Aug 2026)
01_MethodResults.pdf    methodology-and-results deck (v3, 20 Sep 2026; built by code/build_update_deck_v3.py)
```

## History

The code was written from 16 August 2026 in a working folder without version control. The
repository was created on 17 September 2026; commits dated before that carry author dates
taken from the files' modification times and contain only files unchanged since then. Files
that were extended on 17 September appear in that day's commit.

## Generative AI use

Claude Code (Anthropic) was used to scaffold pipeline code, generate the slides from the
results files, and improve wording. Experimental design, analysis choices, results and
conclusions are the group's own and reproduce from this repository.

## Reference

Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, "Nutrition5k:
Towards automatic nutritional understanding of generic food," CVPR 2021.
