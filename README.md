# Seeing what you eat: when should a nutrition app refuse to answer?

COMPSCI 760 (Advanced Machine Learning), University of Auckland, Group 1, semester 2 2026.
Sushant Krishnan · Keanu De Cleene · Nisarg Patel · Shreya Tigga · Kevin Zou.

Photo-based calorie estimators answer every photo with one number. This project freezes the
predictor and benchmarks **the gate**: the signals an app could use to decide whether to answer
at all, scored on the error-versus-reject protocol between refusing at random (the reference)
and a perfect refuser (the ceiling), on Nutrition5k.

## Results (11 Oct 2026)

These are the numbers in the final presentation (`01_FinalResults.pdf`). Unless stated: 90%
of photos answered (the app refuses one in ten), CLIP / SigLIP 2 / DINOv3, change in mean
absolute error (kcal) against refusing at random, negative is better; 95% intervals from a
bootstrap over plate sessions (2,000 resamples), Holm-corrected across the candidate gates.

| | kcal MAE, 507 test dishes |
|---|---|
| Published RGB baseline, Inception V2, trained (Thames et al., CVPR 2021) | 70.6 |
| Ours: frozen CLIP ViT-B/16 + ridge, no training | **70.9 [62.0, 80.4]** |
| Ours: frozen SigLIP 2 / DINOv3 | 67.5 / 63.2 |

1. **Photo quality is not the warning sign.** Which dish it is explains 62% of the error
   variance, photo damage 8.5%. Pixel metrics detect damaged photos (AUC ≈ 1) but their rank
   correlation with the calorie error stays within ±0.08. On all 3,262 clean photos, inside
   size bands, no pixel gate could be worth more than 8% of a perfect refuser's saving (lower
   95% bound); sharpness points the wrong way (+1.2 / +1.0 / +1.0; Holm-significant on CLIP).
2. **Most of a model-aware gate's win is refusing big meals.** Without a size control,
   refusing the meals with the biggest predicted calories removes 7.9 / 7.6 / 7.5 kcal, more
   than the learned error head (5.2 / 6.1 / 6.2), by keeping meals averaging 213–215 kcal
   where a perfect refuser keeps 224–225. Every gate is therefore also scored refusing the
   same share inside each of 10 predicted-size bands.
3. **Inside size bands a small difficulty signal is real.** Pooled over all 3,262 dishes
   (5 repeats of 5-fold cross-fitting by plate cluster; 366 clusters, since train and test
   scans interleave in time): learned error head −1.4 / −1.8 / −2.1, ensemble disagreement
   −1.3 / −2.0 / −1.9 (Holm p ≤ 0.014 on all three), size control −0.6 / −1.0 / −0.8,
   perfect refuser −15.3 / −15.2 / −14.2: about a tenth of the achievable saving. Heads
   trained on shuffled errors (200 per backbone) average zero; the real head is 4.7–7.3
   standard deviations away. With 20 bands the size control falls to −0.3 / −0.7 / −0.1
   while the head holds; at 50% answered the head removes 6.0–7.9 kcal. Inside bands the
   head beats the size control over the whole curve on all three backbones (p ≤ 0.001) and
   at 90% on two of three; the head and ensemble disagreement cannot be separated.
4. **The update talk (20 Sep) overstated this.** Its −6.5 / −6.5 / −4.7 came from one
   fit/calibration carve that was the most favourable of 20 for the head on two backbones
   (4th on DINOv3). Averaged over 20 carves: −2.5 / −2.7 / −3.0; on the 367 train sessions,
   never used to score a gate: −1.3 / −2.6 / −1.9. The session bootstrap held the fitted
   models fixed, so its intervals left out retraining variance. "Half of the win survives
   the size control" is retired.
5. **Size-aware ranges keep big meals honest.** Same-width split-conformal ranges cover
   about 90% overall but only 67 / 71 / 67% of the largest-meal quarter. Conformalised
   quantile regression covers 89 / 91 / 90% of that quarter averaged over 20 carves and
   89 / 88 / 90% on new sessions, with a better interval score; the smallest quarter falls
   to 82–87%. Behind an ensemble-disagreement gate set on calibration (answering 89–91% of
   test photos), ranges calibrated on answered photos cover 91–94% of answered test photos.

Exploratory, designed after the instability above was found and run once: averaging 10
error heads, each calibrated on a different carve, makes one fitted head steadier and
removes 2.1–2.7 kcal more over the whole curve (p ≤ 0.001), with no change at 90%.
Disagreement across the three backbones helps CLIP and hurts DINOv3.

Design and hypotheses of the confirmation runs: the docstrings of `code/replicate.py` and
`code/robustness.py`; every number: `results/replication.json`, `results/robustness.json`.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision timm opencv-python numpy pandas scipy scikit-learn matplotlib tqdm pillow python-pptx
bash reproduce.sh          # everything: downloads, features, every result, figure and the final deck (~100 min)
bash reproduce.sh --fast   # skip downloads and feature extraction once data/cache/ exists (~70 min)
```

`code/README.md` documents each module and says which parts are ours and which are taken
from published work.

## Layout

```
code/           the pipeline (see code/README.md)
reproduce.sh    reruns every experiment, figure and the final deck
results/        gate_probe.json, replication.json, robustness.json, gate_bench.json,
                conformal.json, predictions.csv, train_pixel.csv,
                gate_scores.csv (12,168 rows: 3 backbones × 507 dishes × 8 damage levels),
                figures/
n5k_metadata/   Nutrition5k metadata and official split ids, pulled locally
data/           image manifest (images and feature caches are not committed)
01_Proposal.pdf         proposal deck (Aug 2026)
01_MethodResults.pdf    methodology-and-results deck (v4, 20 Sep 2026; built by code/build_update_deck_v4.py)
01_FinalResults.pdf     final-results deck (19 Oct 2026; built by code/build_final_deck.py)
```

## Generative AI use

Claude Code (Anthropic) was used throughout, under the group's direction: it wrote most of
the pipeline and analysis code, proposed and ran the robustness checks after the group asked
how to answer the update-talk feedback, and drafted the figures, slides and speaker notes.
The group chose the direction and which checks to run, and is responsible for every claim;
every number on the slides is read from the results files by the deck builders.

## Reference

Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, "Nutrition5k:
Towards automatic nutritional understanding of generic food," CVPR 2021.
