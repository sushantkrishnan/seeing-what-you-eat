#!/usr/bin/env bash
# Reproduce every number, table and figure in the final presentation from scratch.
#
#   bash reproduce.sh            # everything, ~100 min on an M-series laptop once images are local
#   bash reproduce.sh --fast     # skip the downloads and feature extraction (needs data/cache)
#
# Every step writes to results/; nothing is trained by gradient descent (all heads are
# closed-form ridge regressions on frozen features), so there is no convergence to wait for.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(pwd)/.venv/bin/python"
FAST="${1:-}"

if [[ "$FAST" != "--fast" ]]; then
  # 1. data: 3,490 overhead RGB photos from the public Nutrition5k bucket (~1.4 GB, ~8 min)
  "$PY" code/fetch_images.py --manifest
  "$PY" code/fetch_images.py --all
  # 2. the one inference pass: frozen backbones -> cached features (~1 min per rung per backbone)
  cd code
  for m in vit_base_patch16_clip_224.openai vit_base_patch16_siglip_224.v2_webli \
           vit_base_patch16_dinov3.lvd1689m; do
    "$PY" features.py --model "$m" --split train
    for rung in clean blur:3 blur:6 crop:0.4 jpeg:10 downscale:8 noise:0.16 phone; do
      "$PY" features.py --model "$m" --split test --degradation "$rung"
    done
  done
  for rung in blur:3 blur:6 crop:0.4 jpeg:10 downscale:8 noise:0.16 phone; do
    "$PY" features.py --split calib --degradation "$rung"     # CLIP only, for the mixtures
  done
  cd ..
fi

cd code
"$PY" oracle_ladder.py                          # ~2 s   the error budget (metadata only)
"$PY" gate_bench.py                             # ~4.5 min predictor, pixel metrics -> predictions.csv
"$PY" gate_probe.py --all --json                # ~2 min  every gate, 3 backbones -> gate_probe.json
"$PY" replicate.py --all --json --carves 20     # ~35 min fresh-session replication + 20 carves -> replication.json
"$PY" robustness.py --json                      # ~25 min pooled headline, sensitivity, placebo,
                                                #         stabilised heads, ranges re-checked -> robustness.json
"$PY" conformal.py --all --json                 # ~2 min  confidence ranges -> conformal.json
"$PY" error_sources.py                          # ~2 s   where the error comes from
"$PY" figures.py                                # ~30 s  every figure -> results/figures/
"$PY" build_final_deck.py                       # ~2 s   the final deck -> 01_FinalResults.pptx
echo "done: results/*.json, results/figures/ and 01_FinalResults.pptx are up to date"
