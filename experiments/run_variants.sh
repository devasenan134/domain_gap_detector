#!/usr/bin/env bash
# Render every named variant first, then train and evaluate each one. Run from the repo root:
#   bash experiments/run_variants.sh lighting material camera noise background high_randomization
# Safe to re-run: rendering skips finished scenes, training skips variants that have a results file.
#
# Optional environment variables:
#   BLENDER=<blender folder>   defaults to the macOS copy from the README
#   TRAIN_CONFIG=<yaml>        defaults to experiments/train.yaml
#   RENDER_CPU=1               hide the CUDA GPU from Blender, so a GPU shared with other jobs stays free
set -euo pipefail

BLENDER=${BLENDER:-$HOME/blender/blender-4.2.1-macos-arm64/blender_unbundled}
TRAIN_CONFIG=${TRAIN_CONFIG:-experiments/train.yaml}

for v in "$@"; do
  env ${RENDER_CPU:+CUDA_VISIBLE_DEVICES=-1} .venv/bin/blenderproc run --custom-blender-path "$BLENDER" blender/generate.py "blender/configs/$v.yaml"
done

for v in "$@"; do
  [ -f "experiments/results/$v.json" ] || .venv/bin/python src/detector/train.py "$v" --config "$TRAIN_CONFIG"
done
