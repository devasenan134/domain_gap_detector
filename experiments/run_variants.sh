#!/usr/bin/env bash
# Render every named variant first, then train and evaluate each one. Run from the repo root:
#   bash experiments/run_variants.sh lighting material camera noise background high_randomization
# Safe to re-run: rendering skips finished scenes, training skips variants that have a results file.
set -euo pipefail

BLENDER=~/blender/blender-4.2.1-macos-arm64/blender_unbundled

for v in "$@"; do
  .venv/bin/blenderproc run --custom-blender-path "$BLENDER" blender/generate.py "blender/configs/$v.yaml"
done

for v in "$@"; do
  [ -f "experiments/results/$v.json" ] || .venv/bin/python src/detector/train.py "$v"
done
