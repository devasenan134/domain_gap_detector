
# Project: sim2real

## Data related:
YCB-V (YCB Video Dataset): Contains 21 household and kitchen objects with rich textures and colors (e.g., cracker boxes, mustard bottles, soup cans). It features complex real-world clutter, making it excellent for testing object recognition in messy environments.

    1. bop_toolkit from: https://github.com/thodan/bop_toolkit
    2. models from: the original dataset https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_models.zip
    3. test set from BOP`19-24 test images: https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip
    

textures for scenes
uv run python blender/download_textures.py cc_textures 100   # 100 materials at 1K, about 0.5 GB (the stock `blenderproc download cc_textures` pulls all of them, about 55 GB)

## Synthetic data generation

One YAML per variant in `blender/configs/`. `blender/generate.py` renders it and writes YOLO labels directly
(visible boxes from the segmentation map, same as `bbox_visib` in the real test set) — no COCO step needed.

    uv run blenderproc run blender/generate.py blender/configs/baseline.yaml
    # -> data/yolo_dataset/baseline/{images,labels}/{train,val}, dataset.yaml (test: real YCB-V)

Needs `data/ycbv/models` + `data/ycbv/camera_uw.json` (ycbv_models.zip + ycbv_base.zip).
Re-running resumes: finished scenes are skipped. `--num-scenes 2` for a quick test.

macOS: BlenderProc can't install its pip packages inside the protected `Blender.app`, so copy it out once and point at the copy:

    cd ~/blender/blender-4.2.1-macos-arm64 && cp -R Blender.app blender_unbundled && ln -s Contents/Resources/4.2 blender_unbundled/4.2
    uv run blenderproc run --custom-blender-path ~/blender/blender-4.2.1-macos-arm64/blender_unbundled blender/generate.py blender/configs/baseline.yaml

## Training and real-world evaluation

`src/detector/train.py` trains on one variant with the shared settings in `experiments/train.yaml`, picks the checkpoint on the synthetic val split, then evaluates once on the real test set.

    uv run python src/detector/train.py baseline
    # -> runs/detect/baseline/weights/best.pt, experiments/results/baseline.json

W&B runs are logged offline by default. Upload them with `wandb sync wandb/offline-run-*`, or set `WANDB_MODE=online`.

Needs the real test set: unzip ycbv_test_bop19.zip into `data/ycbv/`, then run `notebooks/prep_ycbv_test.ipynb`.

## Sweep

`blender/configs/` has one YAML per variant: `baseline`, `high_randomization`, and five single-axis variants (`lighting`, `material`, `background`, `camera`, `noise`) that change one section of the baseline. Render all of them, then train and evaluate each:

    bash experiments/run_variants.sh lighting material camera noise background high_randomization

## DINOv2 gap scoring

Run on the machine with the complete sweep datasets (currently `enderpearl`,
`~/code/domain_gap_detector_sweep`):

    .venv/bin/python src/gap_scorer/score.py baseline lighting material camera noise background high_randomization
    # -> experiments/gap_scores.json; cached features -> data/embeddings/

Uses the final CLS token from `facebook/dinov2-small`, following the
[Transformers DINOv2 interface](https://huggingface.co/docs/transformers/model_doc/dinov2).
The model commit is resolved and recorded; use `--revision <commit>` to reproduce it.
Each variant uses all 450 synthetic training images against the same 750 real test
images. Synthetic validation images are excluded. Cache keys include model commit,
image paths, sizes, and modification times. Keep source files unchanged to reuse a
cache; image changes regenerate it. `--device cpu` and `--batch-size 8` are available.

These are **DINOv2 feature-space** Frechet and polynomial KID distances, not standard
Inception FID/KID numbers. KID is the full unbiased degree-three polynomial MMD
estimate; a negative estimate is valid. Features are not L2-normalized. Lower scores
mean a smaller measured gap. FID is biased at these small sample sizes.

Current seven-variant results:

| Variant | Real mAP50–95 (%) | DINOv2 FID | DINOv2 KID |
| --- | ---: | ---: | ---: |
| Baseline | 21.07 | 1665.23 | 83.91 |
| Lighting | 20.17 | 1676.31 | 83.82 |
| Material | 26.38 | 1647.44 | 81.61 |
| Camera | 8.60 | 1647.21 | 72.56 |
| Noise | 27.20 | 1629.33 | 79.60 |
| Background | 34.78 | 1533.37 | 68.72 |
| High randomization | 44.66 | 1573.41 | 61.21 |

Against real mAP50–95, FID Pearson/Spearman are **−0.736/−0.750**;
KID Pearson/Spearman are **−0.600/−0.607**. These are exploratory correlations across
seven single-seed runs, not an independent predictive validation: the same real
test images supply both the reference features and detector evaluation. Camera is
a counterexample: its KID is lower than baseline despite much worse detection.

Numerical and cache checks (no model download):

    .venv/bin/python -m unittest discover -s src/gap_scorer -p 'test_*.py'

## Preserved sweep artifacts

`experiments/sweep_inventory.json` records remote checkpoint hashes, dataset hashes,
and image counts. All seven remote variants have 450 training and 50 validation
images, and a best checkpoint. `experiments/gap_scores.json` records the exact
model revision, cache keys, source result hashes, scores, and correlations.

A checksum-verified backup of all seven best checkpoints, result JSONs, render
configs, training config, and dataset YAMLs is stored locally at
`weights/sweep_backup/domain-gap-sweep-backup.tgz` (gitignored).
SHA-256: `70f1cbbc65f8b882166f5594e06b8576b0e44d0744a71553f84b4d0e8dcf4cf4`.
Full image datasets remain on the remote machine; the archive does not duplicate
them. Cached embeddings are also copied locally to `data/sweep_embeddings/`.

`baseline.json` is the current remote CUDA result (21.07% mAP50–95).
`baseline_omen_previous.json` preserves the earlier local CUDA result (16.93%);
the reason for that run's replacement was not established. `baseline_mac.json`
preserves the original MPS/batch-16 run. Neither historical baseline is included
in the seven-variant correlation. All compared runs use CUDA/batch 8; the scorer
rejects differing saved training configurations.
