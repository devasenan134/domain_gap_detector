# Project 1: Synthetic-to-Real Domain Gap Detector for Object Detection

## 1. Problem Statement

Teams that use synthetic data (rendered via Blender/BlenderProc, game engines, or
generative models) to train computer vision models have no fast, quantitative way to
check whether a given synthetic dataset will transfer to the real world before
committing to a full train/eval cycle. In practice, this judgment is made by hand —
someone inspects renders, compares them intuitively to real samples, and only finds
out if the domain gap was too large after training and evaluating on real data. This
project builds a system that:

1. Generates domain-randomized synthetic training data for a bounded object
   detection task.
2. Quantifies the distributional gap between synthetic and real data using a cheap,
   no-training-required metric.
3. Empirically validates that this gap metric correlates with actual real-world
   detection performance, via a controlled ablation across randomization settings.
4. Wraps the trained detector as a served, monitored inference system — not just a
   notebook result.

This is an engineering-first project: the deliverable is a working, deployable
system with one honest, well-scoped empirical finding (which randomization axes
matter most for transfer), not a claim of novel research.

## 2. Task and Data

- **Task:** 2D object detection, YOLO-family model (Ultralytics YOLOv8 or current
  release).
- **Object set:** 8 classes from the YCB-Video (YCB-V) object set, chosen for
  geometric and texture diversity:
  `002_master_chef_can`, `003_cracker_box`, `005_tomato_soup_can`,
  `006_mustard_bottle`, `011_banana`, `021_bleach_cleanser`, `035_power_drill`,
  `037_scissors`.
- **Real-world evaluation set:** YCB-V real test images (via the BOP benchmark
  toolkit), fixed and untouched throughout — this is the ground truth the synthetic data must transfer to.
- **Synthetic data:** generated with BlenderProc, using the official YCB-V CAD
  models loaded via `bproc.loader.load_bop_objs()`, rendered under a sweep of
  domain-randomization configurations.

## 3. Domain-Randomization Axes (the sweep)

Each synthetic dataset "variant" is a fixed combination of settings across these
axes:

- HDRI background / lighting intensity and direction
- Texture/material randomization strength (procedural vs. photoreal)
- Render sample count (noise level)
- Camera pose distribution (constrained to match YCB-V's real capture range)
- Object pose and occlusion randomization

8–12 variants will be generated, spanning from "low randomization" (near-identical
renders, minimal variation) to "high randomization" (aggressive domain
randomization).

## 4. Methodology

### Phase A — Baseline pipeline
Render one low-randomization synthetic variant, train YOLO, evaluate on the real YCB-V test set. This is the starting domain-gap measurement and the sanity check that the full pipeline (render → train → eval) works end to end.

### Phase B — Sweep and ground-truth curve
Repeat across all variants. For each: train (or fine-tune from a common starting checkpoint, to control compute cost), evaluate real-world mAP, log everything to Weights & Biases.
This produces the ground-truth relationship between randomization configuration and real-world transfer performance — the core empirical result of the project.

### Phase C — Gap scorer
For each variant, compute a distributional distance between the synthetic image set and the real reference set in a pretrained feature space (FID or KID, using DINOv2 or CLIP features — DINOv2 is preferred since it sets up reuse in Project 3).
This score requires no training. Compare the gap-score ranking of variants against the real mAP ranking from Phase B to test whether the cheap metric predicts the expensive outcome.

### Phase D — Ablation
Using the Phase B/C data, isolate which individual randomization axes (lighting vs. texture vs. render noise vs. pose) contribute most to closing the domain gap, and whether that differs by object (e.g., low-texture objects like the banana may be more lighting-sensitive than high-texture objects like the cracker box).
This is the one legitimate empirical claim in the project — keep it honest and bounded to what the data actually shows.

### Phase E — Serving
Wrap the best-performing detector as a Dockerized FastAPI inference service:
endpoint accepts an image, returns detections.
Include the gap scorer as a second endpoint: given a new synthetic batch, return a predicted-gap score against the stored real reference embeddings, without requiring a training run.

### Phase F — Drift monitoring
A lightweight monitor that compares incoming "production" image embeddings against the stored real reference distribution (e.g., MMD or KS-test on feature projections), flags when drift exceeds a threshold, and logs the result.
Doesn't need to be elaborate — a working detector with a sensible threshold and a log is sufficient.

## 5. System Architecture

```
synthetic-domain-gap/
├── blender/              # BlenderProc render scripts, randomization configs (YAML per variant)
├── data/                 # synthetic variants + real YCB-V eval set (gitignored/DVC)
├── src/
│   ├── detector/          # YOLO train/eval wrappers, config-driven (Hydra)
│   ├── gap_scorer/         # FID/KID computation over DINOv2 features
│   ├── drift/              # drift monitor (MMD/KS-test)
│   └── api/                # FastAPI app: /detect, /gap-score, /drift-check
├── docker/
├── experiments/            # W&B run configs and sweep results
├── notebooks/               # analysis and plots only, no pipeline logic
└── README.md
```

## 6. Evaluation Criteria (what "done" looks like)

- Working render → train → eval pipeline, reproducible from config files.
- A plot: randomization configuration vs. real-world mAP, across all variants.
- A plot: gap score vs. real-world mAP, with a stated correlation (Pearson/Spearman)
  — reported honestly even if the correlation is moderate, not overstated.
- A short ablation write-up: which randomization axis mattered most, and for which
  object types.
- A working Dockerized API, callable with `curl` or a simple client script,
  documented in the README.
- A working drift monitor demonstrated on a deliberately shifted input batch
  (e.g., real images under different lighting) to show it actually fires.

## 7. Deliverables

- GitHub repo, structured as above, with a README that states the problem,
  approach, results (plots included), and how to reproduce.
- Resume bullet (drafted once results are in — depends on final mAP/correlation
  numbers).
- A short internal note translating this project's components to your Hyperverge
  experience, for interview use (without disclosing employer-specific data).

## 8. Anticipated Interview Questions (prep as you build, not after)

- Why YOLO instead of a two-stage detector? (Speed/simplicity tradeoff — be ready
  to name the alternative and why you didn't need it.)
- Why FID/KID instead of a simpler pixel-space distance? (Feature-space distance
  captures semantic/textural differences pixel distances miss — but know FID's
  known limitations, e.g. sensitivity to sample size.)
- What would you do with more compute? (Larger sweep resolution, multiple seeds
  per variant to separate randomization effect from training noise.)
- What's a case where the gap score and real mAP disagreed, and why? (You need at
  least one real example of this from your own sweep — don't let the story be
  "it worked perfectly," that's not credible.)
- How does this generalize beyond YCB-V objects? (Be honest about what's
  object-specific vs. general in your findings.)

## 9. Explicit Non-Goals

- Not a novel research contribution — the gap-scoring approach (FID/KID for
  synthetic data quality) exists in the literature; this project applies and
  validates it in a controlled setting, doesn't invent it.
- Not a robotics/ROS2 project — no robot integration, no grasp planning. YCB-V is
  used purely as a convenient, well-matched synthetic/real object pairing.
- Not attempting full YCB-V 21-class coverage — 8 classes is a deliberate scope
  boundary to keep render/train time manageable.
