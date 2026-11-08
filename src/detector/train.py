"""Train YOLO on one synthetic variant and evaluate it on the real YCB-V test set.

Usage (from the repo root):
    uv run python src/detector/train.py baseline
    uv run python src/detector/train.py baseline --epochs 1   # quick test, saved as baseline_e1

The checkpoint is picked on the synthetic val split. The real test set is only
used for the final evaluation, so it can't leak into training.

Output:
    runs/detect/<name>/weights/best.pt
    experiments/results/<name>.json
"""

import argparse
import json
import os
from pathlib import Path

import wandb
import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("variant", help="folder name under data/yolo_dataset/")
parser.add_argument("--config", default="experiments/train.yaml")
parser.add_argument("--epochs", type=int, help="override epochs (quick tests)")
parser.add_argument("--name", help="run name, defaults to the variant name")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
if args.epochs is not None:
    cfg["epochs"] = args.epochs
name = args.name or args.variant
if args.epochs is not None and not args.name:
    name = f"{args.variant}_e{args.epochs}"  # a quick test never overwrites the real run
data = Path("data/yolo_dataset") / args.variant / "dataset.yaml"

train_cfg = dict(cfg)
model = YOLO(cfg.pop("model"))
model.train(data=str(data), name=name, exist_ok=True, **cfg)
best = Path(model.trainer.best)

# Reload the best checkpoint, then evaluate on real images only
metrics = YOLO(best).val(data=str(data), split="test", imgsz=cfg["imgsz"], batch=cfg["batch"],
                         device=cfg["device"], name=f"{name}_test", exist_ok=True)

results = {
    "variant": args.variant,
    "weights": str(best),
    "train_cfg": train_cfg,
    "real_map50": float(metrics.box.map50),
    "real_map50_95": float(metrics.box.map),
    "real_precision": float(metrics.box.mp),
    "real_recall": float(metrics.box.mr),
    "real_map50_95_per_class": {metrics.names[int(c)]: float(metrics.box.maps[int(c)])
                                for c in metrics.box.ap_class_index},
    "real_map50_per_class": {metrics.names[int(c)]: float(metrics.box.ap50[i])
                             for i, c in enumerate(metrics.box.ap_class_index)},
}

out = Path("experiments/results") / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))

# Offline unless WANDB_MODE says otherwise, upload later with `wandb sync`
run = wandb.init(project="domain-gap-detector", name=name, config={**cfg, "variant": args.variant},
                 mode=os.environ.get("WANDB_MODE", "offline"))
run.log({k: v for k, v in results.items() if k.startswith("real_") and isinstance(v, float)})
run.finish()
