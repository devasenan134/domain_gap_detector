"""Detect objects in an image or folder: python detect.py IMAGE --weights BEST.pt."""

import argparse
import contextlib
import json
import sys
from pathlib import Path

from src.gap_scorer.runtime import image_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="PNG/JPEG image or folder")
    parser.add_argument("--weights", type=Path, default=Path("weights/high_randomization.pt"))
    parser.add_argument("--device", help="cpu, mps, or CUDA device, e.g. 0")
    parser.add_argument("--confidence", type=float, default=.25)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Checkpoint not found: {args.weights}; supply --weights with your trained best.pt")
    if not 0 <= args.confidence <= 1:
        parser.error("confidence must be between 0 and 1")
    paths = image_paths(args.images)
    from PIL import Image

    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO

        model = YOLO(args.weights)
        if model.task != "detect":
            raise ValueError("Expected an object detection checkpoint")
        for path in paths:
            with Image.open(path) as image:
                result = model.predict(image.convert("RGB"), conf=args.confidence, device=args.device,
                                       imgsz=640, verbose=False)[0]
            record = {"image": str(path.resolve()), "weights": str(args.weights.resolve()),
                      "height": result.orig_shape[0], "width": result.orig_shape[1],
                      "detections": result.summary(normalize=False), "speed_ms": result.speed}
            print(json.dumps(record, allow_nan=False), file=output)


if __name__ == "__main__":
    main()
