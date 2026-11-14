"""Shared image embedding inputs for the runnable batch scripts."""

import contextlib
import sys
from pathlib import Path

from .score import features

MODEL = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def image_paths(source):
    paths = [source] if source.is_file() else sorted(source.rglob("*"))
    paths = [path for path in paths if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not paths:
        raise ValueError(f"No PNG/JPEG images found: {source}")
    return paths


def add_embedding_arguments(parser):
    parser.add_argument("images", type=Path, help="folder of incoming PNG/JPEG images (at least two)")
    parser.add_argument("--reference", type=Path, default=Path("data/yolo_dataset/test/images"),
                        help="folder of real reference images, cached automatically")
    parser.add_argument("--device", help="cpu, mps, or cuda; automatically selected if omitted")
    parser.add_argument("--batch-size", type=int, default=16)


def load_batches(args, disjoint=False):
    """Use the sweep's pinned DINOv2 model for both cached distributions."""
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not args.images.is_dir() or not args.reference.is_dir():
        raise ValueError("Incoming and reference inputs must be image folders")
    incoming_paths, reference_paths = image_paths(args.images), image_paths(args.reference)
    if min(len(incoming_paths), len(reference_paths)) < 2:
        raise ValueError("Need at least two incoming and two reference images")
    if disjoint and {p.resolve() for p in incoming_paths} & {p.resolve() for p in reference_paths}:
        raise ValueError("Drift inputs and reference must be disjoint; use held-out incoming images")
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    loaded = None

    def load_model():
        nonlocal loaded
        if loaded is None:
            processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)
            model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
            loaded = model, processor, device
        return loaded

    with contextlib.redirect_stdout(sys.stderr):
        reference, reference_key = features(args.reference, Path("data/embeddings"), MODEL,
                                             REVISION, load_model, args.batch_size)
        incoming, incoming_key = features(args.images, Path("data/embeddings"), MODEL,
                                           REVISION, load_model, args.batch_size)
    metadata = {"model": MODEL, "revision": REVISION, "pooling": "last_hidden_state_cls",
                "reference": str(args.reference.resolve()), "images": str(args.images.resolve()),
                "reference_images": len(reference), "incoming_images": len(incoming),
                "reference_cache_key": reference_key, "incoming_cache_key": incoming_key}
    return reference, incoming, metadata
