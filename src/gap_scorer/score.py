"""DINOv2 feature-space FID/KID and correlation with the saved detector sweep."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def distances(x, y):
    """Gaussian Frechet distance and unbiased degree-three polynomial MMD (KID)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] == 0 or x.shape[1] != y.shape[1]
            or min(len(x), len(y)) < 2 or not np.isfinite(x).all()
            or not np.isfinite(y).all()):
        raise ValueError("Need two finite feature matrices with >=2 rows and matching dimensions")
    cx, cy = np.atleast_2d(np.cov(x, rowvar=False)), np.atleast_2d(np.cov(y, rowvar=False))
    values, vectors = np.linalg.eigh(cx)
    root = (vectors * np.sqrt(values.clip(0))) @ vectors.T
    product = root @ cy @ root
    cross_trace = np.sqrt(np.linalg.eigvalsh((product + product.T) / 2).clip(0)).sum()
    fid = max(0.0, float(np.sum((x.mean(0) - y.mean(0)) ** 2)
                         + np.trace(cx) + np.trace(cy) - 2 * cross_trace))
    xx = (x @ x.T / x.shape[1] + 1) ** 3
    yy = (y @ y.T / y.shape[1] + 1) ** 3
    xy = (x @ y.T / x.shape[1] + 1) ** 3
    kid = ((xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1))
           + (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1)) - 2 * xy.mean())
    return {"dinov2_fid": fid, "dinov2_kid": float(kid)}


def ranks(values):
    values = np.asarray(values)
    return np.array([(np.count_nonzero(values < v) + (np.count_nonzero(values == v) - 1) / 2)
                     for v in values])


def correlation(x, y):
    def pearson(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    return {"pearson": pearson(x, y), "spearman": pearson(ranks(x), ranks(y))}


def embed(paths, model, processor, device, batch_size):
    import torch
    from PIL import Image

    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(device)
            batches.append(model(**inputs).last_hidden_state[:, 0].cpu().numpy())
            print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(batches)


def features(folder, cache_dir, model_id, revision, load_model, batch_size):
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) < 2:
        raise ValueError(f"Need >=2 images in {folder}; found {len(paths)}")
    manifest = {"model": model_id, "revision": revision, "pooling": "last_hidden_state_cls",
                "images": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = cache_dir / f"{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            result = saved["features"]
        print(f"Cached {folder}: {len(result)} images", flush=True)
    else:
        print(f"Embedding {folder}: {len(paths)} images", flush=True)
        model, processor, device = load_model()
        result = embed(paths, model, processor, device, batch_size)
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, features=result)
        temporary.replace(cache)
    if (result.ndim != 2 or result.shape[0] != len(paths) or result.shape[1] == 0
            or not np.isfinite(result).all()):
        raise ValueError(f"Invalid cached features: {cache}")
    return result, key


def main():
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoImageProcessor, AutoModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--data-root", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--results", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gap_scores.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--revision", help="Hugging Face model commit; resolved and recorded if omitted")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1 or len(args.variants) < 3 or len(set(args.variants)) != len(args.variants):
        parser.error("Need >=3 unique variants and a positive batch size for correlation")
    results = [json.loads((args.results / f"{v}.json").read_text()) for v in args.variants]
    if any(r["variant"] != v for v, r in zip(args.variants, results)):
        raise ValueError("Result variant does not match its filename")
    if any(r["train_cfg"] != results[0]["train_cfg"] for r in results):
        raise ValueError("Sweep training configurations differ; do not mix the Mac baseline")
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    loaded = None

    def load_model():
        nonlocal loaded
        if loaded is None:
            processor = AutoImageProcessor.from_pretrained(args.model, revision=revision)
            model = AutoModel.from_pretrained(args.model, revision=revision).to(args.device).eval()
            loaded = model, processor, args.device
        return loaded

    def get(folder):
        return features(folder, args.cache_dir, args.model, revision, load_model, args.batch_size)

    real, real_key = get(args.data_root / "test/images")
    rows = []
    for variant, result in zip(args.variants, results):
        synth, key = get(args.data_root / variant / "images/train")
        rows.append({"variant": variant, "synthetic_images": len(synth), "cache_key": key,
                     "result_sha256": hashlib.sha256((args.results / f"{variant}.json").read_bytes()).hexdigest(),
                     **distances(synth, real), "real_map50": result["real_map50"],
                     "real_map50_95": result["real_map50_95"]})
    correlations = {metric: {target: correlation([r[metric] for r in rows], [r[target] for r in rows])
                            for target in ("real_map50", "real_map50_95")}
                    for metric in ("dinov2_fid", "dinov2_kid")}
    report = {"model": args.model, "revision": revision, "pooling": "last_hidden_state_cls",
              "device": args.device, "real_images": len(real), "real_cache_key": real_key,
              "train_cfg": results[0]["train_cfg"], "variants": rows, "correlations": correlations,
              "limitations": ["DINOv2 feature distances, not standard Inception FID/KID",
                              "FID is biased at small sample sizes; KID is unbiased and may be negative",
                              "Seven single-seed variants; correlations are exploratory, not causal",
                              "Real test images also serve as the gap reference; no independent predictive validation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
