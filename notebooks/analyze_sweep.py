"""Reproduce the saved sweep figures: python -m notebooks.analyze_sweep."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

from src.gap_scorer.score import correlation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_sweep(root=ROOT):
    """Reject stale result joins and align class values by name, not JSON order."""
    report = json.loads((root / "experiments/gap_scores.json").read_text())
    rows = report["variants"]
    names = [row["variant"] for row in rows]
    if len(set(names)) != len(names) or "baseline" not in names:
        raise ValueError("Need unique variants including baseline")
    results = []
    for row in rows:
        path = root / "experiments/results" / f"{row['variant']}.json"
        raw = path.read_bytes()
        result = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != row["result_sha256"]:
            raise ValueError(f"Stale gap report: {path}")
        if result["variant"] != row["variant"] or result["train_cfg"] != report["train_cfg"]:
            raise ValueError(f"Mismatched sweep configuration: {path}")
        for metric in ("real_map50", "real_map50_95"):
            if result[metric] != row[metric] or not 0 <= row[metric] <= 1:
                raise ValueError(f"Invalid or mismatched {metric}: {path}")
        results.append(result)
    classes = list(results[names.index("baseline")]["real_map50_95_per_class"])
    if any(set(result["real_map50_95_per_class"]) != set(classes) for result in results):
        raise ValueError("Per-class results have different class coverage")
    aps = np.array([[result["real_map50_95_per_class"][name] for name in classes] for result in results])
    if not np.isfinite(aps).all() or (aps < 0).any() or (aps > 1).any():
        raise ValueError("Invalid per-class AP values")
    np.testing.assert_allclose(aps.mean(axis=1), [row["real_map50_95"] for row in rows], atol=1e-10)
    for metric in ("dinov2_fid", "dinov2_kid"):
        values = [row[metric] for row in rows]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {metric}")
        actual = correlation(values, [row["real_map50_95"] for row in rows])
        for key, value in actual.items():
            np.testing.assert_allclose(value, report["correlations"][metric]["real_map50_95"][key])
    deltas = 100 * (aps - aps[names.index("baseline")])
    return report, classes, aps, deltas


def main():
    report, classes, aps, deltas = load_sweep()
    rows = report["variants"]
    names = [row["variant"] for row in rows]
    labels = [name.replace("_", " ").capitalize() for name in names]
    maps = np.array([row["real_map50_95"] for row in rows]) * 100
    baseline = names.index("baseline")
    output = ROOT / "experiments/analysis"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    bars = ax.barh(labels, maps, color=["#64748b" if i == baseline else "#0f766e" for i in range(len(rows))])
    ax.bar_label(bars, labels=[f"{value:.2f}% ({value - maps[baseline]:+.2f} pp)" for value in maps], padding=7)
    ax.axvline(maps[baseline], color="#64748b", linestyle="--", linewidth=1)
    ax.set(xlim=(0, max(maps) + 25), xlabel="Real mAP50–95 (%)", title="Synthetic randomization → real detection performance")
    ax.invert_yaxis()
    ax.text(0, -.18, "CUDA sweep · 50 epochs · one seed · labels show change from current baseline", transform=ax.transAxes, fontsize=10)
    save(fig, "variant_map")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    colors = plt.get_cmap("tab10")(np.arange(len(rows)))
    offsets = {"baseline": (8, -18), "lighting": (8, 8), "material": (10, -20),
               "noise": (8, 8), "camera": (8, 8), "background": (8, 8), "high_randomization": (8, -18)}
    for ax, metric, title in zip(axes, ("dinov2_fid", "dinov2_kid"), ("DINOv2 FID", "DINOv2 KID")):
        values = [row[metric] for row in rows]
        ax.scatter(values, maps, c=colors, s=65, edgecolors="white", linewidths=.6)
        for row, label, x, y in zip(rows, labels, values, maps):
            offset = offsets[row["variant"]]
            if metric == "dinov2_kid" and row["variant"] in {"baseline", "lighting"}:
                offset = (20, -3) if row["variant"] == "baseline" else (12, -20)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": .6})
        stats = report["correlations"][metric]["real_map50_95"]
        ax.set(xlabel=f"{title} (lower gap ←)", ylabel="Real mAP50–95 (%)",
               title=f"{title}\nPearson r = {stats['pearson']:.3f} · Spearman ρ = {stats['spearman']:.3f}", ylim=(0, 52))
        ax.margins(x=.3)
        ax.grid(alpha=.18)
    fig.suptitle("Gap scores vs. transfer performance · seven single-seed variants")
    save(fig, "gap_vs_map")

    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    limit = float(np.abs(deltas).max())
    image = ax.imshow(deltas, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(classes)), [name[4:].replace("_", " ") for name in classes], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_title("Per-class AP50–95 change vs. current CUDA baseline (percentage points)")
    for (i, j), value in np.ndenumerate(deltas):
        ax.text(j, i, f"{value:+.1f}", ha="center", va="center", color="white" if abs(value) > .55 * limit else "#111827")
    fig.colorbar(image, ax=ax, label="Δ AP50–95 (pp) · blue = improvement", shrink=.8)
    save(fig, "per_class_delta")

    with (output / "per_class.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["variant", "class", "ap50_95_percent", "delta_vs_baseline_pp"])
        for i, name in enumerate(names):
            for j, class_name in enumerate(classes):
                writer.writerow([name, class_name, f"{100 * aps[i, j]:.6f}", f"{deltas[i, j]:.6f}"])
    print(f"Validated {len(rows)} runs / {len(classes)} classes; wrote figures and per_class.csv to {output}")


if __name__ == "__main__":
    main()
