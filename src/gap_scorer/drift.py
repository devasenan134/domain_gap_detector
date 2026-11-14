"""RBF-MMD permutation test on cached image embeddings."""

import numpy as np


def drift_test(reference, incoming, permutations=199, alpha=.05, seed=0):
    x, y = np.asarray(reference, dtype=np.float64), np.asarray(incoming, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] == 0 or x.shape[1] != y.shape[1]
            or min(len(x), len(y)) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Need finite matching feature matrices with >=2 rows each")
    if not 0 < alpha < 1 or permutations < 1 or alpha < 1 / (permutations + 1):
        raise ValueError("Need enough permutations to resolve the chosen alpha")
    rng = np.random.default_rng(seed)
    # shortcut: cap each distribution at 256 images for quadratic MMD; increase for power studies.
    x = x[rng.choice(len(x), min(len(x), 256), replace=False)]
    y = y[rng.choice(len(y), min(len(y), 256), replace=False)]
    pooled = np.concatenate((x, y))
    norms = np.sum(pooled ** 2, axis=1)
    squared = (norms[:, None] + norms[None, :] - 2 * pooled @ pooled.T).clip(0)
    pairs = squared[np.triu_indices(len(pooled), 1)]
    positive = pairs[pairs > 0]
    # Pooled, label-independent bandwidth keeps the permutation test exchangeable under the null.
    sigma_squared = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-squared / (2 * sigma_squared))
    weights = np.r_[np.full(len(x), 1 / len(x)), np.full(len(y), -1 / len(y))]

    def statistic(w):
        return max(0.0, float(w @ kernel @ w))

    observed = statistic(weights)
    null = np.array([statistic(rng.permutation(weights)) for _ in range(permutations)])
    # Treat floating-point ties as ties, including identical constant batches.
    p_value = float((1 + np.count_nonzero(null >= observed - 1e-12)) / (permutations + 1))
    return {"method": "biased_rbf_mmd_squared_permutation_test", "mmd_squared": observed,
            "p_value": p_value, "alpha": alpha, "drift_detected": p_value <= alpha,
            "permutation_threshold": float(np.quantile(null, 1 - alpha, method="higher")),
            "permutations": permutations, "seed": seed, "rbf_sigma_squared": sigma_squared,
            "reference_samples_used": len(x), "incoming_samples_used": len(y)}
