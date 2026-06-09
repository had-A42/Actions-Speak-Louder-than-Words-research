from __future__ import annotations

import numpy as np


def compute_normalized_entropy(
    labels: np.ndarray,
    logits: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    - E-task NE is computed on `is_like`.
    - C-task NE is computed on `is_full_play`.
    """
    y = np.asarray(labels, dtype=np.float64)

    z = np.asarray(logits, dtype=np.float64)
    logloss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))

    model_logloss = float(logloss.mean())
    p_base = float(np.clip(y.mean(), eps, 1.0 - eps))
    baseline_entropy = -(
        p_base * np.log(p_base) + (1.0 - p_base) * np.log(1.0 - p_base)
    )
    return float(model_logloss / baseline_entropy)
