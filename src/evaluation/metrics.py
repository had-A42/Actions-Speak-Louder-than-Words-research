from collections import defaultdict
from typing import Dict, List, Mapping

import numpy as np


def ranking_metrics_at_k(
    targets: List[int],
    candidates: List[int],
    topk: int,
) -> Dict[str, float]:
    if topk <= 0:
        raise ValueError("topk must be positive")
    if not targets:
        raise ValueError("targets must be non-empty")

    candidates_at_k = candidates[:topk]
    if len(candidates_at_k) < topk:
        candidates_at_k = candidates_at_k + [-1] * (topk - len(candidates_at_k))

    hits = np.isin(candidates_at_k, targets)
    ideal_len = min(len(targets), topk)

    dcg = (hits / np.log2(np.arange(2, topk + 2))).sum()
    idcg = (np.ones(ideal_len) / np.log2(np.arange(2, ideal_len + 2))).sum()

    return {
        "hitrate": float(hits.sum() > 0),
        "recall": float(hits.sum() / ideal_len),
        "ndcg": float(dcg / idcg) if idcg > 0 else 0.0,
    }


def evaluate_recommendations(
    targets: Mapping[int, List[int]],
    candidates: Mapping[int, List[int]],
    catalog_size: int,
    topk: int = 100,
) -> Dict[str, float]:
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    if not targets:
        raise ValueError("targets must be non-empty")

    total_metrics: Dict[str, float] = defaultdict(float)
    unique_items = set()

    for uid, user_targets in targets.items():
        if uid not in candidates:
            raise KeyError(f"uid {uid} is present in targets but missing in candidates")

        user_candidates = candidates[uid]
        unique_items.update(user_candidates[:topk])

        for metric, value in ranking_metrics_at_k(
            targets=user_targets,
            candidates=user_candidates,
            topk=topk,
        ).items():
            total_metrics[metric] += value

    num_users = len(targets)
    result = {
        metric: value / num_users
        for metric, value in total_metrics.items()
    }
    result["coverage"] = len(unique_items) / catalog_size
    return result


def compute_normalized_entropy(
    labels: np.ndarray,
    logits: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """
    - E-task NE is computed on `is_like`.
    - C-task NE is computed on `is_full_play`.
    """
    y = np.asarray(labels, dtype=np.float64)

    if logits is not None:
        z = np.asarray(logits, dtype=np.float64)
        logloss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    elif probs is not None:
        p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
        logloss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    else:
        raise ValueError("Pass either logits or probs")

    model_logloss = float(logloss.mean())
    p_base = float(np.clip(y.mean(), eps, 1.0 - eps))
    baseline_entropy = -(p_base * np.log(p_base) + (1.0 - p_base) * np.log(1.0 - p_base))
    return float(model_logloss / baseline_entropy)