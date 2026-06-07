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
