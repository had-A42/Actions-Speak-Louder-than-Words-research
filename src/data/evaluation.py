from typing import List, Dict, Callable
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import DataLoader

def get_metrics(targets: List[int], candidates: List[int], topk: int) -> Dict[str, float]:
    candidates = np.asarray(candidates[:topk])
    targets = np.asarray(targets)
    gu_size = len(np.unique(targets))

    hits = np.isin(candidates, targets).astype(np.int32)

    hr = float(hits.sum() > 0)
    recall = hits.sum() / min(topk, gu_size)

    weights = 1 / np.log2(np.arange(2, topk + 2))
    ndcg = (hits[:topk] * weights[:topk]).sum() / weights[:min(topk, gu_size)].sum()

    return {'hitrate': hr,
            'recall': recall,
            'ndcg': ndcg}


def evaluate(
    targets: Dict[int, List[int]],
    candidates: Dict[int, List[int]],
    catalog_size: int,
    topk: int = 100,
) -> Dict[str, float]:

    metrics_sum = [0, 0, 0]
    items = set()

    for uid, targets in targets.items():
        items.update(candidates[uid])

        um = get_metrics(targets, candidates[uid], topk)
        for k, v in enumerate(um.values()):
            metrics_sum[k] += v

    metrics = defaultdict(float,
     {'hitrate': metrics_sum[0] / len(candidates),
      'recall': metrics_sum[1] / len(candidates),
      'ndcg': metrics_sum[2] / len(candidates),
      'coverage': len(items) / catalog_size})

    return metrics

def eval(
    dataloader: DataLoader,
    model: torch.nn.Module,
    catalog_size: int,
    topk: int,
    device: str = "cuda",
    *,
    targets: Dict[int, List[int]],
    evaluate_fn: Callable[..., Dict[str, float]],
) -> Dict[str, float]:

    model.eval()
    predictions = {}

    with torch.inference_mode():
        for batch in dataloader:
            uids = batch['uid'].tolist()
            batch = {k: v.to(device) for k, v in batch.items()}

            scores = model(batch)
            topk_items = torch.topk(scores, k=topk, dim=-1).indices

            for uid, items in zip(uids, topk_items.cpu().tolist()):
                predictions[uid] = items

    return evaluate_fn(
        targets=targets,
        candidates=predictions,
        catalog_size=catalog_size,
        topk=topk
    )