from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.hstu_dataset import (
    HSTUEvalDataset,
    HSTUTrainDataset,
    StochasticLengthSampling,
    build_train_eval_datasets,
    collate_fn,
)
from src.evaluation.metrics import evaluate_recommendations
from src.models.hstu import HSTUModel, NegativeSamplingStrategy, UserEmbeddingNorm


@dataclass
class HSTUExperimentConfig:
    max_seq_len: int = 50
    train_batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 201
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    topk: int = 100
    embedding_dim: int = 64
    num_blocks: int = 4
    num_heads: int = 4
    linear_dim: int = 16
    attention_dim: int = 16
    num_negatives: int = 512
    softmax_temperature: float = 0.05
    sampling_strategy: NegativeSamplingStrategy = "local"
    user_embedding_norm: UserEmbeddingNorm = "l2_norm"
    l2_norm_embeddings: bool = True
    l2_norm_eps: float = 1e-6
    input_dropout_rate: float = 0.5
    linear_dropout_rate: float = 0.5
    attn_dropout_rate: float = 0.0
    enable_relative_attention_bias: bool = False
    relative_attention_num_buckets: int = 128
    stochastic_length_alpha: float | None = None
    stochastic_length_sampling: StochasticLengthSampling = "random_subsequence"
    item_id_offset: int = 1
    filter_seen: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device | str,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_hstu_dataloaders(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_seq_len: int,
    train_batch_size: int,
    eval_batch_size: int,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    num_workers: int = 0,
    stochastic_length_alpha: float | None = None,
    stochastic_length_sampling: StochasticLengthSampling = "random_subsequence",
) -> Tuple[DataLoader, DataLoader, HSTUTrainDataset, HSTUEvalDataset, Dict[int, List[int]]]:
    train_dataset, eval_dataset, targets = build_train_eval_datasets(
        train=train,
        test=test,
        max_seq_len=max_seq_len,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        stochastic_length_alpha=stochastic_length_alpha,
        stochastic_length_sampling=stochastic_length_sampling,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, eval_loader, train_dataset, eval_dataset, targets


def train_hstu(
    model: HSTUModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str | torch.device,
    grad_clip_norm: Optional[float] = None,
    show_progress: bool = True,
    eval_loader: Optional[DataLoader] = None,
    targets: Optional[Dict[int, List[int]]] = None,
    catalog_size: Optional[int] = None,
    topk: Optional[int] = None,
    filter_seen: bool = True,
) -> List[float] | Tuple[List[float], List[Dict[str, float]]]:
    model.to(device)
    losses = []
    eval_metrics_history: List[Dict[str, float]] = []

    eval_requested = any(
        value is not None
        for value in (eval_loader, targets, catalog_size, topk)
    )
    if eval_requested and (
        eval_loader is None
        or targets is None
        or catalog_size is None
        or topk is None
    ):
        raise ValueError(
            "eval_loader, targets, catalog_size, and topk must all be provided "
            "to evaluate after each epoch"
        )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0

        batch_iter = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{num_epochs}",
            leave=False,
            disable=not show_progress,
        )
        for batch in batch_iter:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_size = int(batch["length"].shape[0])
            epoch_loss += float(loss.item()) * batch_size
            epoch_examples += batch_size
            batch_iter.set_postfix(loss=float(loss.item()))

        avg_loss = epoch_loss / max(epoch_examples, 1)
        losses.append(avg_loss)

        if eval_requested:
            metrics, _ = evaluate_hstu(
                model=model,
                eval_loader=eval_loader,
                targets=targets,
                catalog_size=catalog_size,
                topk=topk,
                device=device,
                filter_seen=filter_seen,
                show_progress=show_progress,
            )
            eval_metrics_history.append(metrics)
            if show_progress:
                metrics_str = ", ".join(
                    f"{metric}={value:.4f}"
                    for metric, value in sorted(metrics.items())
                )
                tqdm.write(
                    f"epoch {epoch + 1}/{num_epochs}: "
                    f"loss={avg_loss:.4f}, {metrics_str}"
                )

    if eval_requested:
        return losses, eval_metrics_history
    return losses



@torch.inference_mode()
def evaluate_hstu(
    model: HSTUModel,
    eval_loader: DataLoader,
    targets: Dict[int, List[int]],
    catalog_size: int,
    topk: int,
    device: str | torch.device,
    filter_seen: bool = True,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], Dict[int, List[int]]]:
    model.to(device)
    model.eval()

    candidates: Dict[int, List[int]] = {}
    for batch in tqdm(
        eval_loader,
        desc="HSTU evaluation",
        disable=not show_progress,
    ):
        batch = move_batch_to_device(batch, device)
        candidates.update(
            model.recommend(
                batch=batch,
                topk=topk,
                filter_seen=filter_seen,
            )
        )

    metrics = evaluate_recommendations(
        targets=targets,
        candidates=candidates,
        catalog_size=catalog_size,
        topk=topk,
    )
    return metrics, candidates
