from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.hstu_typed_dataset import (
    TypedHSTUEvalDataset,
    TypedHSTUTrainDataset,
    TypedTokenSchema,
    amazon_rating_token_schema,
    build_typed_train_eval_datasets,
)
from src.data.loaders import load_amzn_books, split_and_reindex
from src.evaluation.metrics import compute_normalized_entropy
from src.HSTU.models.hstu import NegativeSamplingStrategy, UserEmbeddingNorm
from src.HSTU.models.hstu_typed import TypedHSTUModel
from src.HSTU.training.hstu.hstu_pipeline import move_batch_to_device


@dataclass
class TypedHSTUExperimentConfig:
    max_events_len: int = 50
    train_batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 201
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    topk: int = 100
    output_size: int = 2
    label_columns: Tuple[str, ...] = ("is_like", "is_full_play")
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
    enable_relative_attention_bias: bool = True
    relative_attention_num_buckets: int = 128
    item_id_offset: int = 1
    filter_seen: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_typed_hstu_dataloaders(
    train: pd.DataFrame,
    test: pd.DataFrame,
    schema: TypedTokenSchema,
    max_events_len: int,
    train_batch_size: int,
    eval_batch_size: int,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    label_columns: Tuple[str, ...] = ("is_like", "is_full_play"),
    num_workers: int = 0,
    show_progress: bool = True,
) -> Tuple[
    DataLoader,
    DataLoader,
    TypedHSTUTrainDataset,
    TypedHSTUEvalDataset,
    Dict[int, List[Dict[str, Any]]],
]:
    train_dataset, eval_dataset, targets = build_typed_train_eval_datasets(
        train=train,
        test=test,
        schema=schema,
        max_events_len=max_events_len,
        batch_size=train_batch_size,  # train batch size used for both datasets
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        label_columns=label_columns,
        show_progress=show_progress,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
        drop_last=False,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, eval_loader, train_dataset, eval_dataset, targets


def train_typed_hstu(
    model: TypedHSTUModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str | torch.device,
    grad_clip_norm: Optional[float] = None,
    show_progress: bool = True,
    eval_loader: Optional[DataLoader] = None,
    label_columns: Tuple[str, ...] = ("is_like", "is_full_play"),
) -> List[float] | Tuple[List[float], List[Dict[str, float]]]:
    model.to(device)
    losses: List[float] = []
    eval_metrics_history: List[Dict[str, float]] = []

    eval_requested = eval_loader is not None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_events = 0

        batch_iter = tqdm(
            train_loader,
            desc=f"typed epoch {epoch + 1}/{num_epochs}",
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

            event_count = int(batch["labels"].shape[0])
            epoch_loss += float(loss.item()) * event_count
            epoch_events += event_count
            batch_iter.set_postfix(loss=float(loss.item()))

        avg_loss = epoch_loss / max(epoch_events, 1)
        losses.append(avg_loss)

        if eval_requested:
            metrics, _ = evaluate_typed_hstu(
                model=model,
                eval_loader=eval_loader,
                device=device,
                label_columns=label_columns,
                show_progress=show_progress,
            )
            eval_metrics_history.append(metrics)
            if show_progress:
                metrics_str = ", ".join(
                    f"{metric}={value:.4f}"
                    for metric, value in sorted(metrics.items())
                )
                tqdm.write(
                    f"typed epoch {epoch + 1}/{num_epochs}: "
                    f"loss={avg_loss:.4f}, {metrics_str}"
                )

    if eval_requested:
        return losses, eval_metrics_history
    return losses


@torch.inference_mode()
def evaluate_typed_hstu(
    model: TypedHSTUModel,
    eval_loader: DataLoader,
    device: str | torch.device,
    label_columns: Tuple[str, ...] = ("is_like", "is_full_play"),
    show_progress: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.to(device)
    model.eval()

    all_labels: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    for batch in tqdm(
        eval_loader,
        desc="Typed HSTU NE evaluation",
        disable=not show_progress,
    ):
        batch = move_batch_to_device(batch, device)
        logits = model.predict_logits(batch)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())

    labels = np.concatenate(all_labels, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    if logits.shape[1] != len(label_columns):
        raise ValueError(
            f"logits width {logits.shape[1]} does not match "
            f"label_columns length {len(label_columns)}"
        )

    metrics = {
        "ne_e_task": compute_normalized_entropy(
            labels=labels[:, 0],
            logits=logits[:, 0],
        ),
    }
    if len(label_columns) > 1:
        metrics["ne_c_task"] = compute_normalized_entropy(
            labels=labels[:, 1],
            logits=logits[:, 1],
        )
    return metrics, {"labels": labels, "logits": logits}


def build_typed_hstu_model(
    num_items: int,
    schema: TypedTokenSchema,
    experiment_config: TypedHSTUExperimentConfig,
) -> TypedHSTUModel:
    max_event_seq_len = experiment_config.max_events_len + 1
    max_token_seq_len = max_event_seq_len * schema.tokens_per_event
    return TypedHSTUModel(
        num_items=num_items,
        embedding_dim=experiment_config.embedding_dim,
        max_token_seq_len=max_token_seq_len,
        max_event_seq_len=max_event_seq_len,
        num_blocks=experiment_config.num_blocks,
        num_heads=experiment_config.num_heads,
        linear_dim=experiment_config.linear_dim,
        attention_dim=experiment_config.attention_dim,
        num_token_types=schema.num_token_types,
        feature_vocab_sizes=schema.feature_vocab_sizes,
        output_size=experiment_config.output_size,
        num_negatives=experiment_config.num_negatives,
        softmax_temperature=experiment_config.softmax_temperature,
        sampling_strategy=experiment_config.sampling_strategy,
        user_embedding_norm=experiment_config.user_embedding_norm,
        l2_norm_embeddings=experiment_config.l2_norm_embeddings,
        l2_norm_eps=experiment_config.l2_norm_eps,
        item_id_offset=experiment_config.item_id_offset,
        input_dropout_rate=experiment_config.input_dropout_rate,
        linear_dropout_rate=experiment_config.linear_dropout_rate,
        attn_dropout_rate=experiment_config.attn_dropout_rate,
        enable_relative_attention_bias=experiment_config.enable_relative_attention_bias,
        relative_attention_num_buckets=experiment_config.relative_attention_num_buckets,
    )


def build_amazon_typed_hstu_dataloaders(
    interactions_path: str,
    loader_config: Dict[str, Any],
    experiment_config: TypedHSTUExperimentConfig,
    schema: Optional[TypedTokenSchema] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    TypedHSTUTrainDataset,
    TypedHSTUEvalDataset,
    Dict[int, List[Dict[str, Any]]],
    Dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    TypedTokenSchema,
]:
    schema = schema or amazon_rating_token_schema(rating_col="feedback")
    df = load_amzn_books(interactions_path, config=loader_config)
    train, test, data_description = split_and_reindex(df, config=loader_config)
    train_loader, eval_loader, train_dataset, eval_dataset, targets = (
        build_typed_hstu_dataloaders(
            train=train,
            test=test,
            schema=schema,
            max_events_len=experiment_config.max_events_len,
            train_batch_size=experiment_config.train_batch_size,
            eval_batch_size=experiment_config.eval_batch_size,
            user_col=data_description["users"],
            item_col=data_description["items"],
            time_col=data_description["timestamp"],
            label_columns=experiment_config.label_columns,
            num_workers=experiment_config.num_workers,
            show_progress=True,
        )
    )
    return (
        train_loader,
        eval_loader,
        train_dataset,
        eval_dataset,
        targets,
        data_description,
        train,
        test,
        schema,
    )


def run_amazon_typed_hstu_experiment(
    interactions_path: str,
    loader_config: Dict[str, Any],
    experiment_config: TypedHSTUExperimentConfig,
    schema: Optional[TypedTokenSchema] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    (
        train_loader,
        eval_loader,
        train_dataset,
        eval_dataset,
        targets,
        data_description,
        train,
        test,
        schema,
    ) = build_amazon_typed_hstu_dataloaders(
        interactions_path=interactions_path,
        loader_config=loader_config,
        experiment_config=experiment_config,
        schema=schema,
    )

    model = build_typed_hstu_model(
        num_items=data_description["n_items"],
        schema=schema,
        experiment_config=experiment_config,
    )
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
    )
    losses = train_typed_hstu(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=experiment_config.num_epochs,
        device=experiment_config.device,
        eval_loader=eval_loader,
        label_columns=experiment_config.label_columns,
        show_progress=show_progress,
    )
    metrics, candidates = evaluate_typed_hstu(
        model=model,
        eval_loader=eval_loader,
        device=experiment_config.device,
        label_columns=experiment_config.label_columns,
        show_progress=show_progress,
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "losses": losses,
        "metrics": metrics,
        "candidates": candidates,
        "targets": targets,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_description": data_description,
        "train": train,
        "test": test,
        "schema": schema,
    }


__all__ = [
    "TypedHSTUExperimentConfig",
    "build_amazon_typed_hstu_dataloaders",
    "build_typed_hstu_dataloaders",
    "build_typed_hstu_model",
    "evaluate_typed_hstu",
    "run_amazon_typed_hstu_experiment",
    "train_typed_hstu",
]
