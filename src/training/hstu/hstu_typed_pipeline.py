from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    typed_collate_fn,
)
from src.data.loaders import load_amzn_books, split_and_reindex
from src.evaluation.metrics import evaluate_recommendations
from src.models.hstu import NegativeSamplingStrategy, UserEmbeddingNorm
from src.models.hstu_typed import TypedHSTUModel
from src.training.hstu.hstu_pipeline import move_batch_to_device


@dataclass
class TypedHSTUExperimentConfig:
    max_events_len: int = 50
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
    num_workers: int = 0,
) -> Tuple[
    DataLoader,
    DataLoader,
    TypedHSTUTrainDataset,
    TypedHSTUEvalDataset,
    Dict[int, List[int]],
]:
    train_dataset, eval_dataset, targets = build_typed_train_eval_datasets(
        train=train,
        test=test,
        schema=schema,
        max_events_len=max_events_len,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=typed_collate_fn,
        drop_last=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=typed_collate_fn,
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
) -> List[float]:
    model.to(device)
    losses: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0

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

            batch_size = int(batch["token_length"].shape[0])
            epoch_loss += float(loss.item()) * batch_size
            epoch_examples += batch_size
            batch_iter.set_postfix(loss=float(loss.item()))

        losses.append(epoch_loss / max(epoch_examples, 1))

    return losses


@torch.inference_mode()
def evaluate_typed_hstu(
    model: TypedHSTUModel,
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
        desc="Typed HSTU evaluation",
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


def build_typed_hstu_model(
    num_items: int,
    schema: TypedTokenSchema,
    experiment_config: TypedHSTUExperimentConfig,
) -> TypedHSTUModel:
    max_token_seq_len = experiment_config.max_events_len * schema.tokens_per_event
    return TypedHSTUModel(
        num_items=num_items,
        embedding_dim=experiment_config.embedding_dim,
        max_token_seq_len=max_token_seq_len,
        max_event_seq_len=experiment_config.max_events_len,
        num_blocks=experiment_config.num_blocks,
        num_heads=experiment_config.num_heads,
        linear_dim=experiment_config.linear_dim,
        attention_dim=experiment_config.attention_dim,
        num_token_types=schema.num_token_types,
        feature_vocab_sizes=schema.feature_vocab_sizes,
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
    Dict[int, List[int]],
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
            num_workers=experiment_config.num_workers,
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
        show_progress=show_progress,
    )
    metrics, candidates = evaluate_typed_hstu(
        model=model,
        eval_loader=eval_loader,
        targets=targets,
        catalog_size=data_description["n_items"],
        topk=experiment_config.topk,
        device=experiment_config.device,
        filter_seen=experiment_config.filter_seen,
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
