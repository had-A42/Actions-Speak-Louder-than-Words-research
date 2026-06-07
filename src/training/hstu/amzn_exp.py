from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.hstu_dataset import HSTUEvalDataset, HSTUTrainDataset
from src.data.loaders import load_amzn_books, split_and_reindex
from src.models.hstu import HSTUModel
from src.training.hstu.hstu_pipeline import (
    HSTUExperimentConfig,
    build_hstu_dataloaders,
    evaluate_hstu,
    train_hstu,
)


def build_amazon_hstu_dataloaders(
    interactions_path: str,
    loader_config: Dict[str, Any],
    experiment_config: HSTUExperimentConfig,
) -> Tuple[
    DataLoader,
    DataLoader,
    HSTUTrainDataset,
    HSTUEvalDataset,
    Dict[int, List[int]],
    Dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    df = load_amzn_books(interactions_path, config=loader_config)
    train, test, data_description = split_and_reindex(df, config=loader_config)
    train_loader, eval_loader, train_dataset, eval_dataset, targets = build_hstu_dataloaders(
        train=train,
        test=test,
        max_seq_len=experiment_config.max_seq_len,
        train_batch_size=experiment_config.train_batch_size,
        eval_batch_size=experiment_config.eval_batch_size,
        user_col=data_description["users"],
        item_col=data_description["items"],
        time_col=data_description["timestamp"],
        num_workers=experiment_config.num_workers,
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
    )


def run_amazon_hstu_experiment(
    interactions_path: str,
    loader_config: Dict[str, Any],
    experiment_config: HSTUExperimentConfig,
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
    ) = build_amazon_hstu_dataloaders(
        interactions_path=interactions_path,
        loader_config=loader_config,
        experiment_config=experiment_config,
    )

    model = HSTUModel(
        num_items=data_description["n_items"],
        embedding_dim=experiment_config.embedding_dim,
        max_seq_len=experiment_config.max_seq_len,
        num_blocks=experiment_config.num_blocks,
        num_heads=experiment_config.num_heads,
        linear_dim=experiment_config.linear_dim,
        attention_dim=experiment_config.attention_dim,
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
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
    )
    losses = train_hstu(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=experiment_config.num_epochs,
        device=experiment_config.device,
        show_progress=show_progress,
    )
    metrics, candidates = evaluate_hstu(
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
    }
