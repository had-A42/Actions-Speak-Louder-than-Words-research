from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.hstu_typed_dataset import (
    TypedHSTUEvalDataset,
    TypedHSTUTrainDataset,
    TypedTokenSchema,
    movielens_rating_token_schema,
)
from src.data.loaders import load_ml20m, split_and_reindex
from src.training.hstu.hstu_typed_pipeline import (
    TypedHSTUExperimentConfig,
    build_typed_hstu_dataloaders,
    build_typed_hstu_model,
    evaluate_typed_hstu,
    train_typed_hstu,
)


def build_movielens_typed_hstu_dataloaders(
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
    schema = schema or movielens_rating_token_schema(rating_col="feedback")
    df = load_ml20m(interactions_path, config=loader_config)
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


def run_movielens_typed_hstu_experiment(
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
    ) = build_movielens_typed_hstu_dataloaders(
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
    "build_movielens_typed_hstu_dataloaders",
    "run_movielens_typed_hstu_experiment",
]
