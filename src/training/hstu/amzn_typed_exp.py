from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from src.data.hstu_typed_dataset import (
    TypedHSTUEvalDataset,
    TypedHSTUTrainDataset,
    TypedTokenSchema,
    amazon_rating_token_schema,
)
from src.training.hstu.hstu_typed_pipeline import (
    TypedHSTUExperimentConfig,
    build_amazon_typed_hstu_dataloaders,
    run_amazon_typed_hstu_experiment,
)


def build_amazon_typed_hstu_experiment_dataloaders(
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
    return build_amazon_typed_hstu_dataloaders(
        interactions_path=interactions_path,
        loader_config=loader_config,
        experiment_config=experiment_config,
        schema=schema or amazon_rating_token_schema(rating_col="feedback"),
    )


def run_amazon_typed_hstu_rating_experiment(
    interactions_path: str,
    loader_config: Dict[str, Any],
    experiment_config: TypedHSTUExperimentConfig,
    show_progress: bool = True,
) -> Dict[str, Any]:
    return run_amazon_typed_hstu_experiment(
        interactions_path=interactions_path,
        loader_config=loader_config,
        experiment_config=experiment_config,
        schema=amazon_rating_token_schema(rating_col="feedback"),
        show_progress=show_progress,
    )


__all__ = [
    "build_amazon_typed_hstu_experiment_dataloaders",
    "run_amazon_typed_hstu_rating_experiment",
]
