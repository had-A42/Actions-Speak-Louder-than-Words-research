from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import mmh3
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from datasets import DatasetDict, load_dataset


@dataclass
class DatasetConfig:
    dataset_type: str = "flat"
    dataset_size: str = "50m"
    interaction_name: str = "multi_event"
    default_like_window_seconds: int = 24 * 60 * 60
    lag_seconds: int = 15 * 60


# YambdaDataset wrapper class (https://huggingface.co/datasets/yandex/yambda)
class YambdaDataset:
    INTERACTIONS = frozenset(
        ["likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"]
    )

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m",
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(
        self,
        event_type: Literal[
            "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
        ],
    ) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")

    @staticmethod
    def _download(data_dir: str, file: str) -> Dataset:
        data = load_dataset(
            "yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet"
        )
        # Returns DatasetDict; extracting the only split
        assert isinstance(data, DatasetDict)
        return data["train"]


def join_item_artist_album(
    listens: pl.DataFrame,
    artists: pl.DataFrame,
    albums: pl.DataFrame,
) -> pl.DataFrame:
    artists_agg = artists.group_by("item_id").agg(
        pl.col("artist_id").unique().sort().alias("artist_ids")
    )

    albums_agg = albums.group_by("item_id").agg(
        pl.col("album_id").unique().sort().alias("album_ids")
    )

    return listens.join(artists_agg, on="item_id", how="left").join(
        albums_agg, on="item_id", how="left"
    )


def temporal_train_test_split(
    df: pl.DataFrame,
    test_quantile: float = 0.9,
    time_column: str = "timestamp",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    split_time = df[time_column].quantile(test_quantile)

    train = df.filter(pl.col(time_column) < split_time)
    test = df.filter(pl.col(time_column) >= split_time)

    return train, test


def _normalize_id_list(row_values) -> list[int]:
    if row_values is None:
        return []

    if isinstance(row_values, list):
        return [int(x) for x in row_values if x is not None]

    if isinstance(row_values, np.ndarray):
        return [int(x) for x in row_values.tolist() if x is not None]

    return [int(row_values)]


class RankerDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        transforms: list[Callable[[Any], Any]],
        label_columns: list[str],
        dense_columns: list[str],
        sparse_columns: list[str],
        multivalent_columns: list[str],
        batch_size: int,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.df = df
        self.transforms = transforms
        self.label_columns = label_columns
        self.dense_columns = dense_columns
        self.sparse_columns = sparse_columns
        self.multivalent_columns = multivalent_columns
        self.batch_size = batch_size
        self.n = len(df)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = idx * self.batch_size
        batch_df = self.df.slice(start, self.batch_size)

        dense_df = batch_df.select(
            [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in self.dense_columns]
        )

        batch = {
            "labels": {
                col: torch.tensor(
                    batch_df[col].to_numpy(),
                    dtype=torch.float32,
                )
                for col in self.label_columns
            },
            "dense_features": torch.tensor(
                dense_df.to_numpy(),
                dtype=torch.float32,
            ),
            "sparse_features": {
                col: torch.tensor(
                    batch_df[col].to_numpy(),
                    dtype=torch.long,
                )
                for col in self.sparse_columns
            },
            "multivalent_features": {},
            "meta": {
                "timestamp": torch.tensor(
                    batch_df["timestamp"].to_numpy(),
                    dtype=torch.long,
                ),
                "uid": torch.tensor(
                    batch_df["uid"].to_numpy(),
                    dtype=torch.long,
                ),
                "item_id": torch.tensor(
                    batch_df["item_id"].to_numpy(),
                    dtype=torch.long,
                ),
            },
        }

        for col in self.multivalent_columns:
            values = []
            lengths = []

            for row_values in batch_df[col].to_list():
                ids = _normalize_id_list(row_values)
                values.extend(ids)
                lengths.append(len(ids))

            batch["multivalent_features"][col] = {
                "values": torch.tensor(values, dtype=torch.long),
                "lengths": torch.tensor(lengths, dtype=torch.long),
            }

        for transform in self.transforms:
            batch = transform(batch)

        return batch


class MultihashTransform:
    def __init__(
        self,
        sparse_features_config: dict,
        sparse_features_name: str,
        multivalent_features_config: dict,
        multivalent_features_name: str,
        cardinality: int,
    ):
        self.sparse_features_config = sparse_features_config
        self.sparse_features_name = sparse_features_name
        self.multivalent_features_config = multivalent_features_config
        self.multivalent_features_name = multivalent_features_name
        self.cardinality = cardinality

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:

        sparse_features = sample[self.sparse_features_name]
        for feature_name, seeds in self.sparse_features_config.items():
            values = sparse_features[feature_name]
            values_list = values.tolist()

            hashed = [
                [
                    mmh3.hash(str(int(v)), seed=seed, signed=False) % self.cardinality
                    for seed in seeds
                ]
                for v in values_list
            ]

            sparse_features[feature_name] = torch.tensor(
                hashed,
                dtype=torch.long,
                device=values.device,
            )

        multivalent_features = sample[self.multivalent_features_name]
        for feature_name, seeds in self.multivalent_features_config.items():
            values = multivalent_features[feature_name]["values"]
            values_list = values.tolist()

            hashed = [
                [
                    mmh3.hash(str(int(v)), seed=seed, signed=False) % self.cardinality
                    for seed in seeds
                ]
                for v in values_list
            ]

            multivalent_features[feature_name]["values"] = torch.tensor(
                hashed,
                dtype=torch.long,
                device=values.device,
            )

        return sample
