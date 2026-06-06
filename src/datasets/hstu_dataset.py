from os import PathLike
from typing import Any

import torch

from .splits import DataSplittingStrategy, split_sequence
from src.data.sequence_store import parse_float_sequence, parse_int_sequence, read_sequences


class HSTUDataset(torch.utils.data.Dataset):


    def __init__(
        self,
        ratings_file: str | PathLike[str],
        max_seq_len: int,
        splitting_strategy: DataSplittingStrategy,
        split_timestamp: int | float | None = None,
        is_train: bool = True,
        shift_id_by: int = 0,
        chronological: bool = True,
    ) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self._data = read_sequences(ratings_file)
        self._max_seq_len = max_seq_len
        self._splitting_strategy = splitting_strategy
        self._split_timestamp = split_timestamp
        self._is_train = is_train
        self._shift_id_by = shift_id_by
        self._chronological = chronological
        self._samples = self._build_samples()

    def _shift(self, values: list[int]) -> list[int]:
        if not self._shift_id_by:
            return values
        return [value + self._shift_id_by for value in values]

    def _build_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for _, row in self._data.iterrows():
            item_ids = self._shift(parse_int_sequence(row["item_ids"]))
            timestamps = (
                parse_float_sequence(row["timestamps"])
                if "timestamps" in self._data.columns
                else None
            )
            ratings = (
                parse_float_sequence(row["ratings"])
                if "ratings" in self._data.columns
                else [1.0] * len(item_ids)
            )
            raw_timestamps = timestamps or [0.0] * len(item_ids)
            history, targets = split_sequence(
                item_ids=item_ids,
                timestamps=timestamps,
                strategy=self._splitting_strategy,
                split_timestamp=self._split_timestamp,
            )
            if self._is_train:
                sequence = history
            else:
                sequence = [*history, *targets[:1]]

            if len(sequence) < 2:
                continue

            target_id = sequence[-1]
            target_pos = len(sequence) - 1
            target_rating = ratings[target_pos] if target_pos < len(ratings) else 1.0
            target_timestamp = (
                raw_timestamps[target_pos] if target_pos < len(raw_timestamps) else 0.0
            )
            historical_ids = sequence[:-1]
            historical_ratings = ratings[: len(historical_ids)]
            historical_timestamps = raw_timestamps[: len(historical_ids)]

            historical_ids = historical_ids[-self._max_seq_len :]
            historical_ratings = historical_ratings[-self._max_seq_len :]
            historical_timestamps = historical_timestamps[-self._max_seq_len :]
            history_length = len(historical_ids)

            if not self._chronological:
                historical_ids = list(reversed(historical_ids))
                historical_ratings = list(reversed(historical_ratings))
                historical_timestamps = list(reversed(historical_timestamps))

            historical_ids = historical_ids + [0] * (self._max_seq_len - history_length)
            historical_ratings = historical_ratings + [0.0] * (
                self._max_seq_len - history_length
            )
            historical_timestamps = historical_timestamps + [0.0] * (
                self._max_seq_len - history_length
            )
            samples.append(
                {
                    "user_id": int(row["user_id"]),
                    "historical_ids": torch.tensor(historical_ids, dtype=torch.long),
                    "historical_ratings": torch.tensor(
                        historical_ratings,
                        dtype=torch.float32,
                    ),
                    "historical_timestamps": torch.tensor(
                        historical_timestamps,
                        dtype=torch.float32,
                    ),
                    "history_lengths": torch.tensor(history_length, dtype=torch.long),
                    "target_ids": torch.tensor(target_id, dtype=torch.long),
                    "target_ratings": torch.tensor(
                        target_rating,
                        dtype=torch.float32,
                    ),
                    "target_timestamps": torch.tensor(
                        target_timestamp,
                        dtype=torch.float32,
                    ),
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]
