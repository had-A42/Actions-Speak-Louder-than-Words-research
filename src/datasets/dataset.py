from os import PathLike
from typing import Any

import pandas as pd
import torch

from .splits import DataSplittingStrategy, split_sequence
from src.data.sequence_store import parse_float_sequence, parse_int_sequence, read_sequences


class SequentialDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        ratings_file: str | PathLike[str],
        max_seq_len: int,
        splitting_strategy: DataSplittingStrategy,
        split_timestamp: int | float | None = None,
        shift_id_by: int = 0,
    ) -> None:
        super().__init__()

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if splitting_strategy == DataSplittingStrategy.GlobalTemporalSplit:
            if split_timestamp is None:
                raise ValueError(
                    "split_timestamp must be provided for GlobalTemporalSplit"
                )

        self._max_seq_len = max_seq_len
        self._splitting_strategy = splitting_strategy
        self._split_timestamp = split_timestamp
        self._shift_id_by = shift_id_by
        self._samples: list[dict[str, Any]] = []

        self._data = read_sequences(ratings_file)
        if "user_id" not in self._data.columns or "item_ids" not in self._data.columns:
            raise ValueError("ratings_file must contain user_id and item_ids columns")
        if (
            self._splitting_strategy == DataSplittingStrategy.GlobalTemporalSplit
            and "timestamps" not in self._data.columns
        ):
            raise ValueError(
                "ratings_file must contain timestamps for GlobalTemporalSplit"
            )

    @property
    def samples(self) -> list[dict[str, Any]]:
        return self._samples

    def _read_user_sequence(
        self,
        row: pd.Series,
    ) -> tuple[int, list[int], list[float] | None]:
        uid = int(row["user_id"])
        item_ids = parse_int_sequence(row["item_ids"])

        if self._shift_id_by:
            item_ids = [item_id + self._shift_id_by for item_id in item_ids]

        timestamps = None
        if self._splitting_strategy == DataSplittingStrategy.GlobalTemporalSplit:
            timestamps = parse_float_sequence(row["timestamps"])
            if len(item_ids) != len(timestamps):
                raise ValueError(
                    f"user_id={uid} has mismatched item_ids and timestamps lengths"
                )

        return uid, item_ids, timestamps

    def _split_sequence(
        self,
        item_ids: list[int],
        timestamps: list[float] | None,
    ) -> tuple[list[int], list[int]]:
        if self._splitting_strategy == DataSplittingStrategy.LeaveOneOut:
            return split_sequence(
                item_ids,
                timestamps,
                self._splitting_strategy,
            )

        return split_sequence(
            item_ids,
            timestamps,
            self._splitting_strategy,
            self._split_timestamp,  # type: ignore[arg-type]
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]


class TrainDataset(SequentialDataset):

    def __init__(
        self,
        ratings_file: str | PathLike[str],
        max_seq_len: int,
        splitting_strategy: DataSplittingStrategy = DataSplittingStrategy.GlobalTemporalSplit,
        split_timestamp: int | float | None = None,
        shift_id_by: int = 0,
    ) -> None:
        super().__init__(
            ratings_file=ratings_file,
            max_seq_len=max_seq_len,
            splitting_strategy=splitting_strategy,
            split_timestamp=split_timestamp,
            shift_id_by=shift_id_by,
        )
        self._build_samples()

    def _build_samples(self) -> None:
        for _, row in self._data.iterrows():
            uid, item_ids, timestamps = self._read_user_sequence(row)
            history_full, _ = self._split_sequence(item_ids, timestamps)

            if len(history_full) < 2:
                continue

            for start in range(0, len(history_full) - 1, self._max_seq_len):
                end = min(start + self._max_seq_len, len(history_full) - 1)
                history = history_full[start:end]
                targets = history_full[start + 1 : end + 1]

                if not history or not targets:
                    continue

                self._samples.append(
                    {
                        "uid": uid,
                        "history": history,
                        "targets": targets,
                        "length": len(history),
                    }
                )


class EvalDataset(SequentialDataset):
    def __init__(
        self,
        ratings_file: str | PathLike[str],
        max_seq_len: int,
        splitting_strategy: DataSplittingStrategy = DataSplittingStrategy.LeaveOneOut,
        split_timestamp: int | float | None = None,
        shift_id_by: int = 0,
    ) -> None:
        super().__init__(
            ratings_file=ratings_file,
            max_seq_len=max_seq_len,
            splitting_strategy=splitting_strategy,
            split_timestamp=split_timestamp,
            shift_id_by=shift_id_by,
        )
        self.targets: dict[int, list[int]] = {}
        self._build_samples()

    def _build_samples(self) -> None:
        for _, row in self._data.iterrows():
            uid, item_ids, timestamps = self._read_user_sequence(row)
            history_full, target_items = self._split_sequence(item_ids, timestamps)

            if not history_full or not target_items:
                continue

            history = history_full[-self._max_seq_len :]
            self.targets[uid] = target_items
            self._samples.append(
                {
                    "uid": uid,
                    "history": history,
                    "length": len(history),
                }
            )


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    uids = torch.tensor([sample["uid"] for sample in batch], dtype=torch.long)
    lengths = torch.tensor([sample["length"] for sample in batch], dtype=torch.long)
    histories = torch.cat(
        [torch.tensor(sample["history"], dtype=torch.long) for sample in batch]
    )

    result = {
        "uid": uids,
        "length": lengths,
        "history": histories,
    }

    if "targets" in batch[0]:
        result["targets"] = torch.cat(
            [torch.tensor(sample["targets"], dtype=torch.long) for sample in batch]
        )

    return result
