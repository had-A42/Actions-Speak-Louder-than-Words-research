from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def create_masked_tensor(
    data: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if lengths.ndim != 1:
        raise ValueError("lengths must be a 1D tensor")

    if len(lengths) == 0:
        shape = (0, 0) + tuple(data.shape[1:])
        return data.new_zeros(shape), torch.zeros(
            (0, 0),
            dtype=torch.bool,
            device=data.device,
        )

    if int(lengths.sum().item()) != data.shape[0]:
        raise ValueError("sum(lengths) must match the first dimension of data")

    sequences = torch.split(data, lengths.tolist())
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,
    )
    max_len = padded.shape[1]
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(
        1
    )
    return padded, mask


class HSTUTrainDataset(Dataset):

    def __init__(
        self,
        histories: Mapping[Any, List[int]],
        timestamp_histories: Mapping[Any, List[int]] | None = None,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, Any]] = []

        for uid, history in histories.items():
            if len(history) < 2:
                continue

            inputs = history[:-1]
            targets = history[1:]
            input_timestamps = None
            if timestamp_histories is not None:
                timestamps = timestamp_histories[uid]
                if len(timestamps) != len(history):
                    raise ValueError(
                        "timestamp history length must match item history length"
                    )
                input_timestamps = timestamps[:-1]
            for start in range(0, len(inputs), max_seq_len):
                hist_part = inputs[start : start + max_seq_len]
                target_part = targets[start : start + max_seq_len]
                if not hist_part or not target_part:
                    continue

                sample = {
                    "uid": uid,
                    "history": hist_part,
                    "targets": target_part,
                    "length": len(hist_part),
                }
                if input_timestamps is not None:
                    sample["timestamps"] = input_timestamps[start : start + max_seq_len]
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class HSTUEvalDataset(Dataset):

    def __init__(
        self,
        histories: Mapping[Any, List[int]],
        targets: Mapping[Any, List[int]],
        timestamp_histories: Mapping[Any, List[int]] | None = None,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.max_seq_len = max_seq_len
        self.targets: Dict[Any, List[int]] = {
            uid: list(user_targets)
            for uid, user_targets in targets.items()
            if user_targets
        }
        self.samples: List[Dict[str, Any]] = []

        for uid, history in histories.items():
            if uid not in self.targets or not history:
                continue

            history_tail = history[-max_seq_len:]
            sample = {
                "uid": uid,
                "history": history_tail,
                "length": len(history_tail),
            }
            if timestamp_histories is not None:
                timestamps = timestamp_histories[uid]
                if len(timestamps) != len(history):
                    raise ValueError(
                        "timestamp history length must match item history length"
                    )
                sample["timestamps"] = timestamps[-max_seq_len:]
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must be non-empty")

    result: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()

    for key in keys:
        values: List[Any] = []
        if isinstance(batch[0][key], list):
            for sample in batch:
                values.extend(sample[key])
        else:
            for sample in batch:
                values.append(sample[key])
        result[key] = torch.tensor(values, dtype=torch.long)

    return result


def build_histories(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> Dict[int, List[int]]:
    ordered = interactions.sort_values([user_col, time_col], ascending=[True, True])
    grouped = ordered.groupby(user_col, sort=False)[item_col].apply(list)
    return grouped.to_dict()


def build_timestamp_histories(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> Dict[int, List[int]]:
    ordered = interactions.sort_values([user_col, time_col], ascending=[True, True])
    grouped = ordered.groupby(user_col, sort=False)[time_col].apply(list)
    return {
        int(uid): [int(ts) for ts in timestamps]
        for uid, timestamps in grouped.items()
    }


def build_targets(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> Dict[int, List[int]]:
    return build_histories(
        interactions=interactions,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
    )


def build_train_eval_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_seq_len: int = 100,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> Tuple[HSTUTrainDataset, HSTUEvalDataset, Dict[int, List[int]]]:
    histories = build_histories(
        interactions=train,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
    )
    timestamp_histories = build_timestamp_histories(
        interactions=train,
        user_col=user_col,
        time_col=time_col,
    )
    raw_targets = build_targets(
        interactions=test,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
    )
    targets = {
        uid: user_targets
        for uid, user_targets in raw_targets.items()
        if uid in histories and histories[uid]
    }

    train_dataset = HSTUTrainDataset(
        histories=histories,
        timestamp_histories=timestamp_histories,
        max_seq_len=max_seq_len,
    )
    eval_dataset = HSTUEvalDataset(
        histories=histories,
        targets=targets,
        timestamp_histories=timestamp_histories,
        max_seq_len=max_seq_len,
    )
    return train_dataset, eval_dataset, targets

def iter_train_targets(dataset: HSTUTrainDataset) -> Iterable[int]:
    for idx in range(len(dataset)):
        yield from dataset[idx]["targets"]


__all__ = [
    "HSTUTrainDataset",
    "HSTUEvalDataset",
    "build_histories",
    "build_timestamp_histories",
    "build_targets",
    "build_train_eval_datasets",
    "build_q_from_train_targets",
    "collate_fn",
    "create_masked_tensor",
    "iter_train_targets",
]