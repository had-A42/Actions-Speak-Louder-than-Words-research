import math
from typing import Any, Dict, Iterable, List, Literal, Mapping, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

StochasticLengthSampling = Literal["random_subsequence", "contiguous_window"]


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
        stochastic_length_alpha: float | None = None,
        stochastic_length_sampling: StochasticLengthSampling = "random_subsequence",
    ) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if stochastic_length_alpha is not None and not (
            1.0 < stochastic_length_alpha <= 2.0
        ):
            raise ValueError("stochastic_length_alpha must be in (1, 2]")
        if stochastic_length_sampling not in (
            "random_subsequence",
            "contiguous_window",
        ):
            raise ValueError(
                "stochastic_length_sampling must be "
                "'random_subsequence' or 'contiguous_window'"
            )

        self.max_seq_len = max_seq_len
        self.stochastic_length_alpha = stochastic_length_alpha
        self.stochastic_length_sampling = stochastic_length_sampling
        self.full_sequence_threshold = max_seq_len
        self.samples: List[Dict[str, Any]] = []
        self._sl_histories: List[Dict[str, Any]] = []

        if stochastic_length_alpha is not None:
            max_history_len = max(
                (len(history) for history in histories.values()),
                default=0,
            )
            if max_history_len < 2:
                return

            threshold = math.ceil(max_history_len ** (stochastic_length_alpha / 2.0))
            self.full_sequence_threshold = min(max_seq_len, max(2, threshold))

            for uid, history in histories.items():
                if len(history) < 2:
                    continue
                sample = {
                    "uid": uid,
                    "history": list(history),
                }
                if timestamp_histories is not None:
                    timestamps = timestamp_histories[uid]
                    if len(timestamps) != len(history):
                        raise ValueError(
                            "timestamp history length must match item history length"
                        )
                    sample["timestamps"] = list(timestamps)
                self._sl_histories.append(sample)
            return

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
        if self.stochastic_length_alpha is not None:
            return len(self._sl_histories)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.stochastic_length_alpha is not None:
            return self._sample_stochastic_length(self._sl_histories[idx])
        return self.samples[idx]

    def _sample_stochastic_length(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        history = sample["history"]
        timestamps = sample.get("timestamps")
        sampled_history, sampled_timestamps = self._sample_history(history, timestamps)

        if len(sampled_history) < 2:
            sampled_history = history[-2:]
            sampled_timestamps = timestamps[-2:] if timestamps is not None else None

        inputs = sampled_history[:-1]
        targets = sampled_history[1:]
        output = {
            "uid": sample["uid"],
            "history": inputs,
            "targets": targets,
            "length": len(inputs),
        }
        if sampled_timestamps is not None:
            output["timestamps"] = sampled_timestamps[:-1]
        return output

    def _sample_history(
        self,
        history: List[int],
        timestamps: List[int] | None,
    ) -> Tuple[List[int], List[int] | None]:
        history_len = len(history)
        threshold = self.full_sequence_threshold
        if history_len <= threshold:
            return list(history), list(timestamps) if timestamps is not None else None

        p_full = min(1.0, (threshold * threshold) / (history_len * history_len))
        if torch.rand(()) < p_full:
            full_len = min(history_len, self.max_seq_len)
            start = int(torch.randint(0, history_len - full_len + 1, ()).item())
            end = start + full_len
            sampled_timestamps = (
                timestamps[start:end] if timestamps is not None else None
            )
            return history[start:end], sampled_timestamps

        if self.stochastic_length_sampling == "contiguous_window":
            start = int(torch.randint(0, history_len - threshold + 1, ()).item())
            end = start + threshold
            sampled_timestamps = (
                timestamps[start:end] if timestamps is not None else None
            )
            return history[start:end], sampled_timestamps

        indices = torch.randperm(history_len)[:threshold].sort().values.tolist()
        sampled_history = [history[index] for index in indices]
        sampled_timestamps = (
            [timestamps[index] for index in indices] if timestamps is not None else None
        )
        return sampled_history, sampled_timestamps


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
    stochastic_length_alpha: float | None = None,
    stochastic_length_sampling: StochasticLengthSampling = "random_subsequence",
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
        stochastic_length_alpha=stochastic_length_alpha,
        stochastic_length_sampling=stochastic_length_sampling,
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
    "StochasticLengthSampling",
]
