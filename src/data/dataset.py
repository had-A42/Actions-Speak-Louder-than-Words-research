import torch
import pandas as pd

from enum import Enum
from typing import Dict, List, Optional

from .utils import parse_int_list, split_leave_one_out, split_global_temporal


class DataSplittingStrategy(Enum):
    GlobalTemporalSplit = 1
    LeaveOneOut = 2


class TrainDataset(torch.utils.data.Dataset):
    """
    Training dataset for next-item prediction.

    For each user, splits their full interaction history into chunks of at most
    `max_seq_len` items (sliding window from start to end). Each chunk produces
    one sample
    """

    def __init__(
        self,
        ratings_file: str,
        max_seq_len: int,
        splitting_strategy: DataSplittingStrategy = DataSplittingStrategy.GlobalTemporalSplit,
        split_timestamp: Optional[int] = None,
        shift_id_by: int = 0,
    ) -> None:
        super().__init__()

        if splitting_strategy == DataSplittingStrategy.GlobalTemporalSplit:
            assert split_timestamp is not None, (
                "split_timestamp must be provided for GlobalTemporalSplit"
            )

        self._max_seq_len = max_seq_len
        self._splitting_strategy = splitting_strategy
        self._split_timestamp = split_timestamp
        self._shift_id_by = shift_id_by

        df = pd.read_csv(ratings_file, delimiter=",")
        # Build flat list of (uid, history_chunk, targets_chunk) samples
        self._samples: List[Dict] = []
        self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            uid = int(row["user_id"])
            item_ids = parse_int_list(str(row["item_ids"]))

            if self._shift_id_by:
                item_ids = [x + self._shift_id_by for x in item_ids]

            # Determine history / targets split
            if self._splitting_strategy == DataSplittingStrategy.LeaveOneOut:
                history_full, _ = split_leave_one_out(item_ids)
            else:
                timestamps = [int(float(t)) for t in str(row["timestamps"]).split(",")]
                history_full, _ = split_global_temporal(
                    item_ids, timestamps, self._split_timestamp  # type: ignore[arg-type]
                )

            if len(history_full) < 2:
                continue

            step = max(1, self._max_seq_len - 1)
            for start in range(0, len(history_full) - 1, step):
                end = min(start + self._max_seq_len, len(history_full))
                chunk = history_full[start:end]
                if len(chunk) < 2:
                    break
                history = chunk[:-1]
                targets = chunk[1:]
                length = len(history)
                self._samples.append(
                    {
                        "uid": uid,
                        "history": history,
                        "targets": targets,
                        "length": length,
                    }
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        return self._samples[idx]


class EvalDataset(torch.utils.data.Dataset):
    """
    Evaluation dataset for next-item prediction.

    Produces exactly one sample per user. Only users that have at least one
    target item (according to the splitting strategy) are included.
    """

    def __init__(
        self,
        ratings_file: str,
        max_seq_len: int,
        splitting_strategy: DataSplittingStrategy = DataSplittingStrategy.LeaveOneOut,
        split_timestamp: Optional[int] = None,
        shift_id_by: int = 0,
    ) -> None:
        super().__init__()

        if splitting_strategy == DataSplittingStrategy.GlobalTemporalSplit:
            assert split_timestamp is not None, (
                "split_timestamp must be provided for GlobalTemporalSplit"
            )

        self._max_seq_len = max_seq_len
        self._splitting_strategy = splitting_strategy
        self._split_timestamp = split_timestamp
        self._shift_id_by = shift_id_by

        df = pd.read_csv(ratings_file, delimiter=",")
        self._samples: List[Dict] = []
        self.targets: Dict[int, List[int]] = {}
        self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            uid = int(row["user_id"])
            item_ids = parse_int_list(str(row["item_ids"]))

            if self._shift_id_by:
                item_ids = [x + self._shift_id_by for x in item_ids]

            if self._splitting_strategy == DataSplittingStrategy.LeaveOneOut:
                history_full, target_items = split_leave_one_out(item_ids)
            else:
                timestamps = [int(float(t)) for t in str(row["timestamps"]).split(",")]
                history_full, target_items = split_global_temporal(
                    item_ids, timestamps, self._split_timestamp  # type: ignore[arg-type]
                )

            if not target_items:
                continue

            history = history_full[-self._max_seq_len :]
            length = len(history)

            self.targets[uid] = target_items
            self._samples.append(
                {
                    "uid": uid,
                    "history": history,
                    "length": length,
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        return self._samples[idx]
