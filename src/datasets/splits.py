from enum import Enum


class DataSplittingStrategy(Enum):
    GlobalTemporalSplit = "global_temporal"
    LeaveOneOut = "leave_one_out"


def split_leave_one_out(item_ids: list[int]) -> tuple[list[int], list[int]]:
    if len(item_ids) < 2:
        return item_ids, []
    return item_ids[:-1], [item_ids[-1]]


def split_global_temporal(
    item_ids: list[int],
    timestamps: list[int | float],
    split_timestamp: int | float,
) -> tuple[list[int], list[int]]:
    history = [iid for iid, ts in zip(item_ids, timestamps) if ts < split_timestamp]
    targets = [iid for iid, ts in zip(item_ids, timestamps) if ts >= split_timestamp]
    return history, targets


def split_sequence(
    item_ids: list[int],
    timestamps: list[int | float] | None,
    strategy: DataSplittingStrategy,
    split_timestamp: int | float | None = None,
) -> tuple[list[int], list[int]]:
    if strategy == DataSplittingStrategy.LeaveOneOut:
        return split_leave_one_out(item_ids)

    if timestamps is None:
        raise ValueError("timestamps are required for GlobalTemporalSplit")
    if split_timestamp is None:
        raise ValueError("split_timestamp is required for GlobalTemporalSplit")
    return split_global_temporal(item_ids, timestamps, split_timestamp)
