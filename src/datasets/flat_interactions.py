from dataclasses import dataclass
from os import PathLike

import pandas as pd

from .splits import DataSplittingStrategy
from src.data.sequence_store import parse_float_sequence, parse_int_sequence, read_sequences


@dataclass(frozen=True)
class ClassicalSplit:
    train: pd.DataFrame
    eval: pd.DataFrame


def build_classical_interactions(
    ratings_file: str | PathLike[str],
    splitting_strategy: DataSplittingStrategy,
    split_timestamp: int | float | None = None,
) -> ClassicalSplit:
    """
    Materialize the same sequence split as flat interactions for classical baselines.
    """
    sequences = read_sequences(ratings_file)
    train_rows: list[dict[str, int | float]] = []
    eval_rows: list[dict[str, int | float]] = []

    for _, row in sequences.iterrows():
        user_id = int(row["user_id"])
        item_ids = parse_int_sequence(row["item_ids"])
        timestamps = (
            parse_float_sequence(row["timestamps"])
            if "timestamps" in sequences.columns
            else None
        )
        ratings = (
            parse_float_sequence(row["ratings"])
            if "ratings" in sequences.columns
            else [1.0] * len(item_ids)
        )
        for pos, item_id in enumerate(item_ids):
            rating = ratings[pos] if pos < len(ratings) else 1.0
            row_data = {
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating,
            }

            if splitting_strategy == DataSplittingStrategy.LeaveOneOut:
                is_eval = pos == len(item_ids) - 1
            else:
                if timestamps is None:
                    raise ValueError("timestamps are required for GlobalTemporalSplit")
                if split_timestamp is None:
                    raise ValueError(
                        "split_timestamp is required for GlobalTemporalSplit"
                    )
                is_eval = timestamps[pos] >= split_timestamp

            if is_eval:
                eval_rows.append(row_data)
            else:
                train_rows.append(row_data)

    return ClassicalSplit(
        train=pd.DataFrame(train_rows, columns=["user_id", "item_id", "rating"]),
        eval=pd.DataFrame(eval_rows, columns=["user_id", "item_id", "rating"]),
    )
