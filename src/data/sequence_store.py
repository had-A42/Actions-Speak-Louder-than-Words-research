from os import PathLike
from typing import Any, Iterable

import pandas as pd

from .schema import DatasetArtifacts, SequenceColumns


def parse_int_sequence(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    return [int(float(x)) for x in str(value).split(",") if x != ""]


def parse_float_sequence(value: Any) -> list[float]:
    if pd.isna(value):
        return []
    return [float(x) for x in str(value).split(",") if x != ""]


def serialize_sequence(values: Iterable[Any]) -> str:
    return ",".join(map(str, values))


def read_sequences(
    path: str | PathLike[str],
    columns: SequenceColumns = SequenceColumns(),
) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = [columns.user_id, columns.item_ids]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"sequence file is missing required columns: {missing}")
    return data


def write_sequences(
    sequences: pd.DataFrame,
    artifacts: DatasetArtifacts,
    sequence_columns: list[str],
) -> None:
    artifacts.root.mkdir(parents=True, exist_ok=True)
    serialized = sequences.copy()
    for column in sequence_columns:
        serialized[column] = serialized[column].apply(serialize_sequence)
    serialized.reset_index(drop=True).to_csv(artifacts.sequences_csv, index=False)


def save_index_mapping(index: pd.Index, output_path: str | PathLike[str]) -> None:
    mapping = pd.DataFrame(
        {
            "new_id": range(len(index)),
            "original_id": index.astype(str),
        }
    )
    mapping.to_csv(output_path, index=False)
