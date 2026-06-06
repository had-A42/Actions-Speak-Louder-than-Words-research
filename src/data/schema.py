from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


SequenceFormat = Literal["csv-list"]
FeatureKind = Literal["dense", "categorical", "multi_categorical", "text", "timestamp"]


@dataclass(frozen=True)
class SequenceColumns:
    user_id: str = "user_id"
    item_ids: str = "item_ids"
    ratings: str = "ratings"
    timestamps: str = "timestamps"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    source: Literal["user", "item", "interaction"]
    cardinality: int | None = None
    shape: list[int] | None = None
    description: str | None = None


@dataclass
class DatasetMetadata:
    dataset_name: str
    dataset_version: str
    sequence_format: SequenceFormat = "csv-list"
    columns: SequenceColumns = field(default_factory=SequenceColumns)
    n_users: int | None = None
    n_items: int | None = None
    min_user_interactions: int | None = None
    min_item_interactions: int | None = None
    split_name: str | None = None
    split_timestamp: float | None = None
    feedback_column: str = "rating"
    timestamp_column: str = "timestamp"
    user_features: list[FeatureSpec] = field(default_factory=list)
    item_features: list[FeatureSpec] = field(default_factory=list)
    interaction_features: list[FeatureSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetArtifacts:
    root: Path

    @property
    def sequences_csv(self) -> Path:
        return self.root / "sequences.csv"

    @property
    def metadata_json(self) -> Path:
        return self.root / "data_description.json"

    @property
    def user_mapping_csv(self) -> Path:
        return self.root / "user_mapping.csv"

    @property
    def item_mapping_csv(self) -> Path:
        return self.root / "item_mapping.csv"


def dataset_artifacts(prefix: str, base_dir: str | Path = "tmp") -> DatasetArtifacts:
    return DatasetArtifacts(root=Path(base_dir) / prefix)
