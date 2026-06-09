from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

import mmh3
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


PADDING_TOKEN_TYPE_ID = 0
ITEM_TOKEN_TYPE_ID = 1
YAMBDA_HASH_CARDINALITY = 65536
YAMBDA_SPARSE_FEATURES_CONFIG = {
    "item_id": (11, 22),
    "uid": (33, 44),
}
YAMBDA_MULTIVALENT_FEATURES_CONFIG = {
    "artist_ids": (55, 66),
    "album_ids": (77, 88),
}
YAMBDA_LABEL_COLUMNS = ("is_like", "is_full_play")


@dataclass(frozen=True)
class FeatureTokenSpec:
    name: str
    type_id: int
    column: str
    num_embeddings: int
    missing_token_id: int = 0
    value_offset: int = 0
    transform: Callable[[Any], int] | None = None
    max_values_per_event: int = 1

    def __post_init__(self) -> None:
        if self.type_id <= ITEM_TOKEN_TYPE_ID:
            raise ValueError("feature type_id must be greater than ITEM_TOKEN_TYPE_ID")
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if self.max_values_per_event <= 0:
            raise ValueError("max_values_per_event must be positive")

    def encode(self, value: Any) -> int:
        if _is_missing_value(value):
            return self.missing_token_id
        if self.transform is not None:
            encoded = self.transform(value)
        else:
            encoded = int(value)
        return encoded + self.value_offset

    def encode_many(self, value: Any) -> List[int]:
        if _is_missing_value(value):
            return [self.missing_token_id]

        if _is_sequence_value(value):
            values = list(value)[: self.max_values_per_event]
            if not values:
                return []
            return [self.encode(item) for item in values]

        return [self.encode(value)]


@dataclass(frozen=True)
class TypedTokenSchema:
    feature_specs: Tuple[FeatureTokenSpec, ...]
    padding_type_id: int = PADDING_TOKEN_TYPE_ID
    item_type_id: int = ITEM_TOKEN_TYPE_ID

    def __post_init__(self) -> None:
        seen_type_ids = {self.padding_type_id, self.item_type_id}
        for spec in self.feature_specs:
            if spec.type_id in seen_type_ids:
                raise ValueError(f"duplicate token type id: {spec.type_id}")
            seen_type_ids.add(spec.type_id)

    @property
    def num_token_types(self) -> int:
        max_type_id = max(
            [self.padding_type_id, self.item_type_id]
            + [spec.type_id for spec in self.feature_specs]
        )
        return max_type_id + 1

    @property
    def feature_vocab_sizes(self) -> Dict[int, int]:
        return {spec.type_id: spec.num_embeddings for spec in self.feature_specs}

    @property
    def tokens_per_event(self) -> int:
        total = 1  # Start with item token
        for spec in self.feature_specs:
            total += spec.max_values_per_event
        return total


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if _is_sequence_value(value):
        return False
    return bool(pd.isna(value))


def _is_sequence_value(value: Any) -> bool:
    return (
        not isinstance(value, (str, bytes, dict))
        and hasattr(value, "__iter__")
    )


def _normalize_id_list(value: Any) -> List[int]:
    if value is None:
        return []
    if _is_sequence_value(value):
        return [int(item) for item in value if item is not None and int(item) >= 0]
    if pd.isna(value) or int(value) < 0:
        return []
    return [int(value)]


def _normalize_id_tuple(value: Any) -> Tuple[int, ...]:
    return tuple(_normalize_id_list(value))


def multihash_values(
    values: Any,
    seeds: Tuple[int, ...],
    cardinality: int = YAMBDA_HASH_CARDINALITY,
) -> List[int]:
    hashed_tokens: List[int] = []
    for value in _normalize_id_list(values):
        hashed_tokens.extend(
            mmh3.hash(str(int(value)), seed=seed, signed=False) % cardinality
            for seed in seeds
        )
    return hashed_tokens


def multihash_id_tuple(
    values: Tuple[int, ...],
    seeds: Tuple[int, ...],
    cardinality: int = YAMBDA_HASH_CARDINALITY,
) -> List[int]:
    hashed_tokens: List[int] = []
    for value in values:
        hashed_tokens.extend(
            mmh3.hash(str(int(value)), seed=seed, signed=False) % cardinality
            for seed in seeds
        )
    return hashed_tokens


def _first_element_hash(
    values: Any,
    seeds: Tuple[int, ...],
    cardinality: int,
) -> int:
    """Hash only the first element of a list/scalar with the first seed.

    Returns a single integer token id.  If the value is missing or empty,
    returns 0 (the missing-token sentinel).
    """
    ids = _normalize_id_list(values)
    if not ids:
        return 0
    return mmh3.hash(str(ids[0]), seed=seeds[0], signed=False) % cardinality


def _cached_first_element_hash_series(
    series: pd.Series,
    seeds: Tuple[int, ...],
    cardinality: int,
    cache: Dict[Tuple[int, ...], int],
    desc: str,
    show_progress: bool,
) -> pd.Series:
    tokens: List[int] = []
    iterator = tqdm(
        series,
        total=len(series),
        desc=desc,
        disable=not show_progress,
    )
    for values in iterator:
        key = _normalize_id_tuple(values)
        hashed = cache.get(key)
        if hashed is None:
            hashed = _first_element_hash(values, seeds=seeds, cardinality=cardinality)
            cache[key] = hashed
        tokens.append(hashed)
    return pd.Series(tokens, index=series.index, dtype=int)


def _cached_multihash_series(
    series: pd.Series,
    seeds: Tuple[int, ...],
    cardinality: int,
    cache: Dict[Tuple[int, ...], List[int]],
    desc: str,
    show_progress: bool,
) -> pd.Series:
    tokens: List[List[int]] = []
    iterator = tqdm(
        series,
        total=len(series),
        desc=desc,
        disable=not show_progress,
    )
    for values in iterator:
        key = _normalize_id_tuple(values)
        hashed = cache.get(key)
        if hashed is None:
            hashed = multihash_id_tuple(
                values=key,
                seeds=seeds,
                cardinality=cardinality,
            )
            cache[key] = hashed
        tokens.append(hashed)
    return pd.Series(tokens, index=series.index, dtype=object)


def add_yambda_multihash_feature_tokens(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cardinality: int = YAMBDA_HASH_CARDINALITY,
    multivalent_features_config: Mapping[
        str,
        Tuple[int, ...],
    ] = YAMBDA_MULTIVALENT_FEATURES_CONFIG,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for column, seeds in multivalent_features_config.items():
        token_column = f"{column}_token"
        cache: Dict[Tuple[int, ...], int] = {}
        train[token_column] = _cached_first_element_hash_series(
            series=train[column],
            seeds=seeds,
            cardinality=cardinality,
            cache=cache,
            desc=f"hash train {column} (first element)",
            show_progress=show_progress,
        )
        test[token_column] = _cached_first_element_hash_series(
            series=test[column],
            seeds=seeds,
            cardinality=cardinality,
            cache=cache,
            desc=f"hash test {column} (first element)",
            show_progress=show_progress,
        )

    return train, test


def yambda_multihash_token_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cardinality: int = YAMBDA_HASH_CARDINALITY,
) -> TypedTokenSchema:
    """Build a TypedTokenSchema with exactly one token per artist and album.

    Because ``add_yambda_multihash_feature_tokens`` now stores a plain int
    (not a list) in the ``*_token`` columns, ``max_values_per_event`` is
    always 1 for both features.
    """
    return TypedTokenSchema(
        feature_specs=(
            FeatureTokenSpec(
                name="artist",
                type_id=2,
                column="artist_ids_token",
                num_embeddings=cardinality,
                max_values_per_event=1,
            ),
            FeatureTokenSpec(
                name="album",
                type_id=3,
                column="album_ids_token",
                num_embeddings=cardinality,
                max_values_per_event=1,
            ),
        )
    )


def amazon_rating_token_schema(
    rating_col: str = "feedback",
    rating_type_id: int = 2,
    max_rating: int = 5,
) -> TypedTokenSchema:
    return TypedTokenSchema(
        feature_specs=(
            FeatureTokenSpec(
                name="rating",
                type_id=rating_type_id,
                column=rating_col,
                num_embeddings=max_rating + 1,
            ),
        )
    )


def encode_half_star_rating(value: Any) -> int:
    return int(round(float(value) * 2.0))


def movielens_rating_token_schema(
    rating_col: str = "feedback",
    rating_type_id: int = 2,
    max_rating: float = 5.0,
) -> TypedTokenSchema:
    return TypedTokenSchema(
        feature_specs=(
            FeatureTokenSpec(
                name="rating",
                type_id=rating_type_id,
                column=rating_col,
                num_embeddings=int(round(max_rating * 2.0)) + 1,
                transform=encode_half_star_rating,
            ),
        )
    )


def _event_to_tokens(
    event: Mapping[str, Any],
    item_col: str,
    schema: TypedTokenSchema,
) -> Tuple[List[int], List[int]]:
    token_ids = [_event_item_id(event, item_col)]
    token_types = [schema.item_type_id]
    for spec_idx, spec in enumerate(schema.feature_specs):
        feature_token_ids = spec.encode_many(
            _event_feature_value(event, spec.column, spec_idx)
        )
        token_ids.extend(feature_token_ids)
        token_types.extend([spec.type_id] * len(feature_token_ids))
    return token_ids, token_types


def _event_item_id(event: Any, item_col: str) -> int:
    if isinstance(event, Mapping):
        return int(event[item_col])
    return int(event[0])


def _event_timestamp(event: Any, time_col: str) -> int:
    if isinstance(event, Mapping):
        return int(event[time_col])
    return int(event[1])


def _event_feature_value(event: Any, column: str, spec_idx: int) -> Any:
    if isinstance(event, Mapping):
        return event.get(column)
    return event[2][spec_idx]


def _event_labels(event: Any, label_columns: Tuple[str, ...]) -> List[float]:
    if isinstance(event, Mapping):
        return [float(event[column]) for column in label_columns]
    return [float(value) for value in event[3]]


def _build_typed_sample(
    uid: int,
    context_events: List[Any],
    target_event: Any,
    schema: TypedTokenSchema,
    item_col: str,
    time_col: str,
    label_columns: Tuple[str, ...],
) -> Dict[str, Any]:
    sequence_events = context_events + [target_event]
    token_ids: List[int] = []
    token_types: List[int] = []
    token_event_positions: List[int] = []
    token_timestamps: List[int] = []
    history_items: List[int] = []

    for local_event_pos, event in enumerate(sequence_events):
        event_token_ids, event_token_types = _event_to_tokens(
            event=event,
            item_col=item_col,
            schema=schema,
        )
        token_ids.extend(event_token_ids)
        token_types.extend(event_token_types)
        token_event_positions.extend([local_event_pos] * len(event_token_ids))
        token_timestamps.extend(
            [_event_timestamp(event, time_col)] * len(event_token_ids)
        )
        if event is not target_event:
            history_items.append(_event_item_id(event, item_col))

    return {
        "uid": int(uid),
        "token_ids": token_ids,
        "token_types": token_types,
        "token_event_positions": token_event_positions,
        "token_timestamps": token_timestamps,
        "token_length": len(token_ids),
        "supervision_token_positions": [len(token_ids) - 1],
        "targets": [_event_item_id(target_event, item_col)],
        "labels": _event_labels(target_event, label_columns),
        "history": history_items,
        "length": len(history_items),
    }


def _build_typed_chunk_sample(
    uid: int,
    history_events: List[Any],
    target_events: List[Any],
    schema: TypedTokenSchema,
    item_col: str,
    time_col: str,
    label_columns: Tuple[str, ...],
) -> Dict[str, Any]:
    if len(history_events) != len(target_events):
        raise ValueError("history_events and target_events must have the same length")
    if not history_events:
        raise ValueError("history_events must be non-empty")

    token_ids: List[int] = []
    token_types: List[int] = []
    token_event_positions: List[int] = []
    token_timestamps: List[int] = []
    supervision_token_positions: List[int] = []
    history_items: List[int] = []
    targets: List[int] = []
    labels: List[float] = []

    for local_event_pos, (history_event, target_event) in enumerate(
        zip(history_events, target_events)
    ):
        event_token_start = len(token_ids)
        event_token_ids, event_token_types = _event_to_tokens(
            event=history_event,
            item_col=item_col,
            schema=schema,
        )
        token_ids.extend(event_token_ids)
        token_types.extend(event_token_types)
        token_event_positions.extend([local_event_pos] * len(event_token_ids))
        token_timestamps.extend(
            [_event_timestamp(history_event, time_col)] * len(event_token_ids)
        )
        supervision_token_positions.append(
            event_token_start + len(event_token_ids) - 1
        )
        history_items.append(_event_item_id(history_event, item_col))
        targets.append(_event_item_id(target_event, item_col))
        labels.extend(_event_labels(target_event, label_columns))

    return {
        "uid": int(uid),
        "token_ids": token_ids,
        "token_types": token_types,
        "token_event_positions": token_event_positions,
        "token_timestamps": token_timestamps,
        "token_length": len(token_ids),
        "supervision_token_positions": supervision_token_positions,
        "targets": targets,
        "labels": labels,
        "history": history_items,
        "length": len(history_items),
    }


def _build_typed_eval_chunk_sample(
    uid: int,
    context_events: List[Any],
    target_events: List[Any],
    schema: TypedTokenSchema,
    item_col: str,
    time_col: str,
    label_columns: Tuple[str, ...],
) -> Dict[str, Any]:
    if not target_events:
        raise ValueError("target_events must be non-empty")

    sequence_events = context_events + target_events
    first_target_pos = len(context_events)
    token_ids: List[int] = []
    token_types: List[int] = []
    token_event_positions: List[int] = []
    token_timestamps: List[int] = []
    supervision_token_positions: List[int] = []
    history_items: List[int] = []
    targets: List[int] = []
    labels: List[float] = []

    for local_event_pos, event in enumerate(sequence_events):
        event_token_start = len(token_ids)
        event_token_ids, event_token_types = _event_to_tokens(
            event=event,
            item_col=item_col,
            schema=schema,
        )
        token_ids.extend(event_token_ids)
        token_types.extend(event_token_types)
        token_event_positions.extend([local_event_pos] * len(event_token_ids))
        token_timestamps.extend(
            [_event_timestamp(event, time_col)] * len(event_token_ids)
        )
        if local_event_pos < first_target_pos:
            history_items.append(_event_item_id(event, item_col))
        else:
            supervision_token_positions.append(
                event_token_start + len(event_token_ids) - 1
            )
            targets.append(_event_item_id(event, item_col))
            labels.extend(_event_labels(event, label_columns))

    return {
        "uid": int(uid),
        "token_ids": token_ids,
        "token_types": token_types,
        "token_event_positions": token_event_positions,
        "token_timestamps": token_timestamps,
        "token_length": len(token_ids),
        "supervision_token_positions": supervision_token_positions,
        "targets": targets,
        "labels": labels,
        "history": history_items,
        "length": len(history_items),
    }


def build_event_histories(
    interactions: pd.DataFrame,
    schema: TypedTokenSchema,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    extra_columns: Tuple[str, ...] = (),
    show_progress: bool = True,
) -> Dict[int, List[Tuple[Any, ...]]]:
    columns = [user_col, item_col, time_col] + [
        spec.column for spec in schema.feature_specs if spec.column in interactions.columns
    ]
    columns += [column for column in extra_columns if column in interactions.columns]
    columns = list(dict.fromkeys(columns))
    ordered = interactions[columns].sort_values(
        [user_col, time_col],
        ascending=[True, True],
    )
    item_idx = columns.index(item_col)
    time_idx = columns.index(time_col)
    feature_indices = [
        columns.index(spec.column) if spec.column in columns else None
        for spec in schema.feature_specs
    ]
    extra_indices = [
        columns.index(column) if column in columns else None
        for column in extra_columns
    ]

    histories: Dict[int, List[Tuple[Any, ...]]] = {}
    grouped = ordered.groupby(user_col, sort=False)
    for uid, user_events in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="build event histories",
        disable=not show_progress,
    ):
        histories[int(uid)] = [
            (
                row[item_idx],
                row[time_idx],
                tuple(
                    row[feature_idx] if feature_idx is not None else None
                    for feature_idx in feature_indices
                ),
                tuple(
                    row[extra_idx] if extra_idx is not None else 0.0
                    for extra_idx in extra_indices
                ),
            )
            for row in user_events.itertuples(index=False, name=None)
        ]
    return histories


def build_item_targets(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    show_progress: bool = True,
) -> Dict[int, List[int]]:
    ordered = interactions.sort_values([user_col, time_col], ascending=[True, True])
    grouped = ordered.groupby(user_col, sort=False)[item_col].apply(list)
    return {
        int(uid): [int(item) for item in items]
        for uid, items in tqdm(
            grouped.items(),
            total=len(grouped),
            desc="build item targets",
            disable=not show_progress,
        )
    }


def _collate_samples(batch_samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not batch_samples:
        raise ValueError("batch_samples must be non-empty")

    result: Dict[str, List] = {
        "uid": [],
        "token_ids": [],
        "token_types": [],
        "token_event_positions": [],
        "token_timestamps": [],
        "token_length": [],
        "history": [],
        "length": [],
    }

    has_supervision = "supervision_token_positions" in batch_samples[0]
    label_dim = 0
    if has_supervision:
        first_label_count = len(batch_samples[0].get("labels", []))
        first_supervision_count = len(batch_samples[0]["supervision_token_positions"])
        if first_supervision_count == 0:
            raise ValueError("supervised samples must contain supervision positions")
        if first_label_count % first_supervision_count != 0:
            raise ValueError("labels must be divisible by supervision positions")
        label_dim = first_label_count // first_supervision_count
        result["supervision_token_positions"] = []
        result["targets"] = []
        result["labels"] = []

    token_offset = 0
    for sample in batch_samples:
        result["uid"].append(sample["uid"])
        result["token_ids"].extend(sample["token_ids"])
        result["token_types"].extend(sample["token_types"])
        result["token_event_positions"].extend(sample["token_event_positions"])
        result["token_timestamps"].extend(sample["token_timestamps"])
        result["token_length"].append(sample["token_length"])
        result["history"].extend(sample["history"])
        result["length"].append(sample["length"])

        if has_supervision:
            result["supervision_token_positions"].extend(
                token_offset + pos for pos in sample["supervision_token_positions"]
            )
            result["targets"].extend(sample["targets"])
            result["labels"].extend(sample["labels"])
        token_offset += sample["token_length"]

    output = {
        key: torch.tensor(values, dtype=torch.long)
        for key, values in result.items()
        if key != "labels"
    }
    if "labels" in result:
        output["labels"] = torch.tensor(
            result["labels"],
            dtype=torch.float32,
        ).view(-1, label_dim)
    return output

class TypedHSTUTrainDataset(Dataset):
    """Training dataset — batch-indexed, each item is a pre-collated tensor dict.

    Usage with DataLoader::

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda batch: batch[0],
            num_workers=0,
        )
    """

    def __init__(
        self,
        histories: Mapping[int, List[Mapping[str, Any]]],
        schema: TypedTokenSchema,
        max_events_len: int = 100,
        batch_size: int = 256,
        item_col: str = "item_id",
        time_col: str = "timestamp",
        label_columns: Tuple[str, ...] = YAMBDA_LABEL_COLUMNS,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        if max_events_len <= 0:
            raise ValueError("max_events_len must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.schema = schema
        self.max_events_len = max_events_len
        self.batch_size = batch_size
        self.label_columns = label_columns
        self.item_col = item_col
        self.time_col = time_col

        # Build raw (un-collated) samples first
        raw_samples: List[Dict[str, Any]] = []
        for uid, history in tqdm(
            histories.items(),
            total=len(histories),
            desc="build typed train chunks",
            disable=not show_progress,
        ):
            if len(history) < 2:
                continue
            int_uid = int(uid)
            for start in range(0, len(history) - 1, max_events_len):
                context_events = [history[start]]
                target_part = history[start + 1 : start + 1 + max_events_len]
                if not target_part:
                    continue
                raw_samples.append(
                    _build_typed_eval_chunk_sample(
                        uid=int_uid,
                        context_events=context_events,
                        target_events=target_part,
                        schema=self.schema,
                        item_col=self.item_col,
                        time_col=self.time_col,
                        label_columns=self.label_columns,
                    )
                )

        # Pre-collate into batches so __getitem__ returns a tensor dict directly
        self._batches: List[Dict[str, torch.Tensor]] = []
        for batch_start in tqdm(
            range(0, len(raw_samples), batch_size),
            desc="collate train batches",
            disable=not show_progress,
        ):
            batch_slice = raw_samples[batch_start : batch_start + batch_size]
            self._batches.append(_collate_samples(batch_slice))

        # Keep a flat view for compatibility (e.g. iter_typed_train_targets)
        self.samples = raw_samples

    def __len__(self) -> int:
        """Number of pre-collated batches."""
        return len(self._batches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the pre-collated tensor dict for batch *idx*."""
        return self._batches[idx]


class TypedHSTUEvalDataset(Dataset):
    """Evaluation dataset — batch-indexed, each item is a pre-collated tensor dict.

    Usage with DataLoader:

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch[0],
            num_workers=0,
        )
    """

    def __init__(
        self,
        histories: Mapping[int, List[Mapping[str, Any]]],
        targets: Mapping[int, List[Mapping[str, Any]]],
        schema: TypedTokenSchema,
        max_events_len: int = 100,
        batch_size: int = 256,
        item_col: str = "item_id",
        time_col: str = "timestamp",
        label_columns: Tuple[str, ...] = YAMBDA_LABEL_COLUMNS,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        if max_events_len <= 0:
            raise ValueError("max_events_len must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.schema = schema
        self.max_events_len = max_events_len
        self.batch_size = batch_size
        self.label_columns = label_columns
        self.item_col = item_col
        self.time_col = time_col

        self.targets: Dict[int, List[Mapping[str, Any]]] = {
            int(uid): user_targets
            for uid, user_targets in targets.items()
            if user_targets
        }

        # Build raw (un-collated) samples first
        raw_samples: List[Dict[str, Any]] = []
        for uid, history in tqdm(
            histories.items(),
            total=len(histories),
            desc="build typed eval chunks",
            disable=not show_progress,
        ):
            int_uid = int(uid)
            if int_uid not in self.targets or not history:
                continue
            user_targets = self.targets[int_uid]
            for start in range(0, len(user_targets), max_events_len):
                target_part = user_targets[start : start + max_events_len]
                previous_targets = user_targets[:start]
                context_budget = max_events_len + 1 - len(target_part)
                target_context_events = previous_targets[-context_budget:]
                remaining_history_len = context_budget - len(target_context_events)
                if remaining_history_len > 0:
                    context_events = history[-remaining_history_len:] + target_context_events
                else:
                    context_events = target_context_events
                raw_samples.append(
                    _build_typed_eval_chunk_sample(
                        uid=int_uid,
                        context_events=context_events,
                        target_events=target_part,
                        schema=self.schema,
                        item_col=self.item_col,
                        time_col=self.time_col,
                        label_columns=self.label_columns,
                    )
                )

        # Pre-collate into batches so __getitem__ returns a tensor dict directly
        self._batches: List[Dict[str, torch.Tensor]] = []
        for batch_start in tqdm(
            range(0, len(raw_samples), batch_size),
            desc="collate eval batches",
            disable=not show_progress,
        ):
            batch_slice = raw_samples[batch_start : batch_start + batch_size]
            self._batches.append(_collate_samples(batch_slice))

        # Keep a flat view for compatibility
        self.samples = raw_samples

    def __len__(self) -> int:
        """Number of pre-collated batches."""
        return len(self._batches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the pre-collated tensor dict for batch *idx*."""
        return self._batches[idx]


def typed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return _collate_samples(batch)


def build_typed_train_eval_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    schema: TypedTokenSchema,
    max_events_len: int = 100,
    batch_size: int = 256,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    label_columns: Tuple[str, ...] = YAMBDA_LABEL_COLUMNS,
    show_progress: bool = True,
) -> Tuple["TypedHSTUTrainDataset", "TypedHSTUEvalDataset", Dict[int, List[Mapping[str, Any]]]]:
    histories = build_event_histories(
        interactions=train,
        schema=schema,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        extra_columns=label_columns,
        show_progress=show_progress,
    )
    test_histories = build_event_histories(
        interactions=test,
        schema=schema,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        extra_columns=label_columns,
        show_progress=show_progress,
    )
    targets = {
        uid: user_targets
        for uid, user_targets in test_histories.items()
        if uid in histories and histories[uid]
    }
    train_dataset = TypedHSTUTrainDataset(
        histories=histories,
        schema=schema,
        max_events_len=max_events_len,
        batch_size=batch_size,
        item_col=item_col,
        time_col=time_col,
        label_columns=label_columns,
        show_progress=show_progress,
    )
    eval_dataset = TypedHSTUEvalDataset(
        histories=histories,
        targets=targets,
        schema=schema,
        max_events_len=max_events_len,
        batch_size=batch_size,
        item_col=item_col,
        time_col=time_col,
        label_columns=label_columns,
        show_progress=show_progress,
    )
    return train_dataset, eval_dataset, targets


def iter_typed_train_targets(dataset: "TypedHSTUTrainDataset") -> Iterable[int]:
    for sample in dataset.samples:
        targets = sample.get("targets", [])
        if torch.is_tensor(targets):
            yield from targets.tolist()
        else:
            yield from targets


__all__ = [
    "FeatureTokenSpec",
    "ITEM_TOKEN_TYPE_ID",
    "PADDING_TOKEN_TYPE_ID",
    "TypedHSTUEvalDataset",
    "TypedHSTUTrainDataset",
    "TypedTokenSchema",
    "YAMBDA_HASH_CARDINALITY",
    "YAMBDA_LABEL_COLUMNS",
    "YAMBDA_MULTIVALENT_FEATURES_CONFIG",
    "YAMBDA_SPARSE_FEATURES_CONFIG",
    "add_yambda_multihash_feature_tokens",
    "amazon_rating_token_schema",
    "build_event_histories",
    "build_item_targets",
    "build_typed_train_eval_datasets",
    "encode_half_star_rating",
    "iter_typed_train_targets",
    "movielens_rating_token_schema",
    "multihash_values",
    "typed_collate_fn",
    "yambda_multihash_token_schema",
]
