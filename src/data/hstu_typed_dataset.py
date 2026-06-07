from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


PADDING_TOKEN_TYPE_ID = 0
ITEM_TOKEN_TYPE_ID = 1


@dataclass(frozen=True)
class FeatureTokenSpec:
    name: str
    type_id: int
    column: str
    num_embeddings: int
    missing_token_id: int = 0
    value_offset: int = 0
    transform: Optional[Callable[[Any], int]] = None

    def __post_init__(self) -> None:
        if self.type_id <= ITEM_TOKEN_TYPE_ID:
            raise ValueError("feature type_id must be greater than ITEM_TOKEN_TYPE_ID")
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")

    def encode(self, value: Any) -> int:
        if pd.isna(value):
            return self.missing_token_id
        if self.transform is not None:
            encoded = self.transform(value)
        else:
            encoded = int(value)
        return encoded + self.value_offset


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
        return 1 + len(self.feature_specs)


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
    token_ids = [int(event[item_col])]
    token_types = [schema.item_type_id]
    for spec in schema.feature_specs:
        token_ids.append(spec.encode(event.get(spec.column)))
        token_types.append(spec.type_id)
    return token_ids, token_types


def build_event_histories(
    interactions: pd.DataFrame,
    schema: TypedTokenSchema,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> Dict[int, List[Dict[str, Any]]]:
    columns = [user_col, item_col, time_col] + [
        spec.column for spec in schema.feature_specs if spec.column in interactions.columns
    ]
    ordered = interactions[columns].sort_values(
        [user_col, time_col],
        ascending=[True, True],
    )
    histories: Dict[int, List[Dict[str, Any]]] = {}
    for uid, user_events in ordered.groupby(user_col, sort=False):
        histories[int(uid)] = user_events.to_dict("records")
    return histories


def build_item_targets(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> Dict[int, List[int]]:
    ordered = interactions.sort_values([user_col, time_col], ascending=[True, True])
    grouped = ordered.groupby(user_col, sort=False)[item_col].apply(list)
    return {int(uid): [int(item) for item in items] for uid, items in grouped.items()}


class TypedHSTUTrainDataset(Dataset):

    def __init__(
        self,
        histories: Mapping[int, List[Mapping[str, Any]]],
        schema: TypedTokenSchema,
        max_events_len: int = 100,
        item_col: str = "item_id",
        time_col: str = "timestamp",
    ) -> None:
        super().__init__()
        if max_events_len <= 0:
            raise ValueError("max_events_len must be positive")

        self.schema = schema
        self.max_events_len = max_events_len
        self.samples: List[Dict[str, Any]] = []

        for uid, history in histories.items():
            if len(history) < 2:
                continue

            inputs = history[:-1]
            target_events = history[1:]
            for start in range(0, len(inputs), max_events_len):
                input_part = inputs[start : start + max_events_len]
                target_part = target_events[start : start + max_events_len]
                if not input_part or not target_part:
                    continue

                token_ids: List[int] = []
                token_types: List[int] = []
                token_event_positions: List[int] = []
                token_timestamps: List[int] = []
                supervision_positions: List[int] = []
                targets: List[int] = []
                history_items: List[int] = []

                for local_event_pos, (event, target_event) in enumerate(
                    zip(input_part, target_part)
                ):
                    event_token_ids, event_token_types = _event_to_tokens(
                        event=event,
                        item_col=item_col,
                        schema=schema,
                    )
                    token_start = len(token_ids)
                    token_ids.extend(event_token_ids)
                    token_types.extend(event_token_types)
                    token_event_positions.extend(
                        [local_event_pos] * len(event_token_ids)
                    )
                    token_timestamps.extend(
                        [int(event[time_col])] * len(event_token_ids)
                    )
                    supervision_positions.append(token_start + len(event_token_ids) - 1)
                    targets.append(int(target_event[item_col]))
                    history_items.append(int(event[item_col]))

                self.samples.append(
                    {
                        "uid": int(uid),
                        "token_ids": token_ids,
                        "token_types": token_types,
                        "token_event_positions": token_event_positions,
                        "token_timestamps": token_timestamps,
                        "token_length": len(token_ids),
                        "supervision_token_positions": supervision_positions,
                        "targets": targets,
                        "history": history_items,
                        "length": len(history_items),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class TypedHSTUEvalDataset(Dataset):
    """Evaluation samples: one user and a typed-token tail of known history."""

    def __init__(
        self,
        histories: Mapping[int, List[Mapping[str, Any]]],
        targets: Mapping[int, List[int]],
        schema: TypedTokenSchema,
        max_events_len: int = 100,
        item_col: str = "item_id",
        time_col: str = "timestamp",
    ) -> None:
        super().__init__()
        if max_events_len <= 0:
            raise ValueError("max_events_len must be positive")

        self.schema = schema
        self.max_events_len = max_events_len
        self.targets: Dict[int, List[int]] = {
            int(uid): list(user_targets)
            for uid, user_targets in targets.items()
            if user_targets
        }
        self.samples: List[Dict[str, Any]] = []

        for uid, history in histories.items():
            if uid not in self.targets or not history:
                continue

            history_tail = history[-max_events_len:]
            token_ids: List[int] = []
            token_types: List[int] = []
            token_event_positions: List[int] = []
            token_timestamps: List[int] = []
            history_items: List[int] = []
            for local_event_pos, event in enumerate(history_tail):
                event_token_ids, event_token_types = _event_to_tokens(
                    event=event,
                    item_col=item_col,
                    schema=schema,
                )
                token_ids.extend(event_token_ids)
                token_types.extend(event_token_types)
                token_event_positions.extend([local_event_pos] * len(event_token_ids))
                token_timestamps.extend([int(event[time_col])] * len(event_token_ids))
                history_items.append(int(event[item_col]))

            self.samples.append(
                {
                    "uid": int(uid),
                    "token_ids": token_ids,
                    "token_types": token_types,
                    "token_event_positions": token_event_positions,
                    "token_timestamps": token_timestamps,
                    "token_length": len(token_ids),
                    "history": history_items,
                    "length": len(history_items),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def typed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must be non-empty")

    result: Dict[str, List[int]] = {
        "uid": [],
        "token_ids": [],
        "token_types": [],
        "token_event_positions": [],
        "token_timestamps": [],
        "token_length": [],
        "history": [],
        "length": [],
    }
    has_supervision = "supervision_token_positions" in batch[0]
    if has_supervision:
        result["supervision_token_positions"] = []
        result["targets"] = []

    token_offset = 0
    for sample in batch:
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
        token_offset += sample["token_length"]

    return {key: torch.tensor(values, dtype=torch.long) for key, values in result.items()}


def build_typed_train_eval_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    schema: TypedTokenSchema,
    max_events_len: int = 100,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> Tuple[TypedHSTUTrainDataset, TypedHSTUEvalDataset, Dict[int, List[int]]]:
    histories = build_event_histories(
        interactions=train,
        schema=schema,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
    )
    raw_targets = build_item_targets(
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
    train_dataset = TypedHSTUTrainDataset(
        histories=histories,
        schema=schema,
        max_events_len=max_events_len,
        item_col=item_col,
        time_col=time_col,
    )
    eval_dataset = TypedHSTUEvalDataset(
        histories=histories,
        targets=targets,
        schema=schema,
        max_events_len=max_events_len,
        item_col=item_col,
        time_col=time_col,
    )
    return train_dataset, eval_dataset, targets


def iter_typed_train_targets(dataset: TypedHSTUTrainDataset) -> Iterable[int]:
    for idx in range(len(dataset)):
        yield from dataset[idx]["targets"]


__all__ = [
    "FeatureTokenSpec",
    "ITEM_TOKEN_TYPE_ID",
    "PADDING_TOKEN_TYPE_ID",
    "TypedHSTUEvalDataset",
    "TypedHSTUTrainDataset",
    "TypedTokenSchema",
    "amazon_rating_token_schema",
    "build_event_histories",
    "build_item_targets",
    "build_typed_train_eval_datasets",
    "encode_half_star_rating",
    "iter_typed_train_targets",
    "movielens_rating_token_schema",
    "typed_collate_fn",
]
