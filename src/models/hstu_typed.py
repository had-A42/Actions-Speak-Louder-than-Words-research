import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.hstu_dataset import create_masked_tensor
from src.data.hstu_typed_dataset import ITEM_TOKEN_TYPE_ID, PADDING_TOKEN_TYPE_ID
from src.models.hstu import (
    HSTUBlock,
    NegativeSamplingStrategy,
    RelativeBucketedTimeAndPositionBasedBias,
    UserEmbeddingNorm,
)


def _truncated_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class TypedTokenEmbedding(nn.Module):

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_token_types: int,
        feature_vocab_sizes: Dict[int, int],
        item_id_offset: int = 1,
        item_token_type_id: int = ITEM_TOKEN_TYPE_ID,
        padding_token_type_id: int = PADDING_TOKEN_TYPE_ID,
    ) -> None:
        super().__init__()
        if num_token_types <= item_token_type_id:
            raise ValueError("num_token_types must include the item token type")

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_token_types = num_token_types
        self.item_id_offset = item_id_offset
        self.item_token_type_id = item_token_type_id
        self.padding_token_type_id = padding_token_type_id

        self.item_embeddings = nn.Embedding(
            num_items + item_id_offset,
            embedding_dim,
            padding_idx=0,
        )
        self.feature_embeddings = nn.ModuleDict(
            {
                str(type_id): nn.Embedding(
                    num_embeddings,
                    embedding_dim,
                    padding_idx=0,
                )
                for type_id, num_embeddings in feature_vocab_sizes.items()
            }
        )
        self.type_embeddings = nn.Embedding(
            num_token_types,
            embedding_dim,
            padding_idx=padding_token_type_id,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _truncated_normal_(self.item_embeddings.weight, std=0.02)
        if self.item_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.item_embeddings.weight[self.item_embeddings.padding_idx].zero_()
        for table in self.feature_embeddings.values():
            _truncated_normal_(table.weight, std=0.02)
            if table.padding_idx is not None:
                with torch.no_grad():
                    table.weight[table.padding_idx].zero_()
        _truncated_normal_(self.type_embeddings.weight, std=0.02)
        if self.type_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.type_embeddings.weight[self.type_embeddings.padding_idx].zero_()

    def shift_item_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.item_id_offset == 0:
            return item_ids
        return item_ids + self.item_id_offset

    def unshift_item_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.item_id_offset == 0:
            return item_ids
        return item_ids - self.item_id_offset

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embeddings(item_ids)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_types: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.shape != token_types.shape:
            raise ValueError("token_ids and token_types must have the same shape")

        embeddings = token_ids.new_zeros(
            token_ids.shape + (self.embedding_dim,),
            dtype=self.item_embeddings.weight.dtype,
        )

        item_mask = token_types == self.item_token_type_id
        if item_mask.any():
            shifted_item_ids = self.shift_item_ids(token_ids[item_mask].long())
            embeddings[item_mask] = self.item_embeddings(shifted_item_ids)

        for type_id_key, table in self.feature_embeddings.items():
            type_id = int(type_id_key)
            feature_mask = token_types == type_id
            if feature_mask.any():
                embeddings[feature_mask] = table(token_ids[feature_mask].long())

        return embeddings + self.type_embeddings(token_types.long())


class TypedHSTUEncoder(nn.Module):

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_token_seq_len: int = 100,
        max_event_seq_len: int = 100,
        num_blocks: int = 2,
        num_heads: int = 1,
        linear_dim: int = 64,
        attention_dim: int = 64,
        num_token_types: int = 3,
        feature_vocab_sizes: Optional[Dict[int, int]] = None,
        input_dropout_rate: float = 0.0,
        linear_dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        item_id_offset: int = 1,
        concat_ua: bool = False,
        enable_relative_attention_bias: bool = True,
        relative_attention_num_buckets: int = 128,
    ) -> None:
        super().__init__()
        if max_token_seq_len <= 0:
            raise ValueError("max_token_seq_len must be positive")
        if max_event_seq_len <= 0:
            raise ValueError("max_event_seq_len must be positive")

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_token_seq_len = max_token_seq_len
        self.max_event_seq_len = max_event_seq_len
        self.token_embedding = TypedTokenEmbedding(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_token_types=num_token_types,
            feature_vocab_sizes=feature_vocab_sizes or {},
            item_id_offset=item_id_offset,
        )
        self.pos_embeddings = nn.Embedding(max_token_seq_len, embedding_dim)
        self.event_pos_embeddings = nn.Embedding(max_event_seq_len, embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=embedding_dim,
                    linear_dim=linear_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    dropout_rate=linear_dropout_rate,
                    attn_dropout_rate=attn_dropout_rate,
                    concat_ua=concat_ua,
                    relative_attention_bias=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_token_seq_len,
                            num_buckets=relative_attention_num_buckets,
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                )
                for _ in range(num_blocks)
            ]
        )
        self.reset_parameters()

    @property
    def item_id_offset(self) -> int:
        return self.token_embedding.item_id_offset

    def reset_parameters(self) -> None:
        _truncated_normal_(
            self.pos_embeddings.weight,
            std=math.sqrt(1.0 / self.embedding_dim),
        )
        _truncated_normal_(
            self.event_pos_embeddings.weight,
            std=math.sqrt(1.0 / self.embedding_dim),
        )

    def shift_item_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding.shift_item_ids(item_ids)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding.get_item_embeddings(item_ids)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_types: torch.Tensor,
        token_event_positions: torch.Tensor,
        token_lengths: torch.Tensor,
        token_timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_ids, valid_mask = create_masked_tensor(
            token_ids.long(),
            token_lengths.long(),
        )
        padded_types, _ = create_masked_tensor(token_types.long(), token_lengths.long())
        padded_event_positions, _ = create_masked_tensor(
            token_event_positions.long(),
            token_lengths.long(),
        )
        padded_timestamps = None
        if token_timestamps is not None:
            padded_timestamps, _ = create_masked_tensor(
                token_timestamps.long(),
                token_lengths.long(),
            )

        seq_len = padded_ids.shape[1]
        if seq_len > self.max_token_seq_len:
            raise ValueError(
                "Batch token sequence length "
                f"{seq_len} exceeds max_token_seq_len={self.max_token_seq_len}"
            )
        if padded_event_positions.numel() > 0:
            max_event_pos = int(padded_event_positions.max().item())
            if max_event_pos >= self.max_event_seq_len:
                raise ValueError(
                    "Batch event position "
                    f"{max_event_pos} exceeds max_event_seq_len={self.max_event_seq_len}"
                )

        x = self.token_embedding(padded_ids, padded_types) * math.sqrt(
            self.embedding_dim
        )
        token_positions = torch.arange(seq_len, device=token_ids.device)
        x = x + self.pos_embeddings(token_positions).unsqueeze(0)
        x = x + self.event_pos_embeddings(padded_event_positions)
        x = self.input_dropout(x)
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        for block in self.blocks:
            x = block(x, valid_mask=valid_mask, timestamps=padded_timestamps)

        flat_outputs = x[valid_mask]
        return flat_outputs, x, valid_mask

    def encode_last(
        self,
        token_ids: torch.Tensor,
        token_types: torch.Tensor,
        token_event_positions: torch.Tensor,
        token_lengths: torch.Tensor,
        token_timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, padded_outputs, _ = self.forward(
            token_ids=token_ids,
            token_types=token_types,
            token_event_positions=token_event_positions,
            token_lengths=token_lengths,
            token_timestamps=token_timestamps,
        )
        batch_idx = torch.arange(token_lengths.shape[0], device=token_lengths.device)
        last_idx = token_lengths.long().clamp_min(1) - 1
        return padded_outputs[batch_idx, last_idx]


class TypedHSTUModel(nn.Module):

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_token_seq_len: int = 100,
        max_event_seq_len: int = 100,
        num_blocks: int = 4,
        num_heads: int = 4,
        linear_dim: int = 16,
        attention_dim: int = 16,
        num_token_types: int = 3,
        feature_vocab_sizes: Optional[Dict[int, int]] = None,
        output_size: int = 2,
        num_negatives: int = 512,
        softmax_temperature: float = 0.05,
        sampling_strategy: NegativeSamplingStrategy = "local",
        user_embedding_norm: UserEmbeddingNorm = "l2_norm",
        l2_norm_embeddings: bool = True,
        l2_norm_eps: float = 1e-6,
        item_id_offset: int = 1,
        input_dropout_rate: float = 0.5,
        linear_dropout_rate: float = 0.5,
        attn_dropout_rate: float = 0.0,
        concat_ua: bool = False,
        enable_relative_attention_bias: bool = True,
        relative_attention_num_buckets: int = 128,
    ) -> None:
        super().__init__()
        if num_negatives <= 0:
            raise ValueError("num_negatives must be positive")
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        if sampling_strategy not in ("in-batch", "local"):
            raise ValueError("sampling_strategy must be 'in-batch' or 'local'")
        if user_embedding_norm not in ("none", "l2_norm"):
            raise ValueError("user_embedding_norm must be 'none' or 'l2_norm'")

        self.encoder = TypedHSTUEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_token_seq_len=max_token_seq_len,
            max_event_seq_len=max_event_seq_len,
            num_blocks=num_blocks,
            num_heads=num_heads,
            linear_dim=linear_dim,
            attention_dim=attention_dim,
            num_token_types=num_token_types,
            feature_vocab_sizes=feature_vocab_sizes,
            input_dropout_rate=input_dropout_rate,
            linear_dropout_rate=linear_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            item_id_offset=item_id_offset,
            concat_ua=concat_ua,
            enable_relative_attention_bias=enable_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
        )
        self.num_items = num_items
        self.output_size = output_size
        self.num_negatives = num_negatives
        self.softmax_temperature = softmax_temperature
        self.sampling_strategy = sampling_strategy
        self.user_embedding_norm = user_embedding_norm
        self.l2_norm_embeddings = l2_norm_embeddings
        self.l2_norm_eps = l2_norm_eps
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_size),
        )
        self.reset_prediction_head()

    def reset_prediction_head(self) -> None:
        for module in self.prediction_head:
            if isinstance(module, nn.Linear):
                _truncated_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _maybe_l2_norm(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.l2_norm_embeddings:
            return embeddings
        norm = torch.linalg.norm(embeddings, ord=2, dim=-1, keepdim=True)
        return embeddings / norm.clamp_min(self.l2_norm_eps)

    def _normalize_user_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.user_embedding_norm == "none":
            return embeddings
        return embeddings / torch.linalg.norm(
            embeddings,
            ord=2,
            dim=-1,
            keepdim=True,
        ).clamp_min(self.l2_norm_eps)

    def _sample_negative_ids(self, positive_ids: torch.Tensor) -> torch.Tensor:
        output_shape = positive_ids.shape + (self.num_negatives,)
        if self.sampling_strategy == "local":
            offsets = torch.randint(
                low=0,
                high=self.num_items,
                size=output_shape,
                dtype=positive_ids.dtype,
                device=positive_ids.device,
            )
            return offsets + self.encoder.item_id_offset

        source_ids = positive_ids
        sampled_offsets = torch.randint(
            low=0,
            high=source_ids.shape[0],
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        return source_ids[sampled_offsets]

    def compute_multitask_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != labels.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} must match "
                f"labels shape {tuple(labels.shape)}"
            )
        if logits.shape[1] == 1:
            return F.binary_cross_entropy_with_logits(logits[:, 0], labels[:, 0])
        if logits.shape[1] == 2:
            loss_e = F.binary_cross_entropy_with_logits(logits[:, 0], labels[:, 0])
            loss_c = F.binary_cross_entropy_with_logits(logits[:, 1], labels[:, 1])
            return 0.5 * (loss_e + loss_c)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def predict_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        flat_outputs, _, _ = self.encoder(
            token_ids=batch["token_ids"],
            token_types=batch["token_types"],
            token_event_positions=batch["token_event_positions"],
            token_lengths=batch["token_length"],
            token_timestamps=batch.get("token_timestamps"),
        )
        output_embeddings = flat_outputs[batch["supervision_token_positions"].long()]
        return self.prediction_head(output_embeddings)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.predict_logits(batch)
        return self.compute_multitask_loss(
            logits=logits,
            labels=batch["labels"].float(),
        )

    @torch.inference_mode()
    def score_all_items(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_embeddings = self.encoder.encode_last(
            token_ids=batch["token_ids"],
            token_types=batch["token_types"],
            token_event_positions=batch["token_event_positions"],
            token_lengths=batch["token_length"],
            token_timestamps=batch.get("token_timestamps"),
        )
        user_embeddings = self._normalize_user_embeddings(user_embeddings)
        item_ids = torch.arange(
            self.num_items,
            dtype=torch.long,
            device=user_embeddings.device,
        )
        shifted_item_ids = self.encoder.shift_item_ids(item_ids)
        item_embeddings = self._maybe_l2_norm(
            self.encoder.get_item_embeddings(shifted_item_ids)
        )
        return user_embeddings @ item_embeddings.t()

    @torch.inference_mode()
    def recommend(
        self,
        batch: Dict[str, torch.Tensor],
        topk: int,
        filter_seen: bool = True,
    ) -> Dict[int, List[int]]:
        scores = self.score_all_items(batch)
        if filter_seen:
            lengths = batch["length"].to(scores.device)
            history = batch["history"].to(scores.device)
            offsets = torch.repeat_interleave(
                torch.arange(batch["length"].shape[0], device=scores.device),
                lengths,
            )
            seen_items = history.long()
            valid_seen = (seen_items >= 0) & (seen_items < self.num_items)
            scores[offsets[valid_seen], seen_items[valid_seen]] = -torch.inf

        _, topk_ids = torch.topk(scores, k=topk, dim=1)
        return {
            int(uid.item()): topk_ids[row_idx].detach().cpu().tolist()
            for row_idx, uid in enumerate(batch["uid"])
        }


__all__ = [
    "TypedHSTUEncoder",
    "TypedHSTUModel",
    "TypedTokenEmbedding",
]
