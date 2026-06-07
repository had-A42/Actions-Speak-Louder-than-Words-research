import abc
import math
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.hstu_dataset import create_masked_tensor

NegativeSamplingStrategy = Literal["in-batch", "local"]
UserEmbeddingNorm = Literal["none", "l2_norm"]

def _truncated_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def _normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    nn.init.normal_(tensor, mean=0.0, std=std)


def _default_time_bucketization(timestamps_delta: torch.Tensor) -> torch.Tensor:
    deltas = torch.abs(timestamps_delta).clamp(min=1).to(torch.float32)
    return (torch.log(deltas) / 0.301).long()


class RelativeAttentionBiasModule(nn.Module):
    @abc.abstractmethod
    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """Return a relative attention bias with shape [B, S, S]."""
        pass


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int = 128,
        bucketization_fn: Callable[
            [torch.Tensor],
            torch.Tensor,
        ] = _default_time_bucketization,
    ) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")

        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.bucketization_fn = bucketization_fn
        self.ts_w = nn.Parameter(torch.empty(num_buckets + 1))
        self.pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _normal_(self.ts_w, std=0.02)
        _normal_(self.pos_w, std=0.02)

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        if all_timestamps.ndim != 2:
            raise ValueError("all_timestamps must have shape [B, S]")

        batch_size, seq_len = all_timestamps.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                "Timestamp sequence length "
                f"{seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=all_timestamps.device)
        rel_pos_idx = (
            positions.unsqueeze(1) - positions.unsqueeze(0) + self.max_seq_len - 1
        )
        rel_pos_bias = self.pos_w.index_select(0, rel_pos_idx.reshape(-1)).view(
            1,
            seq_len,
            seq_len,
        )

        timestamp_deltas = all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)
        bucketed_timestamps = torch.clamp(
            self.bucketization_fn(timestamp_deltas),
            min=0,
            max=self.num_buckets,
        ).detach()
        rel_ts_bias = self.ts_w.index_select(
            0,
            bucketed_timestamps.reshape(-1),
        ).view(batch_size, seq_len, seq_len)
        return rel_pos_bias + rel_ts_bias


class HSTUBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_dim: int,
        attention_dim: int,
        num_heads: int = 1,
        linear_activation: Literal["silu", "none"] = "silu",
        dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        concat_ua: bool = False,
        relative_attention_bias: Optional[RelativeAttentionBiasModule] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if linear_dim <= 0 or attention_dim <= 0:
            raise ValueError("linear_dim and attention_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.linear_activation = linear_activation
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.concat_ua = concat_ua
        self.relative_attention_bias = relative_attention_bias

        projection_dim = num_heads * (2 * linear_dim + 2 * attention_dim)
        self.uvqk = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.output = nn.Linear(
            linear_dim * num_heads * (3 if concat_ua else 1),
            embedding_dim,
        )
        self.input_norm = nn.LayerNorm(embedding_dim, eps=eps)
        self.attn_norm = nn.LayerNorm(linear_dim * num_heads, eps=eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _normal_(self.uvqk.weight, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, S, D]")

        batch_size, seq_len, _ = x.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=x.device,
            )
        elif valid_mask.shape != (batch_size, seq_len):
            raise ValueError("valid_mask must have shape [B, S]")

        normed_x = self.input_norm(x)
        projected = self.uvqk(normed_x)
        if self.linear_activation == "silu":
            projected = F.silu(projected)
        elif self.linear_activation != "none":
            raise ValueError(f"Unknown linear_activation {self.linear_activation}")

        split_sizes = [
            self.linear_dim * self.num_heads,
            self.linear_dim * self.num_heads,
            self.attention_dim * self.num_heads,
            self.attention_dim * self.num_heads,
        ]
        u, v, q, k = torch.split(projected, split_sizes, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)

        attn = torch.einsum("bshd,bthd->bhst", q, k)
        if self.relative_attention_bias is not None:
            if timestamps is None:
                raise ValueError(
                    "timestamps are required when relative attention bias is enabled"
                )
            if timestamps.shape != (batch_size, seq_len):
                raise ValueError("timestamps must have shape [B, S]")
            attn = attn + self.relative_attention_bias(timestamps).unsqueeze(1)
        attn = F.silu(attn) / max(seq_len, 1)

        causal_mask = torch.ones(
            (seq_len, seq_len),
            dtype=torch.bool,
            device=x.device,
        ).tril()
        pair_mask = (
            causal_mask.unsqueeze(0).unsqueeze(0)
            & valid_mask[:, None, :, None]
            & valid_mask[:, None, None, :]
        )
        attn = attn.masked_fill(~pair_mask, 0.0)
        attn = F.dropout(attn, p=self.attn_dropout_rate, training=self.training)

        attn_output = torch.einsum("bhst,bthd->bshd", attn, v)
        attn_output = attn_output.reshape(
            batch_size,
            seq_len,
            self.num_heads * self.linear_dim,
        )

        if self.concat_ua:
            a = self.attn_norm(attn_output)
            output_input = torch.cat([u, a, u * a], dim=-1)
        else:
            output_input = u * self.attn_norm(attn_output)

        output = self.output(
            F.dropout(output_input, p=self.dropout_rate, training=self.training)
        )
        output = output + x
        return output * valid_mask.unsqueeze(-1).to(output.dtype)


class HSTUEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_seq_len: int = 100,
        num_blocks: int = 2,
        num_heads: int = 1,
        linear_dim: int = 64,
        attention_dim: int = 64,
        input_dropout_rate: float = 0.0,
        linear_dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        item_id_offset: int = 1,
        concat_ua: bool = False,
        enable_relative_attention_bias: bool = True,
        relative_attention_num_buckets: int = 128,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.item_id_offset = item_id_offset

        self.item_embeddings = nn.Embedding(
            num_items + item_id_offset,
            embedding_dim,
            padding_idx=0,
        )
        self.pos_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=embedding_dim,
                    linear_dim=linear_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    linear_activation="silu",
                    dropout_rate=linear_dropout_rate,
                    attn_dropout_rate=attn_dropout_rate,
                    concat_ua=concat_ua,
                    relative_attention_bias=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_seq_len,
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

    def reset_parameters(self) -> None:
        _truncated_normal_(self.item_embeddings.weight, std=0.02)
        if self.item_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.item_embeddings.weight[self.item_embeddings.padding_idx].zero_()
        _truncated_normal_(
            self.pos_embeddings.weight,
            std=math.sqrt(1.0 / self.embedding_dim),
        )

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
        history: torch.Tensor,
        lengths: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shifted_history = self.shift_item_ids(history.long())
        padded_ids, valid_mask = create_masked_tensor(shifted_history, lengths.long())
        padded_timestamps = None
        if timestamps is not None:
            padded_timestamps, _ = create_masked_tensor(
                timestamps.long(),
                lengths.long(),
            )
        seq_len = padded_ids.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Batch sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.item_embeddings(padded_ids) * math.sqrt(self.embedding_dim)
        positions = torch.arange(seq_len, device=history.device)
        x = x + self.pos_embeddings(positions).unsqueeze(0)
        x = self.input_dropout(x)
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        for block in self.blocks:
            x = block(x, valid_mask=valid_mask, timestamps=padded_timestamps)

        flat_outputs = x[valid_mask]
        return flat_outputs, x, valid_mask

    def encode_last(
        self,
        history: torch.Tensor,
        lengths: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, padded_outputs, _ = self.forward(
            history=history,
            lengths=lengths,
            timestamps=timestamps,
        )
        batch_idx = torch.arange(lengths.shape[0], device=lengths.device)
        last_idx = lengths.long().clamp_min(1) - 1
        return padded_outputs[batch_idx, last_idx]


class HSTUModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_seq_len: int = 100,
        num_blocks: int = 4,
        num_heads: int = 4,
        linear_dim: int = 16,
        attention_dim: int = 16,
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

        self.encoder = HSTUEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks,
            num_heads=num_heads,
            linear_dim=linear_dim,
            attention_dim=attention_dim,
            input_dropout_rate=input_dropout_rate,
            linear_dropout_rate=linear_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            item_id_offset=item_id_offset,
            concat_ua=concat_ua,
            enable_relative_attention_bias=enable_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
        )
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.softmax_temperature = softmax_temperature
        self.sampling_strategy = sampling_strategy
        self.user_embedding_norm = user_embedding_norm
        self.l2_norm_embeddings = l2_norm_embeddings
        self.l2_norm_eps = l2_norm_eps

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

    def compute_loss(
        self,
        output_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = self._normalize_user_embeddings(output_embeddings)
        shifted_targets = self.encoder.shift_item_ids(target_ids.long())
        supervision_embeddings = self._maybe_l2_norm(
            self.encoder.get_item_embeddings(shifted_targets)
        )
        negative_ids = self._sample_negative_ids(shifted_targets)
        negative_embeddings = self._maybe_l2_norm(
            self.encoder.get_item_embeddings(negative_ids)
        )

        positive_logits = torch.sum(
            output_embeddings * supervision_embeddings,
            dim=-1,
            keepdim=True,
        )
        negative_logits = torch.sum(
            output_embeddings.unsqueeze(1) * negative_embeddings,
            dim=-1,
        )
        negative_logits = torch.where(
            negative_ids == shifted_targets.unsqueeze(1),
            torch.full_like(negative_logits, -5e4),
            negative_logits,
        )

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits = logits / self.softmax_temperature
        labels = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=logits.device,
        )
        return F.cross_entropy(logits, labels)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_embeddings, _, _ = self.encoder(
            history=batch["history"],
            lengths=batch["length"],
            timestamps=batch.get("timestamps"),
        )
        return self.compute_loss(
            output_embeddings=output_embeddings,
            target_ids=batch["targets"],
        )

    @torch.inference_mode()
    def score_all_items(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_embeddings = self.encoder.encode_last(
            history=batch["history"],
            lengths=batch["length"],
            timestamps=batch.get("timestamps"),
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
    "HSTUBlock",
    "HSTUEncoder",
    "HSTUModel",
    "RelativeAttentionBiasModule",
    "RelativeBucketedTimeAndPositionBasedBias",
    "UserEmbeddingNorm",
]
