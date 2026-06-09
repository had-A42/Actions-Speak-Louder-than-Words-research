from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CategoricalEncoder, MultivalentEncoder, PiecewiseLinearEncoder


class BasisComposedFieldLinear(nn.Module):
    def __init__(
        self,
        num_fields: int,
        in_dim: int,
        out_dim: int,
        num_bases: int = 16,
        meta_dim: int = 32,
        top_k: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        self.num_fields = num_fields
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_bases = num_bases
        self.top_k = min(top_k, num_bases)

        self.basis = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        nn.init.xavier_uniform_(self.basis)

        self.router = nn.Linear(meta_dim, num_bases)

        self.bias = nn.Parameter(torch.zeros(num_fields, out_dim)) if bias else None

    def get_field_weights(self, field_meta: torch.Tensor) -> torch.Tensor:
        scores = self.router(field_meta)

        top_values, top_indices = scores.topk(self.top_k, dim=-1)
        alpha = torch.softmax(top_values, dim=-1)

        selected_basis = self.basis[top_indices]

        W = (alpha[..., None, None] * selected_basis).sum(dim=1)
        return W

    def forward(
        self, x: torch.Tensor, field_ids: torch.Tensor, field_meta: torch.Tensor
    ) -> torch.Tensor:
        W_all = self.get_field_weights(field_meta)
        W = W_all[field_ids]

        out = torch.einsum("bni,nio->bno", x, W)

        if self.bias is not None:
            out = out + self.bias[field_ids].unsqueeze(0)

        return out


class FieldDecomposedAttention(nn.Module):
    def __init__(
        self,
        num_fields: int,
        model_dim: int,
        num_heads: int = 4,
        head_dim: int | None = None,
        num_bases: int = 16,
        meta_dim: int = 32,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_fields = num_fields
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or model_dim // num_heads
        self.inner_dim = num_heads * self.head_dim

        self.q_proj = BasisComposedFieldLinear(
            num_fields,
            model_dim,
            self.inner_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
        )
        self.k_proj = BasisComposedFieldLinear(
            num_fields,
            model_dim,
            self.inner_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
        )
        self.v_proj = BasisComposedFieldLinear(
            num_fields,
            model_dim,
            self.inner_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
        )

        self.out_proj = BasisComposedFieldLinear(
            num_fields,
            self.inner_dim,
            model_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
        )

        self.field_pair_weight = nn.Parameter(
            torch.zeros(num_heads, num_fields, num_fields)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, field_ids: torch.Tensor, field_meta: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q_proj(x, field_ids, field_meta)
        k = self.k_proj(x, field_ids, field_meta)
        v = self.v_proj(x, field_ids, field_meta)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        pair_w = 1.0 + self.field_pair_weight[:, field_ids][:, :, field_ids]

        scores = scores * pair_w.unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.inner_dim)

        return self.out_proj(context, field_ids, field_meta)


class FieldAwareFFN(nn.Module):
    def __init__(
        self,
        num_fields: int,
        model_dim: int,
        ffn_dim: int,
        num_bases: int = 16,
        meta_dim: int = 32,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc1 = BasisComposedFieldLinear(
            num_fields,
            model_dim,
            ffn_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
        )
        self.fc2 = BasisComposedFieldLinear(
            num_fields,
            ffn_dim,
            model_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, field_ids: torch.Tensor, field_meta: torch.Tensor
    ) -> torch.Tensor:
        x = self.fc1(x, field_ids, field_meta)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x, field_ids, field_meta)
        return x


class FATBlock(nn.Module):
    def __init__(
        self,
        num_fields: int,
        model_dim: int,
        num_heads: int = 4,
        head_dim: int | None = None,
        ffn_mult: int = 4,
        num_bases: int = 16,
        meta_dim: int = 32,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = FieldDecomposedAttention(
            num_fields=num_fields,
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
            dropout=dropout,
        )

        self.ffn = FieldAwareFFN(
            num_fields=num_fields,
            model_dim=model_dim,
            ffn_dim=ffn_mult * model_dim,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, field_ids: torch.Tensor, field_meta: torch.Tensor
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, field_ids, field_meta)))
        x = self.norm2(x + self.dropout(self.ffn(x, field_ids, field_meta)))
        return x


class FieldAwareTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_fields: int,
        model_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        head_dim: int | None = None,
        ffn_mult: int = 4,
        num_bases: int = 16,
        meta_dim: int = 32,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.field_bias = nn.Embedding(num_fields, model_dim)

        self.field_meta = nn.Parameter(torch.randn(num_fields, meta_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                FATBlock(
                    num_fields=num_fields,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_mult=ffn_mult,
                    num_bases=num_bases,
                    meta_dim=meta_dim,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, field_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.field_bias(field_ids).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, field_ids, self.field_meta)

        return x


class FieldAwareTransformerRanker(nn.Module):
    def __init__(
        self,
        embedding_size,
        dense_train_df,
        n_bins,
        train_df_slice,
        cardinality=65536,
        output_size: int = 2,
        num_hashes: int = 2,
        n_dense_tokens: int = 4,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        head_dim: int | None = None,
        ffn_mult: int = 4,
        num_bases: int = 16,
        meta_dim: int = 32,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        embeddings = nn.Embedding(cardinality, embedding_size)

        self.categorical_encoder = CategoricalEncoder(embeddings)
        self.multivalent_encoder = MultivalentEncoder(embeddings)

        self.dense_encoder = PiecewiseLinearEncoder.from_dataset(
            dense_train_df=dense_train_df,
            n_bins=n_bins,
            train_df_slice=train_df_slice,
        )

        dense_input_dim = sum(self.dense_encoder.n_bins)

        self.n_dense_tokens = n_dense_tokens
        self.model_dim = model_dim

        self.dense_projector = nn.Sequential(
            nn.LayerNorm(dense_input_dim),
            nn.Linear(dense_input_dim, n_dense_tokens * model_dim),
            nn.SiLU(),
        )

        token_in_dim = num_hashes * embedding_size

        self.feature_projectors = nn.ModuleDict(
            {
                "item_id": nn.Linear(token_in_dim, model_dim),
                "uid": nn.Linear(token_in_dim, model_dim),
                "artist_ids": nn.Linear(token_in_dim, model_dim),
                "album_ids": nn.Linear(token_in_dim, model_dim),
            }
        )

        num_fields = n_dense_tokens + 4
        self.register_buffer("field_ids", torch.arange(num_fields, dtype=torch.long))

        self.fat = FieldAwareTransformerEncoder(
            num_fields=num_fields,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            ffn_mult=ffn_mult,
            num_bases=num_bases,
            meta_dim=meta_dim,
            top_k=top_k,
            dropout=dropout,
        )

        self.output_layer = nn.Linear(model_dim, output_size)

    def _token_from_hash_emb(self, emb: torch.Tensor, name: str) -> torch.Tensor:
        return self.feature_projectors[name](emb.flatten(start_dim=1)).unsqueeze(1)

    def forward(self, inputs: dict) -> torch.Tensor:
        b = inputs["dense_features"].shape[0]

        dense_emb = self.dense_encoder(inputs["dense_features"])

        dense_tokens = self.dense_projector(dense_emb).view(
            b,
            self.n_dense_tokens,
            self.model_dim,
        )

        item_emb = self.categorical_encoder(inputs["sparse_features"]["item_id"])
        uid_emb = self.categorical_encoder(inputs["sparse_features"]["uid"])

        artist_emb = self.multivalent_encoder(
            inputs["multivalent_features"]["artist_ids"]["values"],
            inputs["multivalent_features"]["artist_ids"]["lengths"],
        )

        album_emb = self.multivalent_encoder(
            inputs["multivalent_features"]["album_ids"]["values"],
            inputs["multivalent_features"]["album_ids"]["lengths"],
        )

        tokens = torch.cat(
            [
                dense_tokens,
                self._token_from_hash_emb(item_emb, "item_id"),
                self._token_from_hash_emb(uid_emb, "uid"),
                self._token_from_hash_emb(artist_emb, "artist_ids"),
                self._token_from_hash_emb(album_emb, "album_ids"),
            ],
            dim=1,
        )

        tokens = self.fat(tokens, self.field_ids)

        pooled = tokens.sum(dim=1)

        logits = self.output_layer(pooled)
        return logits
