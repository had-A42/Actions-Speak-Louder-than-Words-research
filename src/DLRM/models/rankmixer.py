from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CategoricalEncoder, MultivalentEncoder, PiecewiseLinearEncoder


class RankMixerTokenMixing(nn.Module):
    def __init__(self, num_tokens: int, model_dim: int):
        super().__init__()
        if model_dim % num_tokens != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by num_tokens={num_tokens}"
            )
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.num_heads = num_tokens
        self.head_dim = model_dim // num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        x = x.contiguous().view(B, T, D)
        return x


class PerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.hidden_dim = ffn_mult * model_dim

        self.w1 = nn.Parameter(torch.empty(num_tokens, model_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, self.hidden_dim))

        self.w2 = nn.Parameter(torch.empty(num_tokens, self.hidden_dim, model_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, model_dim))

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.einsum("btd,tdh->bth", x, self.w1) + self.b1
        x = F.gelu(x)
        x = self.dropout(x)

        x = torch.einsum("bth,thd->btd", x, self.w2) + self.b2
        return x


class RankMixerBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_mixing = RankMixerTokenMixing(
            num_tokens=num_tokens,
            model_dim=model_dim,
        )

        self.pffn = PerTokenFFN(
            num_tokens=num_tokens,
            model_dim=model_dim,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.token_mixing(x)))

        x = self.norm2(x + self.dropout(self.pffn(x)))

        return x


class RankMixerEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        num_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                RankMixerBlock(
                    num_tokens=num_tokens,
                    model_dim=model_dim,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class RankMixerRanker(nn.Module):
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
        ffn_mult: int = 4,
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
            nn.GELU(),
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

        num_tokens = n_dense_tokens + 4

        self.rankmixer = RankMixerEncoder(
            num_tokens=num_tokens,
            model_dim=model_dim,
            num_layers=num_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        self.output_layer = nn.Linear(model_dim, output_size)

    def _token_from_hash_emb(self, emb: torch.Tensor, name: str) -> torch.Tensor:
        emb = emb.flatten(start_dim=1)
        return self.feature_projectors[name](emb).unsqueeze(1)

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

        tokens = self.rankmixer(tokens)

        pooled = tokens.mean(dim=1)

        logits = self.output_layer(pooled)
        return logits
