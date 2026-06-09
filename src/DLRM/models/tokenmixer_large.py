from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CategoricalEncoder, MultivalentEncoder, PiecewiseLinearEncoder


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class PerTokenSwiGLU(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.1,
        down_init_scale: float = 0.01,
    ):
        super().__init__()

        hidden_dim = expansion * input_dim

        self.w_up = nn.Parameter(torch.empty(num_tokens, input_dim, hidden_dim))
        self.b_up = nn.Parameter(torch.zeros(num_tokens, hidden_dim))

        self.w_gate = nn.Parameter(torch.empty(num_tokens, input_dim, hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, hidden_dim))

        self.w_down = nn.Parameter(torch.empty(num_tokens, hidden_dim, input_dim))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, input_dim))

        nn.init.xavier_uniform_(self.w_up)
        nn.init.xavier_uniform_(self.w_gate)
        nn.init.xavier_uniform_(self.w_down)
        self.w_down.data.mul_(down_init_scale)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        up = torch.einsum("btd,tdh->bth", x, self.w_up) + self.b_up
        gate = torch.einsum("btd,tdh->bth", x, self.w_gate) + self.b_gate

        x = F.silu(gate) * up
        x = self.dropout(x)

        x = torch.einsum("bth,thd->btd", x, self.w_down) + self.b_down
        return x


class MixingReverting(nn.Module):
    def __init__(self, num_tokens: int, model_dim: int, num_mixed_tokens: int):
        super().__init__()

        if model_dim % num_mixed_tokens != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by num_mixed_tokens={num_mixed_tokens}"
            )
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.num_mixed_tokens = num_mixed_tokens
        self.head_dim = model_dim // num_mixed_tokens
        self.mixed_dim = num_tokens * self.head_dim

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        x = x.view(B, T, self.num_mixed_tokens, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, self.num_mixed_tokens, self.mixed_dim)

        return x

    def revert(self, x: torch.Tensor) -> torch.Tensor:
        B, H, M = x.shape

        x = x.view(B, self.num_mixed_tokens, self.num_tokens, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, self.num_tokens, self.model_dim)

        return x


class TokenMixerLargeBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        num_mixed_tokens: int = 8,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mixing = MixingReverting(
            num_tokens=num_tokens,
            model_dim=model_dim,
            num_mixed_tokens=num_mixed_tokens,
        )

        mixed_dim = self.mixing.mixed_dim

        self.input_norm = RMSNorm(model_dim)

        self.mixed_norm = RMSNorm(mixed_dim)
        self.mixed_swiglu = PerTokenSwiGLU(
            num_tokens=num_mixed_tokens,
            input_dim=mixed_dim,
            expansion=expansion,
            dropout=dropout,
        )

        self.token_norm = RMSNorm(model_dim)
        self.token_swiglu = PerTokenSwiGLU(
            num_tokens=num_tokens,
            input_dim=model_dim,
            expansion=expansion,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.input_norm(x)
        h = self.mixing.mix(h)
        h = h + self.dropout(self.mixed_swiglu(self.mixed_norm(h)))

        h = self.mixing.revert(h)

        out = residual + self.dropout(self.token_swiglu(self.token_norm(h)))

        return out


class TokenMixerLargeEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        num_layers: int = 4,
        num_mixed_tokens: int = 8,
        expansion: int = 4,
        dropout: float = 0.1,
        interval_residual: int = 2,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TokenMixerLargeBlock(
                    num_tokens=num_tokens,
                    model_dim=model_dim,
                    num_mixed_tokens=num_mixed_tokens,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.interval_residual = interval_residual
        self.final_norm = RMSNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_state = x

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if (
                self.interval_residual is not None
                and self.interval_residual > 0
                and (i + 1) % self.interval_residual == 0
                and i != len(self.layers) - 1
            ):
                x = x + skip_state
                skip_state = x

        return self.final_norm(x)


class TokenMixerLargeRanker(nn.Module):
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
        num_layers: int = 4,
        num_mixed_tokens: int = 8,
        expansion: int = 4,
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

        self.num_base_tokens = n_dense_tokens + 4

        self.global_projector = nn.Sequential(
            nn.LayerNorm(self.num_base_tokens * model_dim),
            nn.Linear(self.num_base_tokens * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        self.num_tokens = self.num_base_tokens + 1

        self.tokenmixer = TokenMixerLargeEncoder(
            num_tokens=self.num_tokens,
            model_dim=model_dim,
            num_layers=num_layers,
            num_mixed_tokens=num_mixed_tokens,
            expansion=expansion,
            dropout=dropout,
            interval_residual=2,
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

        base_tokens = torch.cat(
            [
                dense_tokens,
                self._token_from_hash_emb(item_emb, "item_id"),
                self._token_from_hash_emb(uid_emb, "uid"),
                self._token_from_hash_emb(artist_emb, "artist_ids"),
                self._token_from_hash_emb(album_emb, "album_ids"),
            ],
            dim=1,
        )

        global_token = self.global_projector(
            base_tokens.flatten(start_dim=1)
        ).unsqueeze(1)

        tokens = torch.cat([global_token, base_tokens], dim=1)

        tokens = self.tokenmixer(tokens)

        pooled = tokens.mean(dim=1)

        logits = self.output_layer(pooled)
        return logits
