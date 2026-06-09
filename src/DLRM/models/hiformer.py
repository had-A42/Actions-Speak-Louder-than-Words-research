from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CategoricalEncoder, MultivalentEncoder, PiecewiseLinearEncoder


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankCompositeProjection(nn.Module):
    def __init__(self, seq_len: int, model_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.left = nn.Linear(seq_len * model_dim, rank, bias=False)
        self.right = nn.Linear(rank, seq_len * out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = x.reshape(b, -1)
        z = self.right(self.left(z))
        return z.view(b, self.seq_len, self.out_dim)


class LowRankTaskProjection(nn.Module):
    def __init__(self, model_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.left = nn.Linear(model_dim, rank, bias=False)
        self.right = nn.Linear(rank, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x))


class TokenwiseLinear(nn.Module):
    def __init__(self, num_tokens: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_tokens, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_tokens, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bti,tio->bto", x, self.weight) + self.bias


class HeterogeneousFFN(nn.Module):
    def __init__(self, num_tokens: int, model_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = TokenwiseLinear(num_tokens, model_dim, ffn_dim)
        self.fc2 = TokenwiseLinear(num_tokens, ffn_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class HiformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_tasks: int,
        model_dim: int,
        num_heads: int = 4,
        dk: int | None = None,
        dv: int | None = None,
        rank_qk: int = 64,
        rank_v: int = 128,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        prune_to_tasks: bool = True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.num_tasks = num_tasks
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dk = dk or model_dim // num_heads
        self.dv = dv or model_dim // num_heads
        self.prune_to_tasks = prune_to_tasks  # глава3.5.2 утверждает, что в последнеи слое можно оставлять только task-токены в качестве query, а key/value брать из всех токенов

        if prune_to_tasks:
            self.q_proj = nn.ModuleList(
                [
                    LowRankTaskProjection(model_dim, self.dk, rank_qk)
                    for _ in range(num_heads)
                ]
            )
            out_tokens = num_tasks
        else:
            self.q_proj = nn.ModuleList(
                [
                    LowRankCompositeProjection(seq_len, model_dim, self.dk, rank_qk)
                    for _ in range(num_heads)
                ]
            )
            out_tokens = seq_len

        self.k_proj = nn.ModuleList(
            [
                LowRankCompositeProjection(seq_len, model_dim, self.dk, rank_qk)
                for _ in range(num_heads)
            ]
        )

        self.v_proj = nn.ModuleList(
            [
                LowRankCompositeProjection(seq_len, model_dim, self.dv, rank_v)
                for _ in range(num_heads)
            ]
        )

        self.out_proj = TokenwiseLinear(out_tokens, num_heads * self.dv, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = HeterogeneousFFN(out_tokens, model_dim, ffn_mult * model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x[:, : self.num_tasks, :] if self.prune_to_tasks else x
        contexts = []

        for h in range(self.num_heads):
            q = self.q_proj[h](residual if self.prune_to_tasks else x)
            k = self.k_proj[h](x)
            v = self.v_proj[h](x)

            attn = torch.softmax(
                torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk),
                dim=-1,
            )
            contexts.append(torch.matmul(attn, v))

        z = torch.cat(contexts, dim=-1)
        z = self.out_proj(z)

        z = self.norm1(residual + self.dropout(z))
        z = self.norm2(z + self.dropout(self.ffn(z)))
        return z


class HiformerEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_tasks: int,
        model_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        rank_qk: int = 64,
        rank_v: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                HiformerBlock(
                    seq_len=seq_len,
                    num_tasks=num_tasks,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    rank_qk=rank_qk,
                    rank_v=rank_v,
                    dropout=dropout,
                    prune_to_tasks=(i == num_layers - 1),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class HiformerRanker(nn.Module):
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
        num_heads: int = 4,
        num_layers: int = 1,
        rank_qk: int = 64,
        rank_v: int = 128,
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
        self.output_size = output_size

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

        self.task_tokens = nn.Parameter(torch.randn(output_size, model_dim) * 0.02)

        seq_len = output_size + n_dense_tokens + 4

        self.hiformer = HiformerEncoder(
            seq_len=seq_len,
            num_tasks=output_size,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            rank_qk=rank_qk,
            rank_v=rank_v,
            dropout=dropout,
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

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

        feature_tokens = torch.cat(
            [
                dense_tokens,
                self._token_from_hash_emb(item_emb, "item_id"),
                self._token_from_hash_emb(uid_emb, "uid"),
                self._token_from_hash_emb(artist_emb, "artist_ids"),
                self._token_from_hash_emb(album_emb, "album_ids"),
            ],
            dim=1,
        )

        task_tokens = self.task_tokens.unsqueeze(0).expand(b, -1, -1)

        tokens = torch.cat([task_tokens, feature_tokens], dim=1)

        task_out = self.hiformer(tokens)
        logits = self.output_layer(task_out).squeeze(-1)

        return logits
