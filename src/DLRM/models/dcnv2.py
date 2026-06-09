from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    CategoricalEncoder,
    MultivalentEncoder,
    PiecewiseLinearEncoder,
    build_deep_network,
)


class MixtureLowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, rank: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.rank = rank

        self.U = nn.Parameter(torch.empty(num_experts, input_dim, rank))
        self.V = nn.Parameter(torch.empty(num_experts, input_dim, rank))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.gate = nn.Linear(input_dim, num_experts)

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:

        # xl: [B, D]
        # V:  [E, D, R]
        # U:  [E, D, R]

        low_rank = torch.einsum("bd,edr->ber", xl, self.V)
        transformed = torch.einsum("ber,edr->bed", low_rank, self.U)
        experts_out = x0.unsqueeze(1) * (transformed + self.bias)
        gates = F.softmax(self.gate(xl), dim=-1)
        mixed = torch.sum(experts_out * gates.unsqueeze(-1), dim=1)

        return mixed + xl


class MixtureLowRankCrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, num_experts: int, rank: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MixtureLowRankCrossLayer(
                    input_dim=input_dim,
                    num_experts=num_experts,
                    rank=rank,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DCNV2(nn.Module):
    def __init__(
        self,
        embedding_size,
        cross_layers,
        deep_units,
        input_size,
        dense_train_df,
        n_bins,
        train_df_slice,
        cardinality=65536,
        num_experts: int = 4,
        low_rank: int = 32,
        deep_network: str = "mlp",
        output_size: int = 2,
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

        self.cross_network = MixtureLowRankCrossNetwork(
            input_dim=input_size,
            num_layers=cross_layers,
            num_experts=num_experts,
            rank=low_rank,
        )
        self.deep_network = build_deep_network(
            input_dim=input_size,
            hidden_units=deep_units,
            deep_type=deep_network,
        )

        self.output_layer = nn.Linear(input_size + deep_units[-1], output_size)

    def forward(self, inputs: dict) -> torch.Tensor:
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

        dense_emb = self.dense_encoder(inputs["dense_features"])

        item_emb = item_emb.flatten(start_dim=1)
        uid_emb = uid_emb.flatten(start_dim=1)
        artist_emb = artist_emb.flatten(start_dim=1)
        album_emb = album_emb.flatten(start_dim=1)

        x = torch.cat(
            [dense_emb, item_emb, uid_emb, artist_emb, album_emb],
            dim=-1,
        )

        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)

        out = torch.cat([cross_out, deep_out], dim=-1)
        out = self.output_layer(out)
        return out
