from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalEncoder(nn.Module):
    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(ids)


class MultivalentEncoder(nn.Module):
    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        num_hashes = ids.shape[1]
        embedding_dim = self.embeddings.embedding_dim

        if ids.shape[0] == 0:
            return torch.zeros(
                batch_size,
                num_hashes,
                embedding_dim,
                dtype=self.embeddings.weight.dtype,
                device=ids.device,
            )

        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=lengths.device),
                lengths.cumsum(0)[:-1],
            ]
        )

        outputs = []
        for h in range(num_hashes):
            out = F.embedding_bag(
                ids[:, h],
                self.embeddings.weight,
                offsets=offsets,
                mode="mean",
            )
            outputs.append(out)

        return torch.stack(outputs, dim=1)


class PiecewiseLinearEncoder(nn.Module):
    @staticmethod
    def compute_bins(
        X: torch.Tensor,
        n_bins: int,
    ) -> list[torch.Tensor]:
        quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=X.device, dtype=X.dtype)
        bins = []

        for i in range(X.shape[1]):
            feature_bins = torch.quantile(X[:, i], quantiles)
            feature_bins = torch.unique(feature_bins, sorted=True)

            n_bin = feature_bins.numel() - 1
            assert n_bin >= 1, "There is a column with only one unique value"

            bins.append(feature_bins)

        return bins

    @classmethod
    def from_dataset(cls, dense_train_df, n_bins=32, train_df_slice: int = 1_000_000):
        if isinstance(dense_train_df, torch.Tensor):
            X = dense_train_df[:train_df_slice].to(torch.float32)
        else:
            dense_slice = dense_train_df[:train_df_slice]
            if hasattr(dense_slice, "to_numpy"):
                X = torch.as_tensor(dense_slice.to_numpy(), dtype=torch.float32)
            else:
                X = torch.as_tensor(dense_slice, dtype=torch.float32)

        bins = cls.compute_bins(X, n_bins)
        n_bins_list = [len(b) - 1 for b in bins]

        n_features = len(bins)
        max_n_bins = max(n_bins_list)

        weight = torch.zeros(n_features, max_n_bins, dtype=torch.float32)
        bias = torch.zeros(n_features, max_n_bins, dtype=torch.float32)

        need_mask = len(set(n_bins_list)) > 1
        mask = (
            torch.zeros(n_features, max_n_bins, dtype=torch.bool) if need_mask else None
        )

        single_bin_mask = torch.tensor([n == 1 for n in n_bins_list], dtype=torch.bool)
        if not single_bin_mask.any():
            single_bin_mask = None

        for i, feature_bins in enumerate(bins):
            deltas = feature_bins[1:] - feature_bins[:-1]
            w = 1.0 / deltas
            c = -feature_bins[:-1] / deltas

            k = len(feature_bins) - 1
            weight[i, :k] = w
            bias[i, :k] = c

            if mask is not None:
                mask[i, :k] = True

        return cls(
            weight=weight,
            bias=bias,
            mask=mask.reshape(-1) if mask is not None else None,
            n_bins=n_bins_list,
            single_bin_mask=single_bin_mask,
        )

    def __init__(self, weight, bias, mask, n_bins, single_bin_mask):
        super().__init__()
        self._n_bins = list(n_bins)

        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        if single_bin_mask is not None:
            self.register_buffer("single_bin_mask", single_bin_mask)
        else:
            self.single_bin_mask = None

        n_features, max_n_bins = weight.shape

        first_mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
        middle_mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
        last_mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)

        for i, k in enumerate(self._n_bins):
            if k == 1:
                continue
            first_mask[i, 0] = True
            last_mask[i, k - 1] = True
            if k > 2:
                middle_mask[i, 1 : k - 1] = True

        self.register_buffer("_first_mask", first_mask.reshape(-1))
        self.register_buffer("_middle_mask", middle_mask.reshape(-1))
        self.register_buffer("_last_mask", last_mask.reshape(-1))

    @property
    def n_bins(self):
        return list(self._n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.bias.unsqueeze(0) + self.weight.unsqueeze(0) * x.unsqueeze(-1)
        encoded = encoded.reshape(x.shape[0], -1)

        if self._first_mask.any():
            encoded[:, self._first_mask] = encoded[:, self._first_mask].clamp_max(1.0)

        if self._middle_mask.any():
            encoded[:, self._middle_mask] = encoded[:, self._middle_mask].clamp(
                0.0, 1.0
            )

        if self._last_mask.any():
            encoded[:, self._last_mask] = encoded[:, self._last_mask].clamp_min(0.0)

        if self.mask is not None:
            encoded = encoded[:, self.mask]

        return encoded


class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()

        layers = []
        in_dim = input_dim

        for out_dim in hidden_units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        self.nn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        out = out + residual
        out = self.relu(out)
        return out


class ResDeepNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int]):
        super().__init__()
        dims = [input_dim] + hidden_units
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DenseDeepNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int]):
        super().__init__()

        layers = []
        current_input_dim = input_dim

        for out_dim in hidden_units:
            layers.append(nn.Linear(current_input_dim, out_dim))
            current_input_dim += out_dim

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]

        for layer in self.layers:
            layer_input = torch.cat(features, dim=-1)
            out = self.relu(layer(layer_input))
            features.append(out)

        return features[-1]


def build_deep_network(
    input_dim: int,
    hidden_units: list[int],
    deep_type: str = "mlp",
) -> nn.Module:
    """Factory for the deep tower: ``mlp``, ``resnet``, or ``densenet``."""
    if deep_type == "mlp":
        return DeepNetwork(input_dim, hidden_units)
    if deep_type == "resnet":
        return ResDeepNetwork(input_dim, hidden_units)
    if deep_type == "densenet":
        return DenseDeepNetwork(input_dim, hidden_units)
    raise ValueError(
        f"Unknown deep_type={deep_type!r}, expected 'mlp', 'resnet', or 'densenet'"
    )
