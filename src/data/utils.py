from pathlib import Path
from typing import List, Tuple

import requests
from tqdm import tqdm
import torch

def download_file(
    url: str,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"File already exists: {output_path}")
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(tmp_path, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

    tmp_path.replace(output_path)

    print(f"Downloaded to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Sequence-level split helpers
# (operate on already-aggregated per-user sequences, not flat DataFrames)
# ---------------------------------------------------------------------------

def parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated string into a list of ints.

    Example
    -------
    >>> parse_int_list("1,2,3")
    [1, 2, 3]
    """
    return [int(x) for x in str(s).split(",")]


def split_leave_one_out(
    item_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """LeaveOneOut split on a single user's item sequence.

    Returns
    -------
    history : List[int]
        All items except the last one.
    targets : List[int]
        A single-element list containing the last item.
        Empty list if the sequence has fewer than 2 items.

    Example
    -------
    >>> split_leave_one_out([1, 2, 3, 4, 5])
    ([1, 2, 3, 4], [5])
    """
    if len(item_ids) < 2:
        return item_ids, []
    return item_ids[:-1], [item_ids[-1]]


def split_global_temporal(
    item_ids: List[int],
    timestamps: List[int],
    split_timestamp: int,
) -> Tuple[List[int], List[int]]:
    """GlobalTemporalSplit on a single user's item sequence.

    Items with ``timestamp < split_timestamp`` go to *history*;
    items with ``timestamp >= split_timestamp`` go to *targets*.

    Parameters
    ----------
    item_ids : List[int]
        Ordered list of item IDs (chronological).
    timestamps : List[int]
        Corresponding timestamps, same length as ``item_ids``.
    split_timestamp : int
        The boundary timestamp.

    Returns
    -------
    history : List[int]
    targets : List[int]

    Example
    -------
    >>> split_global_temporal([1, 2, 3], [100, 200, 300], 200)
    ([1], [2, 3])
    """
    history = [iid for iid, ts in zip(item_ids, timestamps) if ts < split_timestamp]
    targets = [iid for iid, ts in zip(item_ids, timestamps) if ts >= split_timestamp]
    return history, targets

def create_masked_tensor(data: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a batch of flattened variable-length sequences into a padded tensor and mask.
    Supports:
        - indices: data shape (total_num_elements,)
        - embeddings/features: data shape (total_num_elements, d1, d2, ...)

    Parameters
    ----------
    data : torch.Tensor
        Input tensor containing flattened sequences:
        - For indices: shape (total_num_elements,)
        - For embeddings: shape (total_num_elements, embedding_dim)
    lengths : torch.Tensor
        1D tensor of sequence lengths, shape (batch_size,). Specifies the actual length
        of each sequence.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - padded_tensor: Padded tensor of shape:
            - (batch_size, max_seq_len) for indices
            - (batch_size, max_seq_len, embedding_dim) for embeddings
            Shorter sequences are right-padded with zeros.
        - mask: Boolean mask of shape (batch_size, max_seq_len) where True indicates
            valid elements and False indicates padding. Can be used in attention or loss computation.

    Examples
    --------
    >>> data = torch.tensor([1, 2, 3, 4, 5, 6])  # sequences: [1,2], [3,4,5], [6]
    >>> lengths = torch.tensor([2, 3, 1])
    >>> padded, mask = create_masked_tensor(data, lengths)
    >>> padded
    tensor([[1, 2, 0],
            [3, 4, 5],
            [6, 0, 0]])
    >>> mask
    tensor([[ True,  True, False],
            [ True,  True,  True],
            [ True, False, False]])
    """

    padded = torch.zeros((len(lengths), lengths.max()) + tuple(data.shape[1:]), dtype=data.dtype, device=data.device)

    mask = (torch.arange(lengths.max(), device=lengths.device).expand(len(lengths), lengths.max()) < lengths.unsqueeze(1))
    padded.masked_scatter_(mask.reshape(mask.shape + (1,) * len(data.shape[1:])).expand_as(padded), data)

    return padded, mask


def build_q_from_train_targets(
    train_targets: torch.Tensor,
    catalog_size: int
) -> torch.Tensor:
    if train_targets.numel() == 0:
        raise ValueError("train_targets пустой")
    flat = train_targets.flatten()
    if (flat < 0).any() or (flat >= catalog_size).any():
        raise ValueError("Некорректные id")
    return torch.bincount(flat, minlength=catalog_size).float()