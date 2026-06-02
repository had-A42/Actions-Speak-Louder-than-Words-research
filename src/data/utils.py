from pathlib import Path
from typing import List, Tuple

import requests
from tqdm import tqdm

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