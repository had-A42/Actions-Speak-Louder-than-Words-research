from typing import Optional, Union, Tuple
import pandas as pd
import torch


def transform_indices(data: pd.DataFrame, users: str, items:str, inplace: bool=False):
    '''
    Reindex columns that correspond to users and items.
    New index is contiguous starting from 0.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data to be reindexed.
    users : str
        The name of the column in `data` that contains user IDs.
    items : str
        The name of the column in `data` that contains item IDs.
    inplace : bool
        whether the data should be modified inplace, `False` by default.

    Returns
    -------
    pandas.DataFrame, dict
        The reindexed data and a dictionary with mapping between original IDs and the new numeric IDs.
        The keys of the dictionary are 'users' and 'items'.
        The values of the dictionary are pandas Index objects.

    Examples
    --------
    >>> data = pd.DataFrame({'customers': ['A', 'B', 'C'], 'products': ['X', 'Y', 'Z'], 'rating': [1, 2, 3]})
    >>> data_reindexed, data_index = transform_indices(data, 'customers', 'products')
    >>> data_reindexed
       users  items  rating
    0      0      0       1
    1      1      1       2
    2      2      2       3
    >>> data_index
    {
        'users': Index(['A', 'B', 'C'], dtype='object', name='customers'),
        'items': Index(['X', 'Y', 'Z'], dtype='object', name='products')
    }
    '''
    data_index = {}
    data_codes = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        if inplace:
            data[field] = new_index
        else:
            data_codes[field] = new_index

    if data_codes:
        data = data.assign(**data_codes) # makes a copy of data
    return data, data_index


def to_numeric_id(data: pd.DataFrame, field: str):
    """
    This function takes in two arguments, data and field. It converts the data field
    into categorical values and creates a new contiguous index. It then creates an
    idx_map which is a renamed version of the field argument. Finally, it returns the
    idx and idx_map variables.
    """
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def reindex_data(
        data: pd.DataFrame,
        data_index: dict,
        entities: Optional[Union[str, list[str]]] = None,
        filter_invalid: bool = True,
        inplace: bool = False
    ):
    '''
    Reindex provided data with the specified index mapping.
    By default, will take the name of the fields to reindex from `data_index`.
    It is also possible to specify which field to reindex by providing `entities`.
    '''
    if entities is None:
        entities = data_index.keys()
    if isinstance(entities, str): # handle single entity provided as a string
        entities = [entities]

    data_codes = {}
    for entity in entities:
        entity_index = data_index[entity]
        field = entity_index.name # extract the field name
        new_index = entity_index.get_indexer(data[field])
        if inplace:
            data[field] = new_index # assign new values inplace
        else:
            data_codes[field] = new_index # store new values

    if data_codes:
        data = data.assign(**data_codes) # assign new values by making a copy

    if filter_invalid: # discard unrecognized entity index
        valid_values = [f'{data_index[entity].name}>=0' for entity in entities]
        data = data.query(' and '.join(valid_values))
    return data


def verify_time_split(
        before: pd.DataFrame,
        after: pd.DataFrame,
        target_field: str='user_id',
        timeid: str='timestamp'
    ):
    '''
    Check that items from `after` dataframe have later timestamps than
    any corresponding item from the `before` dataframe. Compare w.r.t target_field.
    Usage example: assert that for any user, the holdout items are the most recent ones.
    '''
    before_ts = before.groupby(target_field)[timeid].max()
    after_ts = after.groupby(target_field)[timeid].min()
    assert (
        before_ts
        .reindex(after_ts.index)
        .combine(after_ts, lambda x, y: True if x!=x else x <= y)
    ).all()


def temporal_split(
        data: pd.DataFrame,
        split_column: str,
        split_value: Union[int, float, str],
        comparison: str = 'less'
    ):
    '''
    Split dataframe temporally based on a column and value comparison.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data to be split.
    split_column : str
        The name of the column to use for splitting (e.g., timestamp, date).
    split_value : int, float, or str
        The value to compare against for splitting.
    comparison : str
        The comparison operator: 'less' for < split_value, 'greater' for >= split_value.
        Default is 'less'.
    
    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        Two dataframes: (before_split, after_split)
        - before_split: rows where split_column < split_value (if comparison='less')
        - after_split: rows where split_column >= split_value (if comparison='less')
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'user_id': [1, 1, 2, 2, 3],
    ...     'timestamp': [100, 200, 150, 250, 300],
    ...     'item_id': ['A', 'B', 'C', 'D', 'E']
    ... })
    >>> train, test = temporal_split(data, 'timestamp', 200, 'less')
    >>> train
       user_id  timestamp item_id
    0        1        100       A
    2        2        150       C
    >>> test
       user_id  timestamp item_id
    1        1        200       B
    3        2        250       D
    4        3        300       E
    '''
    if comparison == 'less':
        before = data[data[split_column] < split_value].copy()
        after = data[data[split_column] >= split_value].copy()
    elif comparison == 'greater':
        before = data[data[split_column] >= split_value].copy()
        after = data[data[split_column] < split_value].copy()
    else:
        raise ValueError(f"comparison must be 'less' or 'greater', got '{comparison}'")
    
    return before, after


def remove_users_without_train_events(
        train: pd.DataFrame,
        test: pd.DataFrame,
        user_id_column: str = 'user_id'
    ):
    '''
    Remove user_ids from test dataframe that have no events in train dataframe.
    
    This ensures that all users in the test set have at least one interaction
    in the training set, which is important for many recommendation algorithms.
    
    Parameters
    ----------
    train : pandas.DataFrame
        The training data containing user interactions.
    test : pandas.DataFrame
        The test data to be filtered.
    user_id_column : str
        The name of the column containing user IDs. Default is 'user_id'.
    
    Returns
    -------
    pandas.DataFrame
        The filtered test dataframe containing only users that have events in train.
    
    Examples
    --------
    >>> train = pd.DataFrame({
    ...     'user_id': [1, 1, 2, 2],
    ...     'item_id': ['A', 'B', 'C', 'D']
    ... })
    >>> test = pd.DataFrame({
    ...     'user_id': [1, 2, 3, 4],
    ...     'item_id': ['E', 'F', 'G', 'H']
    ... })
    >>> filtered_test = remove_users_without_train_events(train, test)
    >>> filtered_test
       user_id item_id
    0        1       E
    1        2       F
    '''
    train_users = set(train[user_id_column].unique())
    filtered_test = test[test[user_id_column].isin(train_users)].copy()
    
    return filtered_test


def filter_core_records(
        data: pd.DataFrame,
        user_id_column: str = 'user_id',
        item_id_column: str = 'item_id',
        min_user_interactions: int = 5,
        min_item_interactions: int = 5
    ):
    '''
    Filter records to keep only core users and items with minimum interaction counts.
    
    This function filters the dataframe to retain only users and items that have
    at least the specified minimum number of interactions. This is commonly used
    in recommendation systems to remove sparse users and items.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing user-item interactions.
    user_id_column : str
        The name of the column containing user IDs. Default is 'user_id'.
    item_id_column : str
        The name of the column containing item IDs. Default is 'item_id'.
    min_user_interactions : int
        Minimum number of interactions required for a user to be kept. Default is 5.
    min_item_interactions : int
        Minimum number of interactions required for an item to be kept. Default is 5.
    
    Returns
    -------
    pandas.DataFrame
        The filtered dataframe containing only core users and items.
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'user_id': [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4],
    ...     'item_id': ['A', 'B', 'C', 'A', 'D', 'E', 'F', 'F', 'F', 'F', 'F'],
    ...     'rating': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5]
    ... })
    >>> core_data = filter_core_records(data, min_user_interactions=2, min_item_interactions=2)
    >>> core_data
       user_id item_id  rating
    0        1       A       5
    1        1       B       4
    2        1       C       3
    3        2       A       2
    '''
    # Count interactions for users and items
    item_id_count = (
        data[item_id_column]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="item_count")
    )
    user_id_count = (
        data[user_id_column]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="user_count")
    )
    
    # Join counts back to original data
    data = data.join(item_id_count.set_index("unique_values"), on=item_id_column)
    data = data.join(user_id_count.set_index("unique_values"), on=user_id_column)
    
    # Filter based on minimum thresholds
    data = data[data["item_count"] >= min_item_interactions]
    data = data[data["user_count"] >= min_user_interactions]
    
    # Drop the count columns
    data = data.drop(columns=["item_count", "user_count"])
    
    return data

def temporal_train_test_split(
    df: pd.DataFrame,
    test_last_seconds: float,
    time_column: str = "timestamp",
    gap_seconds: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    split_ts = df[time_column].max() - test_last_seconds
    train = df[df[time_column] < split_ts - gap_seconds]
    test = df[df[time_column] >= split_ts]
    return train, test


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