from typing import Dict
import pandas as pd
from polara import get_movielens_data
from src.data.dataprep import transform_indices, verify_time_split, reindex_data, temporal_train_test_split



def load_ml20m(interactions_path: str, config: Dict) -> pd.DataFrame:
    df = get_movielens_data(include_time=True, get_genres=False, local_file=interactions_path)
    df = df.rename(columns=config["col_mapping"])
    return df


def load_yambda(interactions_path: str, config: Dict) -> pd.DataFrame:
    df = pd.read_parquet(interactions_path, columns=list(config["col_mapping"].keys()))
    df = df.rename(columns=config["col_mapping"])
    df["feedback"] = 1
    return df


def load_amzn_books(interactions_path: str, config: Dict) -> pd.DataFrame:
    df = pd.read_csv(interactions_path)
    df = df.rename(columns=config["col_mapping"])
    return df



def split_and_reindex(df: pd.DataFrame, config: Dict):
    train_, test_ = temporal_train_test_split(df, test_last_seconds=df['timestamp'].max() - df['timestamp'].quantile(1 - config["test_quantile"]), gap_seconds=15 * 60)
    
    train, data_index = transform_indices(train_, users='user_id', items='item_id')
    test = reindex_data(test_, data_index, entities=['users', 'items'], filter_invalid=True)
    
    verify_time_split(train, test, target_field='user_id', timeid='timestamp')


    data_description = dict(
        users=data_index['users'].name,
        items=data_index['items'].name,
        timestamp='timestamp',
        feedback='feedback',
        n_users=len(data_index['users']),
        n_items=len(data_index['items']),
    )

    return train, test, data_description