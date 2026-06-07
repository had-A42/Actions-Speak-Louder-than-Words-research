from typing import Dict
import pandas as pd
from polara import get_movielens_data
from src.data.dataprep import transform_indices, verify_time_split, reindex_data, temporal_train_test_split
from huggingface_hub import hf_hub_download



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

def load_yambda_lag(interactions_path: str | None, config: Dict) -> pd.DataFrame:
    path = interactions_path or hf_hub_download(
        repo_id="matfu21/yambda-50m-lag-features",
        repo_type="dataset",
        filename="listens.parquet",
    )
    df = pd.read_parquet(path).rename(columns=config["col_mapping"])

    artist_path = config.get("artist_mapping_path") or hf_hub_download(
        repo_id="yandex/yambda",
        repo_type="dataset",
        filename="artist_item_mapping.parquet",
    )

    artists = pd.read_parquet(artist_path, columns=["item_id", "artist_id"])

    artists = (
        artists
        .sort_values(["item_id", "artist_id"])
        .drop_duplicates("item_id")
    )

    df = df.merge(artists, on="item_id", how="left")
    df["artist_id"] = df["artist_id"].fillna(-1).astype("int64")

    df["feedback"] = 1
    df["action_code"] = df["is_like"].astype("int8") + 2 * df["is_full_play"].astype("int8")
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
