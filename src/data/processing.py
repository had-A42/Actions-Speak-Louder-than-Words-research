import abc
import logging
import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .dataprep import filter_core_records, transform_indices
from .utils import download_file

class DataProcessor:
    def __init__(
        self,
        prefix: str,
        expected_num_unique_users: int | None = None,
        expected_num_unique_items: int | None = None
    ) -> None:
        self._prefix: str = prefix
        self._expected_num_unique_users = expected_num_unique_users
        self._expected_num_unique_items = expected_num_unique_items
    

    @property
    def prefix(self) -> str:
        return self._prefix
    
    @property
    def expected_num_unique_users(self) -> int | None:
        return self._expected_num_unique_users
    
    @property
    def expected_num_unique_items(self) -> int | None:
        return self._expected_num_unique_items
    
    @property
    def output_csv(self) -> str:
        return f"tmp/{self.prefix}/seq_format.csv"
    
    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        sequence_columns: list[str],
        user_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if user_data is not None:
            result = ratings_data.join(
                user_data.set_index("user_id"),
                on="user_id",
            )
        else:
            result = ratings_data.copy()

        for column in sequence_columns:
            result[column] = result[column].apply(
                lambda values: ",".join(map(str, values))
            )

        return result
    
    def _preprocess_logging(self, ratings: pd.DataFrame):
        print(f"{self._prefix} #data points before filter: {ratings.shape[0]}")
        print(
            f"{self._prefix} #user before filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item before filter: {len(set(ratings['item_id'].values))}"
        )

    def _postprocess_logging(self, ratings):
        print(f"{self._prefix} #data points after filter: {ratings.shape[0]}")
        print(
            f"{self._prefix} #user after filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item after filter: {len(set(ratings['item_id'].values))}"
        )

class AmazonDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        expected_num_unique_users: int | None = None,
        expected_num_unique_items: int | None = None,
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_num_unique_users=expected_num_unique_users,
        )
        self._download_path = download_path
        self._saved_name = saved_name

    def download(self) -> None:
        if not os.path.exists(self._saved_name):
            download_file(
                self._download_path,
                self._saved_name,
            )

    def preprocess_rating(self):
        self.download()

        ratings = pd.read_csv(
            self._saved_name,
            sep=",",
            names=["user_id", "parent_asin", "rating", "timestamp"],
        )

        ratings.rename(columns={
            "parent_asin": "item_id",
            },
            inplace=True
        )
        
        # Convert timestamp to numeric to handle mixed types
        ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
        
        self._preprocess_logging(ratings)

        ratings = filter_core_records(
            ratings,
            user_id_column="user_id",
            item_id_column="item_id",
            min_user_interactions=5,
            min_item_interactions=5
        )

        ratings, _ = transform_indices(
            ratings,
            users="user_id",
            items="item_id",
            inplace=True
        )

        self._postprocess_logging(ratings)
        
        ratings_group = ratings.sort_values(by=["timestamp"]).groupby("user_id")

        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.item_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.timestamp.apply(list)),
            }
        )

        seq_ratings_data = self.to_seq_data(seq_ratings_data, ["item_ids", "ratings", "timestamps"])
        seq_ratings_data.reset_index(drop=True).to_csv(
            self.output_csv, index=False, sep=","
        )

        if self.expected_num_unique_items is not None:
            num_unique_items = len(set(ratings['item_id'].values))
            assert self.expected_num_unique_items == num_unique_items, (
                f"expected: {self.expected_num_unique_items}, actual: {num_unique_items}"
            )
        
        if self.expected_num_unique_users is not None:
            num_unique_users = len(set(ratings['user_id'].values))
            assert self.expected_num_unique_users == num_unique_users, (
                f"expected: {self.expected_num_unique_users}, actual: {num_unique_users}"
            )

def get_common_preprocessors() -> dict[str, AmazonDataProcessor]:
    amzn_books_dp = AmazonDataProcessor(
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Books.csv.gz",
        "tmp/ratings_Books.csv.gz",
        prefix="amzn_books",
    )
    return {
        "amzn-books": amzn_books_dp,
    }