import json
import os
from zipfile import ZipFile

import pandas as pd

from .dataprep import (
    filter_core_records,
    reindex_data,
    temporal_train_test_split,
    transform_indices,
    verify_time_split,
)
from .schema import DatasetMetadata, FeatureSpec, dataset_artifacts
from .sequence_store import save_index_mapping, write_sequences
from .utils import download_file

class DataProcessor:
    def __init__(
        self,
        prefix: str,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        expected_num_unique_users: int | None = None,
        expected_num_unique_items: int | None = None
    ) -> None:
        self._prefix: str = prefix
        self._dataset_name = dataset_name or prefix
        self._dataset_version = dataset_version or "unknown"
        self._expected_num_unique_users = expected_num_unique_users
        self._expected_num_unique_items = expected_num_unique_items
        self._artifacts = dataset_artifacts(prefix)
    

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_version(self) -> str:
        return self._dataset_version
    
    @property
    def expected_num_unique_users(self) -> int | None:
        return self._expected_num_unique_users
    
    @property
    def expected_num_unique_items(self) -> int | None:
        return self._expected_num_unique_items

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

    def _save_seq_data(
        self,
        seq_ratings_data: pd.DataFrame,
        sequence_columns: list[str],
    ) -> None:
        write_sequences(
            sequences=seq_ratings_data,
            artifacts=self._artifacts,
            sequence_columns=sequence_columns,
        )

    def _save_data_description(self, data_description: dict) -> None:
        self._artifacts.root.mkdir(parents=True, exist_ok=True)
        with open(self._artifacts.metadata_json, "w", encoding="utf-8") as file:
            json.dump(data_description, file, indent=2)

    def _save_index_mappings(self, data_index: dict) -> None:
        self._artifacts.root.mkdir(parents=True, exist_ok=True)
        if "users" in data_index:
            save_index_mapping(data_index["users"], self._artifacts.user_mapping_csv)
        if "items" in data_index:
            save_index_mapping(data_index["items"], self._artifacts.item_mapping_csv)

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
            dataset_name="amzn-books",
            dataset_version="amazon-2023-books-0core",
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

        ratings, data_index = transform_indices(
            ratings,
            users="user_id",
            items="item_id",
            inplace=True
        )
        test_last_seconds = float(
            ratings["timestamp"].max() - ratings["timestamp"].quantile(0.9)
        )
        split_timestamp = float(ratings["timestamp"].max() - test_last_seconds)

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

        self._save_seq_data(seq_ratings_data, ["item_ids", "ratings", "timestamps"])
        self._save_index_mappings(data_index)

        metadata = DatasetMetadata(
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            n_users=len(data_index["users"]),
            n_items=len(data_index["items"]),
            min_user_interactions=5,
            min_item_interactions=5,
            split_name="global_temporal",
            split_timestamp=split_timestamp,
            feedback_column="rating",
            timestamp_column="timestamp",
            notes=[
                "Amazon Books 2023 has no dense or categorical side features in the current rating-only source.",
                "Item and user feature files are optional artifacts for future datasets.",
            ],
        )
        self._save_data_description(metadata.to_dict())

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


class MovieLensDataProcessor(DataProcessor):
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
            dataset_name="ml-20m",
            dataset_version="movielens-20m",
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

    def _read_ratings(self) -> pd.DataFrame:
        with ZipFile(self._saved_name) as movielens_zip:
            with movielens_zip.open("ml-20m/ratings.csv") as ratings_file:
                ratings = pd.read_csv(ratings_file)

        ratings.rename(
            columns={
                "userId": "user_id",
                "movieId": "item_id",
            },
            inplace=True,
        )
        ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
        return ratings

    def preprocess_rating(self):
        self.download()

        ratings = self._read_ratings()
        self._preprocess_logging(ratings)

        test_last_seconds = float(
            ratings["timestamp"].max() - ratings["timestamp"].quantile(0.9)
        )
        split_timestamp = float(ratings["timestamp"].max() - test_last_seconds)
        train_raw, test_raw = temporal_train_test_split(
            ratings,
            timeid="timestamp",
            test_last_seconds=test_last_seconds,
        )

        train, data_index = transform_indices(
            train_raw,
            users="user_id",
            items="item_id",
        )
        test = reindex_data(
            test_raw,
            data_index,
            entities=["users", "items"],
            filter_invalid=True,
        )

        verify_time_split(
            train,
            test,
            target_field="user_id",
            timeid="timestamp",
        )

        ratings = pd.concat([train, test], axis=0, ignore_index=True)
        self._postprocess_logging(ratings)

        ratings_group = ratings.sort_values(
            by=["user_id", "timestamp"],
            ascending=True,
        ).groupby("user_id")

        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.item_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.timestamp.apply(list)),
            }
        )
        self._save_seq_data(seq_ratings_data, ["item_ids", "ratings", "timestamps"])

        data_description = {
            **DatasetMetadata(
                dataset_name=self.dataset_name,
                dataset_version=self.dataset_version,
                n_users=len(data_index["users"]),
                n_items=len(data_index["items"]),
                split_name="global_temporal",
                split_timestamp=split_timestamp,
                feedback_column="rating",
                timestamp_column="timestamp",
                item_features=[
                    FeatureSpec(
                        name="genres",
                        kind="multi_categorical",
                        source="item",
                        description="Available in MovieLens movies.csv; extraction is planned.",
                    ),
                    FeatureSpec(
                        name="release_year",
                        kind="categorical",
                        source="item",
                        description="Available from MovieLens titles; extraction is planned.",
                    ),
                ],
                notes=[
                    "MovieLens can provide item categorical features; current pipeline only stores interactions.",
                ],
            ).to_dict(),
            "users": data_index["users"].name,
            "items": data_index["items"].name,
        }
        self._save_data_description(data_description)
        self._save_index_mappings(data_index)

        if self.expected_num_unique_items is not None:
            num_unique_items = len(set(ratings["item_id"].values))
            assert self.expected_num_unique_items == num_unique_items, (
                f"expected: {self.expected_num_unique_items}, actual: {num_unique_items}"
            )

        if self.expected_num_unique_users is not None:
            num_unique_users = len(set(ratings["user_id"].values))
            assert self.expected_num_unique_users == num_unique_users, (
                f"expected: {self.expected_num_unique_users}, actual: {num_unique_users}"
            )


def get_common_preprocessors() -> dict[str, DataProcessor]:
    amzn_books_dp = AmazonDataProcessor(
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/0core/rating_only/Books.csv.gz",
        "tmp/ratings_Books.csv.gz",
        prefix="amzn_books",
    )
    movielens_20m_dp = MovieLensDataProcessor(
        "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "tmp/ml-20m.zip",
        prefix="ml-20m",
    )
    return {
        "amzn-books": amzn_books_dp,
        "ml-20m": movielens_20m_dp,
    }
