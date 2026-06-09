DENSE_COLUMNS: tuple[str, ...] = (
    "user_lag_listen_cnt",
    "user_lag_like_cnt",
    "user_lag_full_play_cnt",
    "user_lag_skip_cnt",
    "item_lag_listen_cnt",
    "item_lag_like_cnt",
    "item_lag_full_play_cnt",
    "item_lag_skip_cnt",
    "ui_lag_listen_cnt",
    "ui_lag_like_cnt",
    "ui_lag_full_play_cnt",
    "ui_lag_skip_cnt",
    "user_lag_avg_played_ratio",
    "item_lag_avg_played_ratio",
    "ui_lag_avg_played_ratio",
)
MULTIVALENT_COLUMNS: tuple[str, ...] = ("artist_ids", "album_ids")
SPARSE_COLUMNS: tuple[str, ...] = ("uid", "item_id")
LABEL_COLUMNS: tuple[str, ...] = ("is_like", "is_full_play")

E_TASK_LABEL = "is_like"
C_TASK_LABEL = "is_full_play"
