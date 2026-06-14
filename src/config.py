"""Central configuration: paths, dataset identity, modeling constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "train.parquet"

# The Airbnb competition data is consent-gated and cannot be redistributed, so a
# schema-faithful synthetic stand-in is versioned here. Drop the real
# train_users_2.csv into data/raw to train on the genuine data instead.
SAMPLE_FILENAME: str = "airbnb_users_synthetic.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME
KAGGLE_DATASET: str = "competitions/airbnb-recruiting-new-user-bookings"
REAL_TRAIN_FILENAME: str = "train_users_2.csv"
SYNTHETIC: bool = True

TARGET: str = "country_destination"
ID_COL: str = "id"
# date_first_booking exists only for users who booked, so it is null exactly when
# the target is NDF -> using it would leak the label. It is dropped.
LEAKAGE_COLS: tuple[str, ...] = ("date_first_booking",)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "gender", "signup_method", "language", "affiliate_channel", "affiliate_provider",
    "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser",
)
NUMERIC_FEATURES: tuple[str, ...] = (
    "age", "signup_flow", "account_year", "account_month", "account_dayofweek", "days_active_before_signup",
)
ENGINEERED_FEATURES: tuple[str, ...] = (
    "account_year", "account_month", "account_dayofweek", "days_active_before_signup",
)

TEST_SIZE: float = 0.2
SEED: int = 42
CV_FOLDS: int = 4
TUNING_ITERS: int = 8
SCORING: str = "neg_log_loss"
TOP_K: int = 5

DRACULA = {
    "background": "#282a36", "current_line": "#44475a", "foreground": "#f8f8f2",
    "comment": "#6272a4", "cyan": "#8be9fd", "green": "#50fa7b", "orange": "#ffb86c",
    "pink": "#ff79c6", "purple": "#bd93f9", "red": "#ff5555", "yellow": "#f1fa8c",
}
