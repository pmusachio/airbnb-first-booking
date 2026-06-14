"""Acquisition layer. The Airbnb competition is consent-gated, so the schema-faithful
synthetic stand-in under data/sample is used unless the real train_users_2.csv is
present in data/raw.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, raw_dir: Path = config.RAW_DIR, sample_path: Path = config.SAMPLE_PATH) -> None:
        self.raw_dir = raw_dir
        self.sample_path = sample_path
        self.real_path = raw_dir / config.REAL_TRAIN_FILENAME
        self.raw_path = raw_dir / "users.csv"

    def download(self, force: bool = False) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.real_path.exists():
            logger.info("Using real competition data at %s", self.real_path)
            return self.real_path
        if self.raw_path.exists() and not force:
            return self.raw_path
        if not self.sample_path.exists():
            raise FileNotFoundError(
                f"No data: place the competition {config.REAL_TRAIN_FILENAME} in {self.raw_dir} "
                f"or restore the synthetic sample at {self.sample_path}.")
        shutil.copyfile(self.sample_path, self.raw_path)
        logger.info("Synthetic stand-in materialized at %s", self.raw_path)
        return self.raw_path

    def load(self) -> pd.DataFrame:
        path = self.real_path if self.real_path.exists() else self.download()
        df = pd.read_csv(path)
        logger.info("Loaded %d rows x %d cols from %s", df.shape[0], df.shape[1], path)
        return df
