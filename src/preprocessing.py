"""Transformation layer. Date-derived features and age cleaning live in a custom
first pipeline step so training and serving share the identical transform. The
booking-date column is dropped to prevent target leakage.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    age = pd.to_numeric(out.get("age"), errors="coerce")
    # some users enter a birth year instead of an age
    is_year = age.between(1920, 2015)
    age = age.mask(is_year, 2015 - age)
    out["age"] = age.where(age.between(18, 95))
    acct = pd.to_datetime(out.get("date_account_created"), errors="coerce")
    out["account_year"] = acct.dt.year
    out["account_month"] = acct.dt.month
    out["account_dayofweek"] = acct.dt.dayofweek
    active = pd.to_datetime(out.get("timestamp_first_active").astype(str),
                            format="%Y%m%d%H%M%S", errors="coerce")
    out["days_active_before_signup"] = (acct - active).dt.days.clip(lower=0)
    for c in config.CATEGORICAL_FEATURES:
        if c in out:
            col = out[c].astype(object)
            out[c] = col.where(col.notna(), np.nan).replace({"-unknown-": np.nan, "<NA>": np.nan})
    return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        df = engineer(pd.DataFrame(X).copy())
        cols = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]


def build_column_transformer() -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False))])
    return ColumnTransformer(
        [("num", numeric_pipe, list(config.NUMERIC_FEATURES)),
         ("cat", cat_pipe, list(config.CATEGORICAL_FEATURES))], remainder="drop")


class Preprocessor:
    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if config.TARGET not in df.columns:
            raise ValueError(f"Target '{config.TARGET}' missing")
        d = df.drop(columns=[c for c in config.LEAKAGE_COLS if c in df.columns], errors="ignore")
        y = d[config.TARGET].astype(str)
        X = d.drop(columns=[config.TARGET] + ([config.ID_COL] if config.ID_COL in d.columns else []))
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        eng = engineer(d).copy()
        eng[config.TARGET] = y.values
        eng.head(50000).to_parquet(self.processed_path, index=False)
        logger.info("Prepared %d users, %d destination classes", len(X), y.nunique())
        return X, y
