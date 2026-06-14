"""Smoke tests for the data contract, leakage guarantees and the serving surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import FeaturePrep, Preprocessor, engineer  # noqa: E402

USER = {"gender": "FEMALE", "age": 34, "signup_method": "facebook", "signup_flow": 0,
        "language": "en", "affiliate_channel": "direct", "affiliate_provider": "direct",
        "first_affiliate_tracked": "untracked", "signup_app": "Web",
        "first_device_type": "Mac Desktop", "first_browser": "Chrome",
        "date_account_created": "2014-05-01", "timestamp_first_active": "20140430120000"}


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH)


def test_leakage_and_id_dropped(sample):
    X, y = Preprocessor().run(sample)
    for col in config.LEAKAGE_COLS + (config.ID_COL,):
        assert col not in X.columns
    assert y.nunique() >= 2


def test_engineering_present(sample):
    eng = engineer(sample.head(200))
    for c in config.ENGINEERED_FEATURES:
        assert c in eng.columns


def test_feature_prep_fixed_columns(sample):
    a = FeaturePrep().fit_transform(sample.head(20))
    b = FeaturePrep().transform(sample.head(5))
    assert list(a.columns) == list(b.columns)


def test_ranking_contract():
    pred = Predictor()
    ranked = pred.rank(USER, k=5)
    assert len(ranked) == 5
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(dest in pred.classes for dest, _ in ranked)
