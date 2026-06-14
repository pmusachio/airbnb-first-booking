"""Serving layer: load the serialized pipeline and rank a new user's likely
first-booking destinations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact_path: Path = config.PIPELINE_PATH) -> None:
        import joblib

        if not Path(artifact_path).exists():
            raise FileNotFoundError(f"No artifact at {artifact_path}. Run `python -m src.pipeline` first.")
        art = joblib.load(artifact_path)
        self.pipeline = art["pipeline"]
        self.classes: List[str] = art["classes"]
        self.importances: List[Dict[str, Any]] = art.get("importances", [])
        self.synthetic: bool = art.get("synthetic", False)
        self.majority_class: str = art.get("majority_class", "NDF")

    def rank(self, features: Dict[str, Any], k: int = 5) -> List[Tuple[str, float]]:
        proba = self.pipeline.predict_proba(pd.DataFrame([features]))[0]
        order = np.argsort(proba)[::-1][:k]
        return [(self.classes[i], float(proba[i])) for i in order]

    def top_features(self, n: int = 6) -> List[Dict[str, Any]]:
        return self.importances[:n]
