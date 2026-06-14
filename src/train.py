"""Modeling layer: baseline, cross-validated selection and tuning, holdout
evaluation with multiclass and ranking metrics (accuracy, top-5, NDCG@5, macro-F1),
business translation and serialization of a self-contained pipeline plus a card.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import FeaturePrep, build_column_transformer

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"
N_JOBS = 1


def ndcg_at_k(y_true, proba, classes, k=5) -> float:
    order = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    score = 0.0
    cls_index = {c: i for i, c in enumerate(classes)}
    for i, true in enumerate(y_true):
        ti = cls_index.get(true)
        for rank, idx in enumerate(order[i]):
            if idx == ti:
                score += 1.0 / np.log2(rank + 2)
                break
    return float(score / len(y_true))


def top_k_accuracy(y_true, proba, classes, k=5) -> float:
    order = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    cls = {c: i for i, c in enumerate(classes)}
    return float(np.mean([cls.get(t) in row for t, row in zip(y_true, order)]))


@dataclass
class TrainingResult:
    baseline: Dict[str, Any] = field(default_factory=dict)
    cv_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    holdout: Dict[str, Any] = field(default_factory=dict)
    business: Dict[str, Any] = field(default_factory=dict)
    importances: list = field(default_factory=list)


def _model(name):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=config.SEED)
    return HistGradientBoostingClassifier(random_state=config.SEED)


def _pipeline(name):
    return Pipeline([("prep", FeaturePrep()), ("ct", build_column_transformer()), ("clf", _model(name))])


def _params(name):
    if name == "logreg":
        return {"clf__C": np.logspace(-2, 1, 12)}
    return {"clf__learning_rate": np.logspace(-2, -0.3, 8), "clf__max_leaf_nodes": [15, 31, 63],
            "clf__max_depth": [None, 6], "clf__max_iter": [200, 400]}


class ModelTrainer:
    def __init__(self, X, y, data_source: Path | None = None):
        self.data_source = data_source
        self.X_train, self.X_holdout, self.y_train, self.y_holdout = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y)
        self.majority = y.value_counts(normalize=True).idxmax()
        self.majority_rate = float(y.value_counts(normalize=True).max())
        self.result = TrainingResult()

    def fit_baseline(self):
        # majority-class baseline: always predict the most common destination (NDF)
        self.result.baseline = {"model": f"always-{self.majority}", "accuracy": round(self.majority_rate, 4)}
        logger.info("Baseline (always %s) accuracy=%.4f", self.majority, self.majority_rate)
        return self.result.baseline

    def fit(self):
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
        rows = []
        for name in ("logreg", "hist_gb"):
            sc = cross_val_score(_pipeline(name), self.X_train, self.y_train, cv=cv,
                                 scoring=config.SCORING, n_jobs=N_JOBS)
            rows.append({"model": name, "neg_log_loss_mean": round(sc.mean(),4), "neg_log_loss_std": round(sc.std(),4)})
            logger.info("CV %-10s neg_log_loss=%.4f +/- %.4f", name, sc.mean(), sc.std())
        self.result.cv_table = pd.DataFrame(rows).sort_values("neg_log_loss_mean", ascending=False).reset_index(drop=True)
        best = self.result.cv_table.iloc[0]["model"]
        self.result.best_model = best
        search = RandomizedSearchCV(_pipeline(best), _params(best), n_iter=config.TUNING_ITERS,
                                    scoring=config.SCORING, cv=cv, n_jobs=N_JOBS,
                                    random_state=config.SEED, refit=True).fit(self.X_train, self.y_train)
        self.result.best_params = {k: _j(v) for k, v in search.best_params_.items()}
        self.final_pipeline = search.best_estimator_
        logger.info("Tuned %s best CV neg_log_loss=%.4f", best, search.best_score_)
        return self.final_pipeline

    def evaluate(self):
        proba = self.final_pipeline.predict_proba(self.X_holdout)
        classes = list(self.final_pipeline.classes_)
        pred = self.final_pipeline.predict(self.X_holdout)
        y = self.y_holdout.to_numpy()
        self.result.holdout = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "top5_accuracy": round(top_k_accuracy(y, proba, classes, config.TOP_K), 4),
            "ndcg5": round(ndcg_at_k(y, proba, classes, config.TOP_K), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "n_classes": len(classes),
        }
        logger.info("Holdout acc=%.4f top5=%.4f ndcg5=%.4f macroF1=%.4f",
                    *[self.result.holdout[k] for k in ("accuracy", "top5_accuracy", "ndcg5", "macro_f1")])
        return self.result.holdout

    def to_business_metrics(self):
        h = self.result.holdout
        self.result.business = {
            "headline": (f"The model ranks the true first-booking destination in its top 5 for "
                         f"{h['top5_accuracy']*100:.0f}% of users (NDCG@5 {h['ndcg5']:.3f}), versus a "
                         f"majority-class baseline accuracy of {self.majority_rate*100:.0f}%."),
            "top5_accuracy": h["top5_accuracy"], "ndcg5": h["ndcg5"]}
        return self.result.business

    def compute_importances(self):
        n = min(10000, len(self.X_holdout))
        Xs, ys = self.X_holdout.iloc[:n], self.y_holdout.iloc[:n]
        r = permutation_importance(self.final_pipeline, Xs, ys, n_repeats=3,
                                   random_state=config.SEED, scoring=config.SCORING, n_jobs=N_JOBS)
        cols = list(Xs.columns)
        self.result.importances = sorted(
            [{"feature": cols[i], "importance": round(float(r.importances_mean[i]), 5)} for i in range(len(cols))],
            key=lambda d: d["importance"], reverse=True)[:12]
        return self.result.importances

    def save(self):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": SCHEMA_VERSION, "pipeline": self.final_pipeline,
                     "classes": list(self.final_pipeline.classes_), "best_model": self.result.best_model,
                     "majority_class": self.majority, "importances": self.result.importances,
                     "synthetic": config.SYNTHETIC}, config.PIPELINE_PATH)
        logger.info("Pipeline artifact written to %s", config.PIPELINE_PATH)
        card = {"schema_version": SCHEMA_VERSION,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": config.KAGGLE_DATASET, "data_source": "synthetic stand-in" if config.SYNTHETIC else "kaggle",
                "data_sha256": self._hash(), "target": config.TARGET,
                "problem": "multiclass first-booking destination (12 classes, imbalanced)",
                "best_model": self.result.best_model, "best_params": self.result.best_params,
                "cv_selection": self.result.cv_table.to_dict(orient="records"),
                "baseline": self.result.baseline, "holdout": self.result.holdout,
                "business": self.result.business, "top_features": self.result.importances[:8]}
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
        logger.info("Model card written to %s", config.MODEL_CARD_PATH)

    def _hash(self):
        src = self.data_source or config.SAMPLE_PATH
        return hashlib.sha256(Path(src).read_bytes()).hexdigest() if src and Path(src).exists() else "unknown"


def _j(v):
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer): return int(v)
    return v
