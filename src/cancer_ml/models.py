"""Candidate model pipelines."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import RANDOM_STATE


def get_models() -> dict[str, Pipeline]:
    """Return named, ready-to-fit pipelines (scaling is part of each pipeline,
    so cross-validation never leaks test-fold statistics)."""
    return {
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
        ),
        "svm_rbf": Pipeline(
            [("scale", StandardScaler()),
             ("clf", CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False))]
        ),
        "random_forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]
        ),
        "gradient_boosting": Pipeline(
            [("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]
        ),
    }
