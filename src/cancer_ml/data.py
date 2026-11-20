"""Data loading and splitting."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Wisconsin breast cancer dataset (bundled with scikit-learn).

    Target: 1 = malignant, 0 = benign.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    # sklearn encodes malignant=0; flip so that 1 means malignant (positive class).
    y = (1 - raw.target).rename("malignant")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
