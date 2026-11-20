"""Train, compare, evaluate and save the best model.

Usage:
    python -m cancer_ml.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .data import RANDOM_STATE, load_data, split_data
from .models import get_models

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def compare_models(X_train, y_train) -> pd.DataFrame:
    """5-fold stratified CV on the training set for every candidate."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in get_models().items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({"model": name, **{m: res[f"test_{m}"].mean() for m in scoring}})
    return pd.DataFrame(rows).sort_values("recall", ascending=False).round(4)


def main() -> dict:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 1. Compare candidates with cross-validation
    cv_table = compare_models(X_train, y_train)
    cv_table.to_csv(REPORTS / "cv_results.csv", index=False)
    print("Cross-validation results:\n", cv_table.to_string(index=False))

    # 2. Pick best by F1 (balances missed cancers vs false alarms)
    best_name = cv_table.sort_values("f1", ascending=False).iloc[0]["model"]
    best = get_models()[best_name].fit(X_train, y_train)

    # 3. Final evaluation on the untouched test set
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = {
        "best_model": best_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nTest metrics:", metrics)
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))

    # 4. Plots
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["benign", "malignant"], ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix – {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(best, X_test, y_test, ax=ax)
    ax.set_title(f"ROC curve – {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS / "roc_curve.png", dpi=150)
    plt.close(fig)

    imp = permutation_importance(
        best, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="f1"
    )
    top = (
        pd.Series(imp.importances_mean, index=X.columns).sort_values().tail(10)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top.plot.barh(ax=ax, color="#2a6f97")
    ax.set_title("Top 10 features (permutation importance, F1)")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    # 5. Persist model with the feature order it expects
    joblib.dump({"model": best, "features": list(X.columns)}, MODELS / "best_model.joblib")
    print(f"\nSaved model to {MODELS / 'best_model.joblib'}")
    return metrics


if __name__ == "__main__":
    main()
