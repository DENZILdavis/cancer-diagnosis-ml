"""Predict from a CSV using the saved model.

Usage:
    python -m cancer_ml.predict path/to/samples.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_model.joblib"


def predict_frame(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Return predicted label and malignancy probability for each row."""
    bundle = joblib.load(model_path)
    missing = set(bundle["features"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    X = df[bundle["features"]]
    out = pd.DataFrame(index=df.index)
    out["malignant_probability"] = bundle["model"].predict_proba(X)[:, 1].round(4)
    out["prediction"] = (out["malignant_probability"] >= 0.5).map(
        {True: "malignant", False: "benign"}
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tumour class from a CSV.")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    print(predict_frame(pd.read_csv(args.csv)).to_string())


if __name__ == "__main__":
    main()
