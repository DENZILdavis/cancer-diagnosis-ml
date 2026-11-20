import pandas as pd
import pytest

from cancer_ml.data import load_data, split_data
from cancer_ml.models import get_models


def test_load_data_shape_and_labels():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 212  # malignant cases in the dataset


def test_split_is_stratified():
    X, y = load_data()
    _, _, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    assert len(y_test) == pytest.approx(0.2 * len(y), abs=1)


@pytest.mark.parametrize("name", list(get_models()))
def test_models_beat_baseline(name):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = get_models()[name].fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.90


def test_predict_frame_roundtrip(tmp_path):
    import joblib

    from cancer_ml.predict import predict_frame

    X, y = load_data()
    X_train, X_test, y_train, _ = split_data(X, y)
    model = get_models()["logistic_regression"].fit(X_train, y_train)
    path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "features": list(X.columns)}, path)
    out = predict_frame(X_test.head(5), model_path=path)
    assert list(out.columns) == ["malignant_probability", "prediction"]
    assert len(out) == 5


def test_predict_missing_columns_raises(tmp_path):
    import joblib

    from cancer_ml.predict import predict_frame

    X, y = load_data()
    model = get_models()["logistic_regression"].fit(X, y)
    path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "features": list(X.columns)}, path)
    with pytest.raises(ValueError):
        predict_frame(pd.DataFrame({"foo": [1]}), model_path=path)
