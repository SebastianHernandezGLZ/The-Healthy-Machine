
"""Unit tests for LogisticRegressionModel."""

import numpy as np
import pandas as pd
import pytest

from src.models.implementations.logistic_regression_model import (
    LogisticRegressionModel,
    ModelTrainingError,
    ModelPredictionError,
)


@pytest.fixture
def sample_data():
    """Fixture that provides a synthetic dataset (features X and target y) for testing."""
    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature_2": [8, 7, 6, 5, 4, 3, 2, 1],
        },
        index=range(8),
    )

    # Create a target Series with a balanced binary classification (four 0s and four 1s)
    y = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1],
        index=range(8),
    )

    return X, y


def test_model_initialization():
    model = LogisticRegressionModel()

    assert model.model_name == "Logistic Regression"  # Verify name inherited from BaseModel
    assert model.is_trained is False  # Verify model starts untrained
    assert model.model is not None  # Verify underlying scikit-learn estimator is created


def test_model_rejects_invalid_max_iter():
    """Test that initializing with an invalid max_iter (e.g., 0) raises a ValueError."""
    with pytest.raises(ValueError):
        LogisticRegressionModel(max_iter=0)


def test_model_training(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()

    model.train(X, y)

    assert model.is_trained is True


def test_model_prediction(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()

    model.train(X, y)
    predictions = model.predict(X)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X)
    assert set(predictions.unique()).issubset({0, 1})  # Verify classes are strictly binary (0 or 1)


def test_model_prediction_preserves_index(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()

    model.train(X, y)
    predictions = model.predict(X)

    assert predictions.index.equals(X.index)


def test_model_predict_proba(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()

    model.train(X, y)
    probabilities = model.predict_proba(X)

    assert isinstance(probabilities, pd.DataFrame)
    assert list(probabilities.columns) == [0, 1]  # Verify column labels represent classes 0 and 1
    assert len(probabilities) == len(X)  # Verify row count matches input


def test_model_probabilities_are_valid(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()

    model.train(X, y)
    probabilities = model.predict_proba(X)

    assert np.all(probabilities.to_numpy() >= 0)
    assert np.all(probabilities.to_numpy() <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Row probabilities must sum to 1.0


def test_prediction_before_training_raises_error(sample_data):
    """Test that calling predict on an untrained model raises a ModelPredictionError."""
    X, _ = sample_data
    model = LogisticRegressionModel()  # Not trained

    with pytest.raises(ModelPredictionError):
        model.predict(X)


def test_predict_proba_before_training_raises_error(sample_data):
    X, _ = sample_data
    model = LogisticRegressionModel()  # Not trained

    with pytest.raises(ModelPredictionError):
        model.predict_proba(X)


def test_training_rejects_mismatched_lengths(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()

    with pytest.raises(ModelTrainingError):
        model.train(X, y.iloc[:-1])  # Drop the last element of y to create a mismatch


def test_training_rejects_single_class_target(sample_data):
    X, _ = sample_data
    y = pd.Series([0] * len(X), index=X.index)  # All elements are 0 (only 1 class)
    model = LogisticRegressionModel()

    with pytest.raises(ModelTrainingError):
        model.train(X, y)


def test_training_rejects_missing_values(sample_data):
    X, y = sample_data
    X.loc[0, "feature_1"] = np.nan  # Inject a missing value
    model = LogisticRegressionModel()

    with pytest.raises(ModelTrainingError):
        model.train(X, y)


def test_prediction_rejects_missing_values(sample_data):
    """Test that predicting with feature data containing missing values raises an error."""
    X, y = sample_data
    model = LogisticRegressionModel()
    model.train(X, y)  # Train successfully first

    invalid_X = X.copy()
    invalid_X.loc[0, "feature_1"] = np.nan  # Inject missing value in prediction data

    with pytest.raises(ModelPredictionError):
        model.predict(invalid_X)
