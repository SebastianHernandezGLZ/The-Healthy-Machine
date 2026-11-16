
"""Unit tests for the BaseModel class."""

import pandas as pd
import pytest

from src.models.base.base_model import BaseModel  # Import the abstract base class to be tested


class DummyModel(BaseModel):
    """Concrete model used exclusively for testing."""

    def train(self, X_train, y_train):
        """Simulate model training by marking it as trained."""
        self.is_trained = True

    def predict(self, X):
        """Simulate prediction by returning a dummy pandas Series of zeros."""
        return pd.Series([0] * len(X), index=X.index)

    def predict_proba(self, X):
        """Simulate probability prediction by returning a dummy pandas DataFrame."""
        return pd.DataFrame(
            {
                0: [0.8] * len(X),  # 80% de probabilidad de que sea clase 0
                1: [0.2] * len(X),  # 20% de probabilidad de que sea clase 1
            },
            index=X.index,
        )


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel("Base")


def test_model_initialization():
    """Test that a concrete subclass initializes with the correct name, null model, and untrained status."""
    model = DummyModel("Dummy")

    assert model.model_name == "Dummy"
    assert model.model is None
    assert model.is_trained is False


def test_model_rejects_empty_name():
    with pytest.raises(ValueError):
        DummyModel("")


def test_model_rejects_invalid_name():
    with pytest.raises(ValueError):
        DummyModel(None)


def test_get_model_name():
    model = DummyModel("Dummy")

    assert model.get_model_name() == "Dummy"


def test_training_status_before_training():
    model = DummyModel("Dummy")

    assert model.get_training_status() is False


def test_training_status_after_training():
    """Test that the training status changes to True after calling the train method."""
    model = DummyModel("Dummy")

    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])

    model.train(X_train, y_train)

    assert model.get_training_status() is True


def test_predict_returns_series():
    model = DummyModel("Dummy")
    X = pd.DataFrame({"feature": [1, 2, 3]})

    predictions = model.predict(X)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 3


def test_predict_proba_returns_dataframe():
    model = DummyModel("Dummy")
    X = pd.DataFrame({"feature": [1, 2, 3]})

    probabilities = model.predict_proba(X)

    assert isinstance(probabilities, pd.DataFrame)
    assert list(probabilities.columns) == [0, 1]
    assert len(probabilities) == 3
