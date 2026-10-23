
"""Unit tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.preprocessing.preprocessing_pipeline import (
    PreprocessingError,
    create_preprocessing_pipeline,
    fit_preprocessing_pipeline,
    transform_with_pipeline,
)


@pytest.fixture
def training_data():
    """Create representative training data."""
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L", "M", "H"],
            "Air temperature [K]": [
                298.1, 298.2, 298.3, 298.4, 298.5, 298.6
            ],
            "Process temperature [K]": [
                308.1, 308.3, 308.5, 308.7, 308.9, 309.1
            ],
            "Rotational speed [rpm]": [
                1400, 1500, 1600, 1700, 1800, 1900
            ],
            "Torque [Nm]": [40, 45, 50, 55, 60, 65],
            "Tool wear [min]": [10, 20, 30, 40, 50, 60],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def test_create_pipeline_returns_column_transformer():
    pipeline = create_preprocessing_pipeline()

    assert isinstance(pipeline, ColumnTransformer)


def test_fit_pipeline_learns_transformations(training_data):
    pipeline = fit_preprocessing_pipeline(training_data)

    assert pipeline.named_transformers_["type_encoder"].categories_[0].tolist() == [
        "H", "L", "M"
    ]
    assert pipeline.named_transformers_["numerical_scaler"].n_features_in_ == 5


def test_transform_returns_expected_features(training_data):
    pipeline = fit_preprocessing_pipeline(training_data)

    result = transform_with_pipeline(training_data, pipeline)

    expected_columns = [
        "Type_H",
        "Type_L",
        "Type_M",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    assert result.columns.tolist() == expected_columns
    assert result.shape == (len(training_data), 8)


def test_transform_returns_numeric_features(training_data):
    pipeline = fit_preprocessing_pipeline(training_data)

    result = transform_with_pipeline(training_data, pipeline)

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in result.dtypes
    )


def test_transform_preserves_original_index(training_data):
    pipeline = fit_preprocessing_pipeline(training_data)

    result = transform_with_pipeline(training_data, pipeline)

    assert result.index.equals(training_data.index)


def test_transform_scales_numerical_features(training_data):
    pipeline = fit_preprocessing_pipeline(training_data)

    result = transform_with_pipeline(training_data, pipeline)

    numerical_result = result[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
    ]

    # Extrae las columnas numéricas transformadas y utiliza np.allclose para comprobar que la media de cada una sea 0.0 y su desviación estándar (ddof=0) sea 1.0, confirmando que el escalado dentro del pipeline se aplicó de forma correcta.
    assert np.allclose(numerical_result.mean().to_numpy(), 0.0)
    assert np.allclose(numerical_result.std(ddof=0).to_numpy(), 1.0)


def test_transform_does_not_modify_original_data(training_data):
    original = training_data.copy(deep=True)
    pipeline = fit_preprocessing_pipeline(training_data)

    transform_with_pipeline(training_data, pipeline)

    pd.testing.assert_frame_equal(training_data, original)


def test_fit_rejects_missing_required_columns():
    invalid_data = pd.DataFrame({"Type": ["L"]})

    with pytest.raises(PreprocessingError, match="Missing required columns"):
        fit_preprocessing_pipeline(invalid_data)


def test_fit_rejects_unknown_categories(training_data):
    invalid_data = training_data.copy()
    invalid_data.loc[10, "Type"] = "X"

    with pytest.raises(PreprocessingError, match="Unexpected values in Type"):
        fit_preprocessing_pipeline(invalid_data)


def test_transform_rejects_unfitted_pipeline(training_data):
    pipeline = create_preprocessing_pipeline()

    with pytest.raises(PreprocessingError, match="Transformation failed"):
        transform_with_pipeline(training_data, pipeline)
