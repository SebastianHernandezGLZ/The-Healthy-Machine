
"""Unit tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer  # Import ColumnTransformer type hint and instance checks

from src.preprocessing.preprocessing_pipeline import (
    PreprocessingError,  # Custom pipeline exception
    create_preprocessing_pipeline,  # Function that builds the ColumnTransformer pipeline
    fit_preprocessing_pipeline,  # Function that fits the pipeline on training data
    transform_with_pipeline,  # Function that transforms data using a fitted pipeline
)


@pytest.fixture
def sample_data():
    """Create a sample dataset for preprocessing tests."""
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L", "M", "H"],
            "Air temperature [K]": [
                300.0, 301.0, 302.0, 303.0, 304.0, 305.0
            ],
            "Process temperature [K]": [
                310.0, 311.0, 312.0, 313.0, 314.0, 315.0
            ],
            "Rotational speed [rpm]": [
                1500, 1600, 1700, 1800, 1900, 2000
            ],
            "Torque [Nm]": [
                40.0, 42.0, 44.0, 46.0, 48.0, 50.0
            ],
            "Tool wear [min]": [
                10, 20, 30, 40, 50, 60
            ],
        },
        index=[10, 20, 30, 40, 50, 60],  # Custom indices to test index preservation
    )


def test_create_pipeline_returns_column_transformer():
    pipeline = create_preprocessing_pipeline()

    assert isinstance(pipeline, ColumnTransformer)


def test_fit_pipeline_learns_transformations(sample_data):
    """Test that fitting the pipeline initializes transformers and internal attributes."""
    pipeline = fit_preprocessing_pipeline(sample_data)

    assert isinstance(pipeline, ColumnTransformer)
    assert hasattr(pipeline, "transformers_")  # Check that transformers have been fitted


def test_transform_returns_expected_features(sample_data):
    """Test that the transformed output contains all expected categorical, numerical, and engineered columns."""
    pipeline = fit_preprocessing_pipeline(sample_data)

    result = transform_with_pipeline(sample_data, pipeline)

    expected_columns = [
        "Type_H",
        "Type_L",
        "Type_M",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature difference [K]",
        "Power [W]",
    ]

    assert result.columns.tolist() == expected_columns


def test_transform_returns_numeric_features(sample_data):
    """Test that all columns in the transformed output are of numeric data types."""
    pipeline = fit_preprocessing_pipeline(sample_data)

    result = transform_with_pipeline(sample_data, pipeline)

    assert all(
        pd.api.types.is_numeric_dtype(result[column])
        for column in result.columns
    )


def test_transform_preserves_original_index(sample_data):
    pipeline = fit_preprocessing_pipeline(sample_data)

    result = transform_with_pipeline(sample_data, pipeline)

    assert result.index.equals(sample_data.index)


def test_transform_scales_numerical_features(sample_data):
    """Test that numerical and engineered features are properly scaled (mean close to zero, e.g., StandardScalar)."""
    pipeline = fit_preprocessing_pipeline(sample_data)

    result = transform_with_pipeline(sample_data, pipeline)

    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature difference [K]",
        "Power [W]",
    ]

    for column in numerical_columns:
        # Standard scaling should center the mean around 0 (within floating-point tolerance)
        assert abs(result[column].mean()) < 1e-10


def test_transform_does_not_modify_original_data(sample_data):
    original = sample_data.copy(deep=True)

    pipeline = fit_preprocessing_pipeline(sample_data)
    transform_with_pipeline(sample_data, pipeline)

    pd.testing.assert_frame_equal(sample_data, original)


def test_fit_rejects_missing_required_columns(sample_data):
    sample_data = sample_data.drop(
        columns=["Torque [Nm]"]
    )

    with pytest.raises(
        PreprocessingError,
        match="Missing required columns",
    ):
        fit_preprocessing_pipeline(sample_data)


def test_fit_rejects_unknown_categories(sample_data):
    sample_data.loc[10, "Type"] = "X"

    with pytest.raises(
        PreprocessingError,
        match="Unexpected values in Type",
    ):
        fit_preprocessing_pipeline(sample_data)


def test_transform_rejects_unfitted_pipeline(sample_data):
    pipeline = create_preprocessing_pipeline()

    with pytest.raises(
        PreprocessingError,
        match="Transformation failed",
    ):
        transform_with_pipeline(sample_data, pipeline)


def test_pipeline_includes_engineered_features(sample_data):
    pipeline = fit_preprocessing_pipeline(sample_data)

    result = transform_with_pipeline(sample_data, pipeline)

    assert "Temperature difference [K]" in result.columns
    assert "Power [W]" in result.columns


def test_pipeline_preserves_original_data(sample_data):
    original = sample_data.copy(deep=True)

    pipeline = fit_preprocessing_pipeline(sample_data)
    transform_with_pipeline(sample_data, pipeline)

    pd.testing.assert_frame_equal(sample_data, original)
