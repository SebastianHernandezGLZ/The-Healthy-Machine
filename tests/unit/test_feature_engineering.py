
"""Unit tests for feature engineering utilities."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineering import (
    FeatureEngineeringError,
    engineer_features,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H"],
            "Air temperature [K]": [300.0, 310.0, 320.0],
            "Process temperature [K]": [310.0, 325.0, 340.0],
            "Rotational speed [rpm]": [1000.0, 1500.0, 2000.0],
            "Torque [Nm]": [20.0, 30.0, 40.0],
            "Tool wear [min]": [10, 20, 30],
        },
        index=[10, 20, 30],  # Custom row indices
    )


def test_engineer_features_adds_expected_columns(sample_data):
    result = engineer_features(sample_data)

    assert "Temperature difference [K]" in result.columns
    assert "Power [W]" in result.columns


def test_temperature_difference_is_calculated_correctly(sample_data):
    result = engineer_features(sample_data)

    expected = [10.0, 15.0, 20.0]  # Expected values: Process temp - Air temp

    assert result["Temperature difference [K]"].tolist() == expected  # Compare values


def test_power_is_calculated_correctly(sample_data):
    result = engineer_features(sample_data)

    # Expected formula: Torque * Speed * (2 * pi / 60)
    expected = [
        20.0 * 1000.0 * (2 * np.pi / 60),
        30.0 * 1500.0 * (2 * np.pi / 60),
        40.0 * 2000.0 * (2 * np.pi / 60),
    ]

    np.testing.assert_allclose(
        result["Power [W]"].to_numpy(),
        expected,
    )  # Compare floating-point arrays with tolerance


def test_engineering_preserves_original_columns(sample_data):
    result = engineer_features(sample_data)

    for column in sample_data.columns:
        pd.testing.assert_series_equal(
            result[column],
            sample_data[column],
        )


def test_engineering_preserves_original_index(sample_data):
    result = engineer_features(sample_data)

    assert result.index.equals(sample_data.index)


def test_engineering_does_not_modify_original_data(sample_data):
    original = sample_data.copy(deep=True)  # Deep copy to track original state

    engineer_features(sample_data)

    pd.testing.assert_frame_equal(sample_data, original)


def test_engineering_rejects_missing_columns(sample_data):
    sample_data = sample_data.drop(
        columns=["Torque [Nm]"]
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="Missing required columns",
    ):
        engineer_features(sample_data)


def test_engineering_rejects_missing_values(sample_data):
    sample_data.loc[10, "Torque [Nm]"] = np.nan  # Inject a missing value

    with pytest.raises(
        FeatureEngineeringError,
        match="missing values",
    ):
        engineer_features(sample_data)


def test_engineering_rejects_non_numeric_columns(sample_data):
    sample_data["Torque [Nm]"] = ["low", "medium", "high"]

    with pytest.raises(
        FeatureEngineeringError,
        match="Expected numerical columns",
    ):
        engineer_features(sample_data)


def test_engineering_rejects_infinite_values(sample_data):
    sample_data.loc[10, "Torque [Nm]"] = np.inf  # Inject infinity

    with pytest.raises(
        FeatureEngineeringError,
        match="infinite values",
    ):
        engineer_features(sample_data)


def test_engineering_rejects_empty_dataframe(sample_data):
    sample_data = sample_data.iloc[0:0]  # Keep columns, remove all rows

    with pytest.raises(
        FeatureEngineeringError,
        match="must not be empty",
    ):
        engineer_features(sample_data)


def test_engineering_rejects_non_dataframe():
    with pytest.raises(
        FeatureEngineeringError,
        match="pandas DataFrame",
    ):
        engineer_features([1, 2, 3])
