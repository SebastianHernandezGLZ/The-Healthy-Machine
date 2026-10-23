
"""Unit tests for numerical feature scaling."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing.scaler import (
    NUMERICAL_COLUMNS,
    ScalingError,
    create_scaler,
    fit_scaler,
    transform_with_scaler,
)


@pytest.fixture
def training_data():
    """Create representative training data."""
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L", "M"],
            "Air temperature [K]": [298.1, 298.2, 298.3, 298.4, 298.5],
            "Process temperature [K]": [
                308.1, 308.3, 308.5, 308.7, 308.9
            ],
            "Rotational speed [rpm]": [1400, 1500, 1600, 1700, 1800],
            "Torque [Nm]": [40, 45, 50, 55, 60],
            "Tool wear [min]": [10, 20, 30, 40, 50],
        },
        index=[10, 11, 12, 13, 14],
    )


def test_create_scaler_returns_standard_scaler():
    scaler = create_scaler()

    assert isinstance(scaler, StandardScaler)


def test_fit_scaler_learns_numerical_features(training_data):
    scaler = fit_scaler(training_data)

    # Registre exactamente la cantidad correcta de características de entrada (n_features_in_).
    assert scaler.n_features_in_ == len(NUMERICAL_COLUMNS)
    # Calcule y almacene las medias correspondientes para cada una de las columnas numéricas (mean_.shape).
    assert scaler.mean_.shape == (len(NUMERICAL_COLUMNS),)


def test_transform_scales_numerical_features(training_data):
    scaler = fit_scaler(training_data)

    result = transform_with_scaler(training_data, scaler)

    scaled_values = result[NUMERICAL_COLUMNS].to_numpy()

    # Utiliza np.allclose para comprobar que la media de cada columna escalada sea prácticamente 0.0 y la desviación estándar sea prácticamente 1.0 (propiedad fundamental de la estandarización).
    assert np.allclose(scaled_values.mean(axis=0), 0.0)
    assert np.allclose(scaled_values.std(axis=0), 1.0)


def test_transform_preserves_non_numerical_columns(training_data):
    scaler = fit_scaler(training_data)

    result = transform_with_scaler(training_data, scaler)

    # Asegura que las columnas no numéricas (como "Type") se mantengan totalmente intactas después de aplicar el escalado.  
    pd.testing.assert_series_equal(
        result["Type"],
        training_data["Type"],
    )


def test_transform_preserves_columns_and_index(training_data):
    scaler = fit_scaler(training_data)

    result = transform_with_scaler(training_data, scaler)

    assert result.columns.tolist() == training_data.columns.tolist()
    assert result.index.equals(training_data.index)


def test_transform_does_not_modify_original_data(training_data):
    original = training_data.copy(deep=True)
    scaler = fit_scaler(training_data)

    transform_with_scaler(training_data, scaler)

    pd.testing.assert_frame_equal(training_data, original)


def test_scaler_rejects_missing_numerical_column():
    invalid_data = pd.DataFrame({"Type": ["L"]})

    with pytest.raises(ScalingError, match="Missing numerical columns"):
        fit_scaler(invalid_data)


def test_scaler_rejects_missing_values(training_data):
    invalid_data = training_data.copy()
    invalid_data.loc[10, "Torque [Nm]"] = np.nan

    with pytest.raises(ScalingError, match="missing or infinite"):
        fit_scaler(invalid_data)


def test_scaler_rejects_infinite_values(training_data):
    invalid_data = training_data.copy()
    invalid_data["Torque [Nm]"] = invalid_data["Torque [Nm]"].astype(float)
    invalid_data.loc[10, "Torque [Nm]"] = np.inf

    with pytest.raises(ScalingError, match="missing or infinite"):
        fit_scaler(invalid_data)


def test_transform_rejects_unfitted_scaler(training_data):
    scaler = create_scaler()

    with pytest.raises(ScalingError, match="Scaling failed"):
        transform_with_scaler(training_data, scaler)
