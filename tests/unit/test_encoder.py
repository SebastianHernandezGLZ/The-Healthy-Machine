
"""Unit tests for categorical feature encoding."""

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing.encoder import (
    EncodingError,
    create_encoder,
    fit_encoder,
    transform_with_encoder,
)


# Una fixture es una función que prepara un conjunto de datos o entorno reutilizable para las pruebas.
@pytest.fixture
def training_data():
    """Create representative training data."""
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L", "M", "H"],
            "Air temperature [K]": [
                298.1, 298.2, 298.3, 298.4, 298.5, 298.6
            ],
            "Torque [Nm]": [40, 45, 50, 42, 47, 52],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def test_create_encoder_returns_one_hot_encoder():
    encoder = create_encoder()

    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"


def test_fit_encoder_learns_categories(training_data):
    encoder = fit_encoder(training_data)

    # Verifica que la función fit_encoder() procese los datos de entrenamiento y aprenda las categorías correctas en orden ordenado alfabéticamente por defecto en scikit-learn (["H", "L", "M"]).
    assert encoder.categories_[0].tolist() == ["H", "L", "M"]


def test_transform_encodes_type_and_preserves_numeric_columns(
    training_data,
):
    encoder = fit_encoder(training_data)

    result = transform_with_encoder(training_data, encoder)

    expected_columns = [
        "Type_H",
        "Type_L",
        "Type_M",
        "Air temperature [K]",
        "Torque [Nm]",
    ]

    # Valida mediante aserciones puntuales en el índice 10 y 12 que los valores binarios (one-hot) se asignen correctamente según la categoría original de cada fila.
    assert result.columns.tolist() == expected_columns
    assert result.loc[10, "Type_L"] == 1
    assert result.loc[10, "Type_H"] == 0
    assert result.loc[10, "Type_M"] == 0
    assert result.loc[12, "Type_H"] == 1


def test_transform_preserves_original_index(training_data):
    encoder = fit_encoder(training_data)

    result = transform_with_encoder(training_data, encoder)

    assert result.index.equals(training_data.index)


def test_transform_does_not_modify_original_data(training_data):
    original = training_data.copy(deep=True)
    encoder = fit_encoder(training_data)

    transform_with_encoder(training_data, encoder)

    pd.testing.assert_frame_equal(training_data, original)


def test_transform_handles_unknown_categories(training_data):
    encoder = fit_encoder(training_data)

    new_data = pd.DataFrame(
        {
            "Type": ["X"],
            "Air temperature [K]": [299.0],
            "Torque [Nm]": [48],
        },
        index=[20],
    )

    result = transform_with_encoder(new_data, encoder)

    # Verifica que, al ignorar categorías desconocidas, todas las columnas one-hot sumen 0 para esa fila (ninguna se active) y que las características numéricas se mantengan intactas.
    assert result.loc[20, ["Type_H", "Type_L", "Type_M"]].sum() == 0
    assert result.loc[20, "Torque [Nm]"] == 48


def test_fit_encoder_rejects_unknown_categories(training_data):
    invalid_data = training_data.copy()
    invalid_data.loc[10, "Type"] = "X"

    with pytest.raises(EncodingError, match="Unexpected values"):
        fit_encoder(invalid_data)


# Comprueba que se rechacen datos a los que les falta la columna categórica obligatoria ("Type").
def test_encoder_rejects_missing_categorical_column():
    invalid_data = pd.DataFrame({"Torque [Nm]": [40]})

    with pytest.raises(EncodingError, match="Missing categorical columns"):
        fit_encoder(invalid_data)


def test_encoder_rejects_missing_values(training_data):
    invalid_data = training_data.copy()
    invalid_data.loc[10, "Type"] = None

    with pytest.raises(EncodingError, match="missing values"):
        fit_encoder(invalid_data)


def test_transform_rejects_unfitted_encoder(training_data):
    encoder = create_encoder()

    with pytest.raises(EncodingError, match="Encoding failed"):
        transform_with_encoder(training_data, encoder)
