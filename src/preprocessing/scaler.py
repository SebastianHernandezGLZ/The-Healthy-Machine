


"""Numerical feature scaling utilities."""


import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
# Estandariza características restando la media y dividiendo por la desviación estándar.
from sklearn.preprocessing import StandardScaler



NUMERICAL_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]



class ScalingError(ValueError):
    """Raised when numerical features cannot be scaled."""



def _validate_input(X: pd.DataFrame) -> None:
    """Validate numerical features before scaling."""
    if not isinstance(X, pd.DataFrame):
        raise ScalingError("Input must be a pandas DataFrame.")


    if X.empty:
        raise ScalingError("Input DataFrame must not be empty.")


    missing_columns = [
        column for column in NUMERICAL_COLUMNS if column not in X.columns
    ]


    if missing_columns:
        raise ScalingError(
            f"Missing numerical columns: {missing_columns}"
        )


    non_numeric_columns = [
        column 
        for column in NUMERICAL_COLUMNS
        if not pd.api.types.is_numeric_dtype(X[column])
    ]


    if non_numeric_columns:
        raise ScalingError(
            f"Expected numerical columns: {non_numeric_columns}"
        )


    values = X[NUMERICAL_COLUMNS].to_numpy(dtype=float)


    if not np.isfinite(values).all():
        raise ScalingError(
            "Nummerical columns must not contain missing or infinite values."
        )



def create_scaler() -> StandardScaler:
    """Create a StandardScaler instance."""
    return StandardScaler()



# Diseñada para calcular la media y la desviación estándar exclusivamente a partir de los datos de entrenamiento (X_train).
def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit the scaler using training data only."""
    _validate_input(X_train)


    scaler = create_scaler()
    scaler.fit(X_train[NUMERICAL_COLUMNS])


    return scaler



def transform_with_scaler(
    X: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Scale numerical features while preserving other columns."""
    _validate_input(X)

    if not isinstance(scaler, StandardScaler):
        raise ScalingError(
            "scaler must be a fitted sklearn StandardScaler."
        )

    try:
        scaled_values = scaler.transform(X[NUMERICAL_COLUMNS])
    except (ValueError, NotFittedError) as exc:
        raise ScalingError(
            "Scaling failed. Ensure the scaler has been fitted."
        ) from exc


    result = X.copy()

    # Replace each scaled column with a float Series.
    for index, column in enumerate(NUMERICAL_COLUMNS):
        result[column] = pd.Series(
            scaled_values[:, index],
            index=result.index,
            dtype="float64",
        )

    return result