

"Feature engineering utilities for predictive maintenance."


import numpy as np
import pandas as pd



TEMPERATURE_COLUMN = "Air temperature [K]"
PROCESS_TEMPERATURE_COLUMN = "Process temperature [K]"
ROTATIONAL_SPEED_COLUMN = "Rotational speed [rpm]"
TORQUE_COLUMN = "Torque [Nm]"


REQUIRED_COLUMNS = [
    TEMPERATURE_COLUMN,
    PROCESS_TEMPERATURE_COLUMN,
    ROTATIONAL_SPEED_COLUMN,
    TORQUE_COLUMN,
]


ENGINEERED_COLUMNS = [
    "Temperature difference [K]",
    "Power [W]",
]



class FeatureEngineeringError(ValueError):
    """Raised when feature engineering cannot be completed."""



def _validate_input(df: pd.DataFrame) -> None:
    """Validate the input features before engineering."""
    if not isinstance(df, pd.DataFrame):
        raise FeatureEngineeringError(
            "Input must be a pandas DataFrame."
        )


    if df.empty:
        raise FeatureEngineeringError(
            "Input DataFrame must not be empty."
        )


    missing_columns = [
        column 
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]


    if missing_columns:
        raise FeatureEngineeringError(
            f"Missing required columns: {missing_columns}"
        )


    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise FeatureEngineeringError(
            "Required feature columns must not contain missing values."
        )


    non_numeric_columns = [
        column 
        for column in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]


    if non_numeric_columns:
        raise FeatureEngineeringError(
            f"Expected numerical columns: {non_numeric_columns}"
        )


    try:
        numerical_values = df[REQUIRED_COLUMNS].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(
            "Required features must contain valid numbers."
        ) from exc


    if not np.isfinite(numerical_values).all():
        raise FeatureEngineeringError(
            "Required features must not contain infinite values."
        )



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically motivated features without modifying the input.
    
    
    Adds:
        - Temperature difference [K]
        - Power [W]


    The original features and their row indices are preserved.    
    """

    _validate_input(df)


    result = df.copy()


    result["Temperature difference [K]"] = (
        result[PROCESS_TEMPERATURE_COLUMN]
        - result[TEMPERATURE_COLUMN]
    )


    angular_velocity = (
        result[ROTATIONAL_SPEED_COLUMN] * (2 * np.pi / 60)
    )


    result["Power [W]"] = (
        result[TORQUE_COLUMN] * angular_velocity
    )

    # Extraer los valores de las nuevas características creadas para validación numérica
    engineered_values = result[ENGINEERED_COLUMNS].to_numpy(
        dtype = float
    )


    if not np.isfinite(engineered_values).all():
        raise FeatureEngineeringError(
            "Feature engineering produced non-finite values."
        )


    return result