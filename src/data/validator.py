"""Dataset validation utilities."""

import pandas as pd


REQUIRED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]

NUMERICAL_COLUMNS = [
    "UDI",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]

VALID_PRODUCT_TYPES = {"L", "M", "H"}
VALID_TARGET_VALUES = {0, 1}


class DatasetValidationError(ValueError):
    """Raised when a dataset fails validation."""


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the structure and basic integrity of a dataset.

    Args:
        df: Dataset to validate.

    Raises:
        DatasetValidationError: If the dataset is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise DatasetValidationError(
            "Dataset must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise DatasetValidationError(
            f"Missing required columns: {missing_columns}"
        )

    required_data = df[REQUIRED_COLUMNS]

    if required_data.isnull().any().any():
        null_columns = (
            required_data.columns[
                required_data.isnull().any()
            ].tolist()
        )

        raise DatasetValidationError(
            f"Missing values found in required columns: "
            f"{null_columns}"
        )

    non_numeric_columns = [
        column
        for column in NUMERICAL_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise DatasetValidationError(
            f"Expected numerical columns: {non_numeric_columns}"
        )

    target_values = set(df["Machine failure"].unique())

    if not target_values.issubset(VALID_TARGET_VALUES):
        raise DatasetValidationError(
            "Machine failure must contain only 0 and 1."
        )

    product_types = set(df["Type"].unique())

    if not product_types.issubset(VALID_PRODUCT_TYPES):
        raise DatasetValidationError(
            f"Unexpected product types: "
            f"{product_types - VALID_PRODUCT_TYPES}"
        )


