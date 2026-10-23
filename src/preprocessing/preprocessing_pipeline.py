
"""Preprocessing pipeline for categorical and numerical features."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from src.preprocessing.encoder import (
    CATEGORICAL_COLUMNS,
    TYPE_CATEGORIES,
    create_encoder,
)
from src.preprocessing.scaler import (
    NUMERICAL_COLUMNS,
    create_scaler,
)


REQUIRED_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS


class PreprocessingError(ValueError):
    """Raised when preprocessing cannot be completed."""


def _validate_input(
    X: pd.DataFrame,
    *,
    validate_categories: bool = False,
) -> None:
    """Validate input features before preprocessing."""
    if not isinstance(X, pd.DataFrame):
        raise PreprocessingError(
            "Input must be a pandas DataFrame."
        )

    if X.empty:
        raise PreprocessingError(
            "Input DataFrame must not be empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in X.columns
    ]

    if missing_columns:
        raise PreprocessingError(
            f"Missing required columns: {missing_columns}"
        )

    if X[REQUIRED_COLUMNS].isnull().any().any():
        raise PreprocessingError(
            "Required feature columns must not contain missing values."
        )

    non_numeric_columns = [
        column
        for column in NUMERICAL_COLUMNS
        if not pd.api.types.is_numeric_dtype(X[column])
    ]

    if non_numeric_columns:
        raise PreprocessingError(
            f"Expected numerical columns: {non_numeric_columns}"
        )

    try:
        numerical_values = X[NUMERICAL_COLUMNS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(
            "Numerical features must contain valid numbers."
        ) from exc

    if not np.isfinite(numerical_values).all():
        raise PreprocessingError(
            "Numerical features must not contain infinite values."
        )

    if validate_categories:
        observed_categories = set(X["Type"].unique())
        invalid_categories = observed_categories - set(TYPE_CATEGORIES)

        if invalid_categories:
            raise PreprocessingError(
                f"Unexpected values in Type: {invalid_categories}"
            )


def create_preprocessing_pipeline() -> ColumnTransformer:
    """Create a transformer for categorical and numerical features."""
    return ColumnTransformer(
        transformers=[
            (
                # "type_encoder": Aplica el codificador one-hot sobre las columnas categóricas.
                "type_encoder",
                create_encoder(),
                CATEGORICAL_COLUMNS,
            ),
            (
                # "numerical_scaler": Aplica el escalador estándar sobre las columnas numéricas.
                "numerical_scaler",
                create_scaler(),
                NUMERICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessing_pipeline(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit preprocessing transformations using training data only."""
    _validate_input(X_train, validate_categories=True)

    pipeline = create_preprocessing_pipeline()

    try:
        pipeline.fit(X_train)
    except (ValueError, TypeError) as exc:
        raise PreprocessingError(
            "Failed to fit the preprocessing pipeline."
        ) from exc

    return pipeline


def transform_with_pipeline(
    X: pd.DataFrame,
    pipeline: ColumnTransformer,
) -> pd.DataFrame:
    """Transform features and preserve column names and row indices."""
    _validate_input(X)

    if not isinstance(pipeline, ColumnTransformer):
        raise PreprocessingError(
            "pipeline must be a fitted ColumnTransformer."
        )

    try:
        transformed_values = pipeline.transform(X)
        feature_names = pipeline.get_feature_names_out()
    except (ValueError, TypeError, NotFittedError, AttributeError) as exc:
        raise PreprocessingError(
            "Transformation failed. Ensure the pipeline has been fitted."
        ) from exc

    return pd.DataFrame(
        transformed_values,
        columns=feature_names,
        index=X.index,
    )
