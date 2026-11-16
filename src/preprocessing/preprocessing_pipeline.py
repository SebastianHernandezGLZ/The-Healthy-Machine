
"""Preprocessing pipeline for categorical and numerical features."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer  # Import ColumnTransformer to apply different transformers to columns
from sklearn.exceptions import NotFittedError  # Exception raised if a scikit-learn model/pipeline isn't fitted

from src.preprocessing.encoder import (
    CATEGORICAL_COLUMNS,
    TYPE_CATEGORIES,
    create_encoder,
)

from src.preprocessing.scaler import (
    create_scaler,
    NUMERICAL_COLUMNS,
)

from src.preprocessing.feature_engineering import (
    ENGINEERED_COLUMNS,
    engineer_features,
)


# Original input columns.
INPUT_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

# Original numerical columns plus engineered features.
PIPELINE_NUMERICAL_COLUMNS = (
    NUMERICAL_COLUMNS + ENGINEERED_COLUMNS
)


class PreprocessingError(ValueError):
    """Raised when preprocessing cannot be completed."""


def _validate_input(
    X: pd.DataFrame,
    *,
    validate_categories: bool = False,
) -> None:
    """Validate original input features before preprocessing."""

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
        for column in INPUT_COLUMNS
        if column not in X.columns
    ]

    if missing_columns:
        raise PreprocessingError(
            f"Missing required columns: {missing_columns}"
        )

    # Ensure no null/missing values exist in the required input feature columns
    if X[INPUT_COLUMNS].isnull().any().any():
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

    # Try casting numerical features to a float numpy array to detect conversion errors
    try:
        numerical_values = X[NUMERICAL_COLUMNS].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(
            "Numerical features must contain valid numbers."
        ) from exc

    if not np.isfinite(numerical_values).all():
        raise PreprocessingError(
            "Numerical features must not contain infinite values."
        )

    # Optionally validate categorical columns against expected domain categories
    if validate_categories:
        observed_categories = set(X["Type"].unique())

        invalid_categories = (
            observed_categories - set(TYPE_CATEGORIES)
        )

        if invalid_categories:
            raise PreprocessingError(
                f"Unexpected values in Type: {invalid_categories}"
            )


def create_preprocessing_pipeline() -> ColumnTransformer:
    """Create a transformer for categorical and numerical features."""

    # Return a ColumnTransformer combining the encoder for categories and scaler for numerical/engineered features
    return ColumnTransformer(
        transformers=[
            (
                "type_encoder",
                create_encoder(),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numerical_scaler",
                create_scaler(),
                PIPELINE_NUMERICAL_COLUMNS,
            ),
        ],
        remainder="drop",  # Drop any columns not explicitly specified in transformers
        verbose_feature_names_out=False,  # Keep clean feature names without transformer prefix
    )


def fit_preprocessing_pipeline(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit preprocessing transformations using training data only."""

    # Validate training inputs and enforce strict category validation
    _validate_input(
        X_train,
        validate_categories=True,
    )

    # Generate additional features from the original training data.
    engineered_X_train = engineer_features(X_train)

    pipeline = create_preprocessing_pipeline()

    # Fit the pipeline transformer on the expanded training dataset
    try:
        pipeline.fit(engineered_X_train)

    except (ValueError, TypeError) as exc:
        raise PreprocessingError(
            "Failed to fit the preprocessing pipeline."
        ) from exc

    return pipeline


def transform_with_pipeline(
    X: pd.DataFrame,
    pipeline: ColumnTransformer,
) -> pd.DataFrame:
    """Transform features while preserving column names and indices."""

    # Validate input features
    _validate_input(X)

    if not isinstance(pipeline, ColumnTransformer):
        raise PreprocessingError(
            "pipeline must be a fitted ColumnTransformer."
        )

    # Generate the same additional features used during training.
    engineered_X = engineer_features(X)

    # Transform input data and retrieve output feature names
    try:
        transformed_values = pipeline.transform(engineered_X)

        feature_names = pipeline.get_feature_names_out()

    except (
        ValueError,
        TypeError,
        NotFittedError,
        AttributeError,
    ) as exc:
        raise PreprocessingError(
            "Transformation failed. Ensure the pipeline has been fitted."
        ) from exc

    # Return the processed values wrapped in a pandas DataFrame preserving index and column names
    return pd.DataFrame(
        transformed_values,
        columns=feature_names,
        index=X.index,
    )
