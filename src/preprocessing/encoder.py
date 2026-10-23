"""Categorical feature encoding utilities."""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder



CATEGORICAL_COLUMNS = ["Type"]
TYPE_CATEGORIES = ["H", "L", "M"]



class EncodingError(ValueError):
    """Raised when categorical features cannot be encoded."""



def _validate_input(
    df: pd.DataFrame,
    *,
    validate_categories: bool = False,
) -> None:
    """Validate a DataFrame before encoding categorical features."""
    if not isinstance(df, pd.DataFrame):
        raise EncodingError("Input must be a pandas DataFrame.")


    if df.empty:
        raise EncodingError("Input DataFrame must not be empty.")


    missing_columns = [
        column 
        for column in CATEGORICAL_COLUMNS
        if column not in df.columns
    ]


    if missing_columns:
        raise EncodingError(
            f"Missing categorical columns: {missing_columns}"
        )


    if df[CATEGORICAL_COLUMNS].isnull().any().any():
        raise EncodingError(
            "Categorical columns must not contain missing values."
        )


    if validate_categories:
        observed_categories = set(df["Type"].unique())
        invalid_categories = observed_categories - set(TYPE_CATEGORIES)


        if invalid_categories:
            raise EncodingError(
                f"Unexpected values in Type: {invalid_categories}"
            )



def create_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder with a stable output schema."""
    return OneHotEncoder(
        categories=[TYPE_CATEGORIES],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=int,
    )



def fit_encoder(X_train: pd.DataFrame) -> OneHotEncoder:
    """Fit the encoder using training data only."""
    _validate_input(X_train, validate_categories=True)


    encoder = create_encoder()
    encoder.fit(X_train[CATEGORICAL_COLUMNS])


    return encoder



def transform_with_encoder(
    X: pd.DataFrame,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """Encode categorical columns and preserve numerical features."""
    _validate_input(X)


    if not isinstance(encoder, OneHotEncoder):
        raise EncodingError(
            "Encoder must be a fitted sklearn OneHotEncoder."
        )


    try:
        encoded_values = encoder.transform(X[CATEGORICAL_COLUMNS])
        encoded_columns = encoder.get_feature_names_out(
            CATEGORICAL_COLUMNS
        )
    except (ValueError, AttributeError) as exc:
        raise EncodingError(
            "Encoding failed. Ensure the encoder has been fitted."
        ) from exc


    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoded_columns,
        index=X.index,
    )


    # Separa y copia el resto de las características (columnas numéricas) eliminando la columna categórica original.
    numerical_df = X.drop(columns=CATEGORICAL_COLUMNS).copy()


    # Une horizontalmente (axis=1) las columnas ya codificadas con las columnas numéricas preservadas, devolviendo el DataFrame final listo para modelos de Machine Learning.
    return pd.concat([encoded_df, numerical_df], axis=1)