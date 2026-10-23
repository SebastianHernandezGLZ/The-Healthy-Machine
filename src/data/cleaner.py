
"""Dataset cleaning utilities."""

import pandas as pd


class DataCleaningError(ValueError):
    """Raised when a dataset cannot be cleaned."""


def clean_dataset(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = False,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Clean a dataset using explicitly enabled cleaning rules.

    The original DataFrame is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataCleaningError("Input must be a pandas DataFrame.")

    if df.empty:
        raise DataCleaningError("Dataset must not be empty.")

    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if drop_missing:
        cleaned_df = cleaned_df.dropna()

    if cleaned_df.empty:
        raise DataCleaningError(
            "Cleaning removed all rows. Review the cleaning options."
        )

    return cleaned_df
