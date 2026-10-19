from pathlib import Path

import pandas as pd
import pytest


from src.data.loader import (
    DatasetLoadingError,
    load_dataset
)



def test_load_dataset_returns_dataframe():
    df = load_dataset()


    assert isinstance(df, pd.DataFrame)
    assert not df.empty



def test_load_dataset_has_expected_shape():
    df = load_dataset()


    assert df.shape == (10_000, 14)



def test_load_dataset_raises_error_for_missing_file():
    missing_file = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "this_file_does_not_exist.csv"
    )

    with pytest.raises(DatasetLoadingError):
        load_dataset(missing_file)