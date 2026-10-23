


"""Unit tests for dataset splitting utilities."""


import pandas as pd
import pytest


from src.data.loader import load_dataset
from src.data.splitter import DataSplittingError, split_dataset



@pytest.fixture
def dataset():
    """Load the project dataset."""
    return load_dataset()



def test_split_dataset_returns_expected_shapes(dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)


    assert X_train.shape == (8_000, 6)
    assert X_test.shape == (2_000, 6)
    assert len(y_train) == 8_000
    assert len(y_test) == 2_000



def test_split_dataset_is_reproducible(dataset):
    first_split = split_dataset(dataset, random_state=42)
    second_split = split_dataset(dataset, random_state=42)

    for first, second in zip(first_split, second_split):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)



def test_split_dataset_preserves_target_proportions(dataset):
    _, _, y_train, y_test = split_dataset(dataset)


    full_failure_rate = dataset["Machine failure"].mean()


    assert abs(y_train.mean() - full_failure_rate) < 0.005
    assert abs(y_test.mean() - full_failure_rate) < 0.005



def test_split_dataset_excludes_identifiers_and_failure_types(dataset):
    X_train, X_test, _, _ = split_dataset(dataset)


    excluded_columns = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]


    for column in excluded_columns:
        assert column not in X_train.columns
        assert column not in X_test.columns



def test_split_dataset_rejects_invalid_test_size(dataset):
    with pytest.raises(DataSplittingError, match="test_size must be between 0 and 1"):
        split_dataset(dataset, test_size=1.0)



def test_split_dataset_returns_expected_types(dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)


    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)