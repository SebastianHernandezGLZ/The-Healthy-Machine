


"""Unit tests for dataset cleaning utilities."""


import pandas as pd
import pytest


from src.data.cleaner import DataCleaningError, clean_dataset



@pytest.fixture
def sample_dataset():
    """Create a small dataset for testing."""
    return pd.DataFrame(
        {
            "temperature": [298.1, 299.0, 299.0],
            "speed": [1500, 1600, 1600],
            "Machine failure": [0, 1, 1],
        }
    )



def test_clean_dataset_returns_dataframe(sample_dataset):
    result = clean_dataset(sample_dataset)


    assert isinstance(result, pd.DataFrame)



def test_clean_dataset_does_not_modify_original(sample_dataset):
    original = sample_dataset.copy(deep=True)


    clean_dataset(sample_dataset, drop_duplicates=True)


    pd.testing.assert_frame_equal(sample_dataset, original)



def test_clean_dataset_preserves_rows_by_default(sample_dataset):
    result = clean_dataset(sample_dataset)


    assert len(result) == 3



def test_clean_dataset_removes_duplicate_rows(sample_dataset):
    result = clean_dataset(sample_dataset, drop_duplicates=True)


    assert len(result) == 2
    assert not result.duplicated().any()



def test_clean_dataset_removes_rows_with_missing_values(sample_dataset):
    sample_dataset.loc[0, "temperature"] = float("nan")


    result = clean_dataset(sample_dataset, drop_missing=True)


    assert len(result) == 2
    assert not result.isnull().any().any()



def test_clean_dataset_rejects_non_dataframe_input():
    with pytest.raises(DataCleaningError, match="Input must be a pandas DataFrame"):
        clean_dataset([1, 2, 3])



def test_clean_dataset_rejects_empty_dataframe():
    with pytest.raises(DataCleaningError, match="must not be empty"):
        clean_dataset(pd.DataFrame())




def test_clean_dataset_rejects_result_with_no_rows():
    df = pd.DataFrame({"temperature": [float("nan")]})

    with pytest.raises(DataCleaningError, match="removed all rows"):
        clean_dataset(df, drop_missing=True)
