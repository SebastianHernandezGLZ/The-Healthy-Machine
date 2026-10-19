import pandas as pd
import pytest


from src.data.validator import (
    DatasetValidationError,
    validate_dataset,
)



@pytest.fixture
def valid_dataset():
    return pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["L00001", "M00002"],
        "Type": ["L", "M"],
        "Air temperature [K]": [298.1, 298.2],
        "Process temperature [K]": [308.5, 308.6],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40.0, 35.0],
        "Tool wear [min]": [0, 10],
        "Machine failure": [0, 1],
    })



def test_validate_dataset_accepts_valid_data(valid_dataset):
    assert validate_dataset(valid_dataset) is None



def test_validate_dataset_rejects_missing_column(valid_dataset):
    df = valid_dataset.drop(columns=["Torque [Nm]"])


    with pytest.raises(
        DatasetValidationError,
        match="Missing required columns",
    ):
        validate_dataset(df)



def test_validate_dataset_rejects_missing_values(valid_dataset):
    df = valid_dataset.copy()
    df.loc[0, "Torque [Nm]"] = float("nan")


    with pytest.raises(
        DatasetValidationError,
        match="Missing values",
    ):
        validate_dataset(df)



def test_validate_dataset_rejects_invalid_target(valid_dataset):
    df = valid_dataset.copy()
    df["Machine failure"] = [0, 2]


    with pytest.raises(
        DatasetValidationError,
        match="Machine failure must contain only 0 and 1",
    ):
        validate_dataset(df)



def test_validate_dataset_rejects_invalid_product_type(valid_dataset):
    df = valid_dataset.copy()
    df["Type"] = ["L", "X"]


    with pytest.raises(
        DatasetValidationError,
        match="Unexpected product types",
    ):
        validate_dataset(df)



def test_validate_dataset_rejects_empty_dataframe():
    with pytest.raises(
        DatasetValidationError,
        match="must not be empty"
    ):
        validate_dataset(pd.DataFrame())