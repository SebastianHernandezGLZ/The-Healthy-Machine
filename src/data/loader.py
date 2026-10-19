"""Dataset loading utilities."""

from pathlib import Path 

import pandas as pd 



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
)



class DatasetLoadingError(Exception):
    """Raised when a dataset cannot be loaded."""



def load_dataset(
    file_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """Load a CSV dataset without modifying its contents.


    Args:
        file_path: Path to the CSV file.

    
    Returns:
        A DataFrame containing the loaded observations.

    
    Raises:
        DatasetLoadingError: If the file does not exist or
            cannot be read as a CSV file.
    """

    path = Path(file_path)

    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.is_file():
        raise DatasetLoadingError(
            f"Dataset file not found: {path}"
        )
    
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DatasetLoadingError(
            f"Failed to load dataset from: {path}"
        ) from exc 