


"""Dataset splitting utilities."""


import pandas as pd
from sklearn.model_selection import train_test_split


from src.data.validator import validate_dataset



TARGET_COLUMN = "Machine failure"


FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]



class DataSplittingError(ValueError):
    """Raised when a dataset cannot be split."""



def split_dataset(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,   # Para que esa "aleatoriedad" sea exactamente igual cada vez que ejecutas tu código, le das un punto de partida o semilla.
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataset into training and testing sets.
    
    
    Features and target are separated before splitting. Stratified sampling
    preserves the target class proportions in both sets.
    
    
    Returns:
        A tuple containing X_train, X_test, y_train, and y_test.
    """

    validate_dataset(df)


    if not 0 < test_size < 1:
        raise DataSplittingError("test_size must be between 0 and 1.")


    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()


    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,   # Esto asegura que tanto en el conjunto de entrenamiento como en el de prueba exista la misma proporción de casos de éxito y de fallo
        )
    except ValueError as exc:
        raise DataSplittingError(
            "Unable to split the dataset. Check the target class distribution "
            "and the requested test_size."
        ) from exc


    return X_train, X_test, y_train, y_test