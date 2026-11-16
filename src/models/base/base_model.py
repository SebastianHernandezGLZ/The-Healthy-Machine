

"""Abstract base class for machine learning models."""

# Le indicamos a Python que esta clase no debe instanciarse directamente, sino que sirve como molde para otras clases hijas
from abc import ABC, abstractmethod  # Import ABC and abstractmethod decorator to create abstract base classes
import pandas as pd


class BaseModel(ABC):
    """Define the common interface for all machine learning models."""


    def __init__(self, model_name: str) -> None:
        """Initialize the model with its name."""
        # Validate that model_name is a valid, non-empty string
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name  # Store the identifier name of the model
        self.model = None  # Placeholder for the underlying ML estimator (e.g., scikit-learn model)
        self.is_trained = False  # Flag to track whether the model has been fitted yet


    @abstractmethod  # Esto significa que cualquier clase que herede de BaseModel está obligada a implementar estos métodos; si se te olvidara implementar alguno en una clase hija, Python lanzará un error al intentar crear una instancia de ella.
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Train the model using the provided training data."""
        raise NotImplementedError  # Enforce implementation in child classes


    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate predictions for the provided features."""
        raise NotImplementedError


    @abstractmethod
    def predict_proba(
        self, 
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate class probabilities for the provided features."""
        raise NotImplementedError


    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name


    def get_training_status(self) -> bool:
        """Return whether the model has been trained."""
        return self.is_trained