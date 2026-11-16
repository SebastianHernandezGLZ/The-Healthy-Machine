

"""Logistic Regression implementation for predictive maintenance."""


import numpy as np
import pandas as pd
    
from sklearn.linear_model import LogisticRegression  # Import scikit-learn Logistic Regression estimator
from sklearn.exceptions import NotFittedError  # Exception raised if an unfitted model is used

from src.models.base.base_model import BaseModel  # Import abstract base class



class ModelTrainingError(ValueError):
    """Raised when model training cannot be completed."""



class ModelPredictionError(ValueError):
    """Raised when model predictions cannot be generated."""



class LogisticRegressionModel(BaseModel):
    """Implement Logistic Regression using the BaseModel interface."""


    def __init__(
            self,
            max_iter: int = 1000,  # Parameter: maximum optimization iterations for convergence
            # Fijar esto (por ejemplo, en 42) asegura que cada vez que ejecutes el código obtengas exactamente los mismos resultados, evitando variaciones aleatorias
            random_state: int = 42,  # Parameter: seed for random number generator to ensure reproducibility
            # El modelo penaliza más los errores en la clase minoritaria (las fallas reales), evitando que el algoritmo ignore los fallos por puro desequilibrio de datos.
            class_weight: str | None = "balanced",  # Parameter: handles imbalanced datasets by weighting classes inversely proportional to frequencies
    ) -> None:
        """Initialize the Logistic Regression model."""
        super().__init__("Logistic Regression")  # Initialize base class with model name


        # Validate that max_iter is a positive integer
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")


        # Instantiate the underlying scikit-learn model with the provided hyperparameters
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )

     # Se utiliza para definir una función dentro de una clase que no necesita acceder ni modificar los atributos o el estado de la instancia (self) ni de la clase.
    @staticmethod
    def _validate_features(X: pd.DataFrame) -> None:
        """Validate feature data before training or prediction."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame.")


        if X.empty:
            raise ValueError("Features must not be empty.")


        if X.isnull().any().any():
            raise ValueError("Features must not contain missing values.")


        if not all(
            pd.api.types.is_numeric_dtype(X[column])
            for column in X.columns
        ):
            raise ValueError("All features must be numerical.")


        # Attempt to convert features to a numpy float array to catch casting issues
        try:
            values = X.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Features must contain valid numbers.") from exc


        if not np.isfinite(values.all()):
            raise ValueError("Features must not contain infinite values.")


    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Train Logistic Regression using the provided training data."""
        try:
            # 1. Validate the input feature matrix X_train using our validation method
            self._validate_features(X_train)


            if not isinstance(y_train, pd.Series):
                raise ValueError("Target must be a pandas Series.")


            if y_train.empty:
                raise ValueError("Target must not be empty.")


            if len(X_train) != len(y_train):
                raise ValueError(
                    "Features and target must have the same number of rows."
                )


            if not X_train.index.equals(y_train.index):
                raise ValueError(
                    "Features and target must have matching indices."
                )


            if y_train.isnull().any():
                raise ValueError("Target must not contain missing values.")


            # Ensure classification target has at least two distinct classes (e.g., 0 and 1)
            if y_train.nunique() < 2:
                raise ValueError(
                    "Target must contain at least two different classes."
                )


            # 3. Fit the underlying scikit-learn model using training data
            self.model.fit(X_train, y_train)

            # 4. Update the training status flag inherited from BaseModel
            self.is_trained = True


        except (ValueError, TypeError) as exc:
            # If training fails for any reason, ensure status remains False and wrap error
            self.is_trained = False
            raise ModelTrainingError(
                f"Failed to train {self.model_name}."
            ) from exc


    def predict(self, X:pd.DataFrame) -> pd.Series:
        """Generate class predictions."""

        if not self.is_trained:
            raise ModelPredictionError(
                "Model must be trained before making predictions."
            )


        try:
            # Validate feature matrix structure and values
            self._validate_features(X)

            # Generate discrete class predictions (e.g., 0 or 1) using scikit-learn
            predictions = self.model.predict(X)

            # Wrap raw predictions back into a pandas Series, preserving indices and assigning a name
            return pd.Series(
                predictions,
                index = X.index,
                name = "prediction",
            )


        except (ValueError, TypeError, NotFittedError) as exc:
            raise ModelPredictionError(
                f"Failed to generate predictions with {self.model_name}."
            ) from exc


    def predict_proba(self, X:pd.DataFrame) -> pd.DataFrame:
        """Generate class probabilities."""

        if not self.is_trained:
            raise ModelPredictionError(
                "Model must be trained before predicting probabilites."
            )


        try:
            self._validate_features(X)

            # Generate probability matrix from scikit-learn (e.g., [0.8, 0.2] for classes 0 and 1)
            probabilities = self.model.predict_proba(X)

            # Wrap probabilities into a clean pandas DataFrame using actual class names as columns and keeping original index
            return pd.DataFrame(
                probabilities,
                columns = self.model.classes_,
                index = X.index,
            )


        except (ValueError, TypeError, NotFittedError) as exc:
            raise ModelPredictionError(
                f"Failed to generate probabilities with {self.model_name}."
            ) from exc
        