from abc import ABC, abstractmethod
from typing import Any


class BasePredictor(ABC):
    """
    Abstract base class for all predictors (e.g., battery, thermal, behavior).
    Defines a common interface for training, predicting, saving, and loading.
    """

    @abstractmethod
    def fit(self, training_data: Any, **kwargs) -> None:
        """
        Train or initialize the predictor on provided data.

        Args:
            training_data: Predictor-specific training data
                           (e.g., list of sessions, DataFrame, file paths).
            **kwargs:      Additional parameters (e.g., epochs, learning rate).
        """
        ...

    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """
        Generate predictions for new input.

        Args:
            input_data: Predictor-specific input
                        (e.g., a session list, battery time series).
            **kwargs:   Additional options (e.g., top_k for classification).

        Returns:
            Predictor-specific output
            (e.g., list of (item, probability), numeric value, etc.).
        """
        ...

    def save(self, path: str) -> None:
        """
        Save internal model state to disk.

        Args:
            path: File path (or directory) where to save the model.
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """
        Load internal model state from disk.

        Args:
            path: File path (or directory) from which to load the model.
        """
        raise NotImplementedError
