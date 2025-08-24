"""Abstract base classes for sentiment analysis models."""
import abc
from typing import Any, Dict, Optional, Type

from ..exceptions import ModelError


class BaseSentimentModel(abc.ABC):
    """Base class for all sentiment analysis models.

    All sentiment analysis models must inherit from this class and implement
    the required methods.

    Example:
        class MyModel(BaseSentimentModel):
            async def predict(self, text: str) -> int:
                # Implement prediction logic
                return 1  # positive sentiment
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize the model.

        Args:
            name: Name of the model.
            **kwargs: Model-specific configuration options.
        """
        self.name = name
        self.config = kwargs
        self._is_initialized = False

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the model (load weights, start services, etc.).

        This method should be called before making any predictions.
        It should handle downloading model files, setting up connections, etc.

        Raises:
            ModelError: If initialization fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def predict(self, text: str) -> Optional[int]:
        """Predict sentiment for the given text.

        Args:
            text: The text to analyze.

        Returns:
            Optional[int]: The predicted sentiment class:
                - 0 for negative
                - 1 for neutral
                - 2 for positive
                - None if prediction fails

        Raises:
            ModelError: If there are errors communicating with the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def predict_batch(self, texts: list[str]) -> list[Optional[int]]:
        """Predict sentiment for multiple texts at once.

        Args:
            texts: List of texts to analyze.

        Returns:
            List[Optional[int]]: List of predicted sentiment classes (0, 1, 2, or None).
                               None indicates a failed prediction for that text.

        Raises:
            ModelError: If there are errors communicating with the model.
        """
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Clean up any resources used by the model.

        This method should be called when the model is no longer needed.
        Default implementation does nothing.
        """
        pass

    @property
    def label_map(self) -> dict[str, int]:
        """Get the mapping from text labels to numeric labels.

        Returns:
            dict[str, int]: Mapping from text labels to numeric labels
        """
        return {"negative": 0, "neutral": 1, "positive": 2}

    @property
    def id_to_label(self) -> dict[int, str]:
        """Get the mapping from numeric labels to text labels.

        Returns:
            dict[int, str]: Mapping from numeric labels to text labels
        """
        return {v: k for k, v in self.label_map.items()}

    def __str__(self) -> str:
        """Get string representation of the model.

        Returns:
            str: Model name
        """
        return self.name
