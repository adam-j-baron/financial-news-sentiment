"""Abstract base class for model evaluators."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from ..exceptions import ValidationError


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators.

    This class defines the interface for evaluating sentiment analysis models.
    Each evaluator must implement the evaluate method and provide specific
    evaluation metrics.

    Example:
        ```python
        class AccuracyEvaluator(BaseEvaluator):
            def __init__(self, name: str = "accuracy"):
                super().__init__(name)

            def evaluate(
                self,
                true_labels: List[int],
                predicted_labels: List[int],
            ) -> Dict[str, Any]:
                accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
                return {"accuracy": accuracy}
        ```
    """

    def __init__(self, name: str) -> None:
        """Initialize the evaluator.

        Args:
            name: Name of the evaluator
        """
        self.name = name

    def validate_inputs(
        self,
        true_labels: Sequence[int],
        predicted_labels: Sequence[Optional[int]],
    ) -> None:
        """Validate evaluation inputs.

        Args:
            true_labels: List of true labels
            predicted_labels: List of predicted labels

        Raises:
            ValidationError: If inputs are invalid
        """
        if not true_labels:
            raise ValidationError(
                "True labels list is empty",
                component=self.name,
            )

        if len(true_labels) != len(predicted_labels):
            raise ValidationError(
                "Number of true and predicted labels must match",
                component=self.name,
                details={
                    "true_labels_length": len(true_labels),
                    "predicted_labels_length": len(predicted_labels),
                },
            )

        # Check if labels are within valid range (0, 1, 2)
        valid_labels = {0, 1, 2}
        invalid_true = [i for i, l in enumerate(true_labels) if l not in valid_labels]
        invalid_pred = [
            i
            for i, l in enumerate(predicted_labels)
            if l is not None and l not in valid_labels
        ]

        if invalid_true:
            raise ValidationError(
                "Invalid values in true labels",
                component=self.name,
                details={"invalid_indices": invalid_true},
            )

        if invalid_pred:
            raise ValidationError(
                "Invalid values in predicted labels",
                component=self.name,
                details={"invalid_indices": invalid_pred},
            )

    @abstractmethod
    def evaluate(
        self,
        true_labels: List[int],
        predicted_labels: List[Optional[int]],
    ) -> Dict[str, Any]:
        """Evaluate model predictions.

        Args:
            true_labels: List of true labels (0=negative, 1=neutral, 2=positive)
            predicted_labels: List of predicted labels (may contain None for failed predictions)

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics

        Raises:
            ValidationError: If inputs are invalid
        """
        pass

    def __str__(self) -> str:
        """Get string representation of the evaluator.

        Returns:
            str: Evaluator name
        """
        return self.name
