"""Implementation of scikit-learn based evaluators."""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report  # type: ignore

from .base import BaseEvaluator


class SklearnEvaluator(BaseEvaluator):
    """Evaluator using scikit-learn metrics."""

    def __init__(self, name: str = "sklearn") -> None:
        """Initialize the scikit-learn evaluator.

        Args:
            name: Name of the evaluator
        """
        super().__init__(name)
        self.label_names = ["negative", "neutral", "positive"]

    def evaluate(
        self,
        true_labels: Sequence[int],
        predicted_labels: Sequence[Optional[int]],
    ) -> Dict[str, Any]:
        """Evaluate model predictions using scikit-learn metrics.

        Args:
            true_labels: List of true labels (0=negative, 1=neutral, 2=positive)
            predicted_labels: List of predicted labels (may contain None for failed predictions)

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics:
                - accuracy: Overall accuracy score
                - invalid_count: Number of invalid (None) predictions
                - classification_report: Detailed classification metrics
                - confusion_matrix: Confusion matrix as nested lists

        Raises:
            ValidationError: If inputs are invalid
        """
        self.validate_inputs(true_labels, predicted_labels)

        # Convert lists to numpy arrays for easier handling
        y_true = np.array(true_labels)
        y_pred = np.array(
            [label if label is not None else -1 for label in predicted_labels]
        )

        # Count invalid predictions
        invalid_mask = y_pred == -1
        invalid_count = invalid_mask.sum()

        # Create mask for valid predictions
        valid_mask = ~invalid_mask
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Handle case where all predictions are invalid
        if len(y_true_valid) == 0:
            return {
                "accuracy": 0.0,
                "invalid_count": invalid_count,
                "classification_report": "No valid predictions",
                "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }

        # Calculate metrics
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        report = classification_report(
            y_true_valid,
            y_pred_valid,
            target_names=self.label_names,
            labels=[0, 1, 2],
            zero_division=0,
            output_dict=True,
        )

        return {
            "accuracy": float(accuracy),
            "invalid_count": int(invalid_count),
            "classification_report": report,
        }
