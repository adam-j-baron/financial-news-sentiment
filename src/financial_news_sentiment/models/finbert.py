"""Implementation of the FinBERT sentiment analysis model."""
import asyncio
from typing import Any, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..exceptions import ModelError
from .base import BaseSentimentModel


class FinBERTModel(BaseSentimentModel):
    """FinBERT model for financial sentiment analysis."""

    def __init__(self, name: str = "finbert", **kwargs: Any) -> None:
        """Initialize the FinBERT model.

        Args:
            name: Name of the model
            **kwargs: Additional configuration options

        Note:
            The model is not loaded until initialize() is called.
        """
        super().__init__(name, **kwargs)
        self.device: Optional[torch.device] = None
        self.tokenizer: Any = None
        self.model: Any = None
        # FinBERT has a different label order:
        # 0 (positive) -> 2 (positive)
        # 1 (negative) -> 0 (negative)
        # 2 (neutral) -> 1 (neutral)
        self.finbert_label_map = {0: 2, 1: 0, 2: 1}

    async def initialize(self) -> None:
        """Initialize the model by loading it into memory.

        Raises:
            ModelError: If there is an error loading the model or tokenizer
        """
        if self._is_initialized:
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Run model loading in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def load_tokenizer() -> Any:
                return AutoTokenizer.from_pretrained(
                    "ProsusAI/finbert", revision="main"
                )

            def load_model() -> Any:
                return AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert", revision="main"
                )

            self.tokenizer = await loop.run_in_executor(None, load_tokenizer)
            self.model = await loop.run_in_executor(None, load_model)
            self.model.to(self.device)
            self._is_initialized = True
        except Exception as e:
            raise ModelError(
                "Failed to initialize FinBERT model",
                component="finbert",
                details=str(e),
            )

    async def predict(self, text: str) -> int:
        """Predict sentiment for a given text.

        Args:
            text: Input text to analyze

        Returns:
            int: Predicted sentiment label (0=negative, 1=neutral, 2=positive)

        Raises:
            ModelError: If there is an error during prediction
            ModelError: If the model has not been initialized
        """
        if not self._is_initialized:
            raise ModelError(
                "Model not initialized. Call initialize() first.", component="finbert"
            )

        try:
            if not self.model or not self.tokenizer or not self.device:
                raise ModelError(
                    "Model components not properly initialized", component="finbert"
                )

            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                finbert_label = int(
                    torch.argmax(probabilities, dim=1).item()
                )  # ensure int

            # Map FinBERT's label order to our standard order
            return self.finbert_label_map[finbert_label]
        except Exception as e:
            raise ModelError(
                "Failed to generate prediction", component="finbert", details=str(e)
            )

    async def predict_batch(self, texts: list[str]) -> list[Optional[int]]:
        """Predict sentiment for multiple texts simultaneously.

        Args:
            texts: List of texts to analyze

        Returns:
            list[int]: List of predicted sentiment labels
                      (0=negative, 1=neutral, 2=positive)

        Raises:
            ModelError: If there is an error during prediction
            ModelError: If the model has not been initialized
        """
        if not self._is_initialized:
            raise ModelError(
                "Model not initialized. Call initialize() first.", component="finbert"
            )

        try:
            # Batch tokenize all texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions for all texts at once
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                finbert_labels = torch.argmax(probabilities, dim=1).tolist()

            # Map all predictions to our standard label order
            return [self.finbert_label_map[label] for label in finbert_labels]
        except Exception as e:
            raise ModelError(
                "Failed to generate batch predictions",
                component="finbert",
                details=str(e),
            )
