"""Implementation of the Ollama-based sentiment analysis model."""
import asyncio
from typing import Any, List, Optional

import ollama

from ..exceptions import ModelError
from .base import BaseSentimentModel


class OllamaModel(BaseSentimentModel):
    """Ollama-based model for sentiment analysis."""

    def __init__(
        self,
        name: str = "qwen3",
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama model.

        Args:
            name: Name of the model (e.g., 'qwen3')
            base_url: Base URL for the Ollama API (default: http://localhost:11434)
            system_prompt: System prompt for sentiment analysis
            timeout: API call timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__(name, **kwargs)
        self.base_url = base_url or "http://localhost:11434"
        self.model_name = name
        self.system_prompt = system_prompt or (
            "You are a financial sentiment analyzer. "
            "Read the text and determine if it expresses a positive (2), "
            "neutral (1), or negative (0) sentiment. "
            "Respond with ONLY the number representing the sentiment."
        )
        self.llm = ollama.AsyncClient(host=self.base_url)

    async def initialize(self) -> None:
        """Initialize the connection to the Ollama service.

        Raises:
            ModelError: If there is an error connecting to Ollama
        """
        if self._is_initialized:
            return

        try:
            # Test the model by making a simple request
            await self.llm.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test connection."}],
                think=False,  # faster responses and only the sentiment in response, no <think>
            )
            self._is_initialized = True
        except Exception as e:
            raise ModelError(
                "Failed to initialize Ollama client", component="ollama", details=str(e)
            )

    async def predict(self, text: str) -> int:
        """Predict sentiment for a given text.

        Args:
            text: Input text to analyze

        Returns:
            int: Predicted sentiment label (0=negative, 1=neutral, 2=positive)

        Raises:
            ModelError: If there is an error connecting to the model
            ModelError: If the model has not been initialized
            ModelError: If prediction fails or returns invalid sentiment
        """
        if not self._is_initialized:
            raise ModelError(
                "Model not initialized. Call initialize() first.", component="ollama"
            )

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ]
            response = await self.llm.chat(
                model=self.model_name,
                messages=messages,
                think=False,  # faster responses and only the sentiment in response, no <think>
            )
            if not (
                response
                and hasattr(response, "message")
                and hasattr(response.message, "content")
                and response.message
                and response.message.content
            ):
                raise ModelError(
                    "Invalid response from Ollama API",
                    component="ollama",
                    details="Response missing required fields",
                )

            cleaned = response.message.content.strip()

            # Try to parse the response as a sentiment label
            try:
                sentiment = int(cleaned)
                if sentiment not in {0, 1, 2}:
                    raise ModelError(
                        "Invalid sentiment value from model",
                        component="ollama",
                        details=f"Got {sentiment}, expected 0, 1, or 2",
                    )
                return sentiment
            except (ValueError, TypeError):
                raise ModelError(
                    "Invalid sentiment format from model",
                    component="ollama",
                    details=f"Could not parse {cleaned} as integer",
                )

        except Exception as e:
            raise ModelError(
                "Failed to predict sentiment", component="ollama", details=str(e)
            )

    async def predict_batch(self, texts: List[str]) -> List[Optional[int]]:
        """Predict sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List[Optional[int]]: List of predicted sentiment labels
                              (0=negative, 1=neutral, 2=positive, None for failed predictions)

        Raises:
            ModelError: If the model has not been initialized
        """
        if not self._is_initialized:
            raise ModelError(
                "Model not initialized. Call initialize() first.", component="ollama"
            )

        # Use a semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def predict_with_semaphore(text: str) -> Optional[int]:
            async with semaphore:
                return await self.predict(text)

        try:
            # Predict all texts concurrently with rate limiting
            tasks = [predict_with_semaphore(text) for text in texts]
            return await asyncio.gather(*tasks)
        except Exception as e:
            raise ModelError(
                "Failed to generate batch predictions",
                component="ollama",
                details=str(e),
            )
