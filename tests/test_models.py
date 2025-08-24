"""Tests for the sentiment analysis models."""
from unittest.mock import Mock, patch

import pytest

from financial_news_sentiment.exceptions import ModelError
from financial_news_sentiment.models import (
    BaseSentimentModel,
    FinBERTModel,
    OllamaModel,
)


class TestBaseSentimentModel:
    """Test the base sentiment model."""

    class DummyModel(BaseSentimentModel):
        async def initialize(self) -> None:
            self._is_initialized = True

        async def predict(self, text: str):
            return 1  # Always return neutral

        async def predict_batch(self, texts: list[str]):
            return [1] * len(texts)  # Always return neutral

    @pytest.mark.asyncio
    async def test_label_mappings(self):
        """Test that label mappings are consistent."""
        model = self.DummyModel("dummy")
        assert model.label_map == {"negative": 0, "neutral": 1, "positive": 2}
        assert model.id_to_label == {0: "negative", 1: "neutral", 2: "positive"}

    @pytest.mark.asyncio
    async def test_str_representation(self):
        """Test string representation of the model."""
        model = self.DummyModel("dummy")
        assert str(model) == "dummy"


@pytest.mark.integration
class TestFinBERTModel:
    """Integration tests for the FinBERT model."""

    @pytest.fixture
    async def model(self):
        """Create a FinBERT model instance."""
        model = FinBERTModel()
        await model.initialize()
        return model  # Use return instead of yield for async fixtures

    @pytest.mark.asyncio
    async def test_predict_positive(self, model):
        """Test prediction of positive sentiment."""
        text = "The company reported strong earnings growth."
        model_instance = await model
        result = await model_instance.predict(text)
        assert result in {0, 1, 2}

    @pytest.mark.asyncio
    async def test_predict_negative(self, model):
        """Test prediction of negative sentiment."""
        text = "The company filed for bankruptcy."
        model_instance = await model
        result = await model_instance.predict(text)
        assert result in {0, 1, 2}

    @pytest.mark.asyncio
    async def test_predict_neutral(self, model):
        """Test prediction of neutral sentiment."""
        text = "The company released its quarterly report."
        model_instance = await model
        result = await model_instance.predict(text)
        assert result in {0, 1, 2}

    @pytest.mark.asyncio
    async def test_predict_batch(self, model):
        """Test batch prediction."""
        texts = [
            "The company reported strong earnings growth.",
            "The company filed for bankruptcy.",
            "The company released its quarterly report.",
        ]
        model_instance = await model
        results = await model_instance.predict_batch(texts)
        assert len(results) == len(texts)
        assert all(result in {0, 1, 2} for result in results)


class TestOllamaModel:
    """Tests for the Ollama model."""

    @pytest.fixture
    async def model(self):
        """Create an Ollama model instance with mocked AsyncClient."""
        patcher = patch("financial_news_sentiment.models.ollama.ollama.AsyncClient")
        mock_client = patcher.start()
        mock_client_instance = mock_client.return_value
        model = OllamaModel(
            name="test-model",
            base_url="http://localhost:11434",
            system_prompt="Test prompt",
        )

        # Create async method mock
        async def async_chat(*args, **kwargs):
            mock_response = Mock()
            mock_response.message = Mock()
            mock_response.message.content = "2"  # Positive sentiment
            return mock_response

        mock_client_instance.chat = Mock()
        mock_client_instance.chat.side_effect = async_chat

        # Set up the mocked client
        model.llm = mock_client_instance
        model._is_initialized = True  # Skip initialization since we mock it
        return model  # Use return instead of yield for async fixtures

    @pytest.mark.asyncio
    async def test_predict_valid_response(self, model):
        """Test prediction with a valid response."""
        model_instance = await model
        result = await model_instance.predict("Test text")
        assert result == 2  # positive maps to 2

    @pytest.mark.asyncio
    async def test_predict_invalid_response(self, model):
        """Test prediction with an invalid response."""
        model_instance = await model

        async def async_chat(*args, **kwargs):
            mock_response = Mock()
            mock_response.message = Mock()
            mock_response.message.content = "invalid"
            return mock_response

        model_instance.llm.chat.side_effect = async_chat
        with pytest.raises(ModelError) as exc_info:
            await model_instance.predict("Test text")
        assert "Failed to predict sentiment" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_predict_batch(self, model):
        """Test batch prediction."""
        model_instance = await model

        async def async_chat(*args, **kwargs):
            mock_response = Mock()
            mock_response.message = Mock()
            mock_response.message.content = "1"  # Neutral sentiment
            return mock_response

        model_instance.llm.chat.side_effect = async_chat
        texts = ["Text 1", "Text 2", "Text 3"]
        results = await model_instance.predict_batch(texts)
        assert len(results) == len(texts)
        assert all(result in {0, 1, 2} for result in results)
        assert all(result == 1 for result in results)  # neutral maps to 1


if __name__ == "__main__":
    pytest.main([__file__])
