"""Abstract base class for live data sources."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..exceptions import LiveDataError


class BaseLiveDataSource(ABC):
    """Abstract base class for live data sources.

    This class defines the interface for retrieving live data for sentiment analysis.
    Each live data source must implement the fetch_data method and provide
    appropriate data preprocessing.

    Example:
        ```python
        class NewsAPISource(BaseLiveDataSource):
            def __init__(self, name: str, api_key: str):
                super().__init__(name)
                self.api_key = api_key

            async def fetch_data(self, query: str, max_items: int = 10) -> List[Dict[str, Any]]:
                # Fetch news articles from NewsAPI
                articles = [
                    {
                        "id": "123",
                        "title": "Sample news",
                        "content": "Article content",
                        "url": "https://example.com",
                        "timestamp": "2025-08-23T12:00:00Z"
                    }
                ]
                return articles
        ```
    """

    def __init__(self, name: str) -> None:
        """Initialize the live data source.

        Args:
            name: Name of the data source
        """
        self.name = name

    @abstractmethod
    async def fetch_data(
        self,
        query: str,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch live data asynchronously.

        Args:
            query: Search query or identifier (e.g., stock ticker)
            max_items: Maximum number of items to fetch

        Returns:
            List[Dict[str, Any]]: List of data items with required fields:
                - id: Unique identifier
                - title: Title or headline
                - content: Main content
                - url: Source URL
                - timestamp: Publication timestamp

        Raises:
            LiveDataError: If there is an error fetching data
        """
        pass

    def __str__(self) -> str:
        """Get string representation of the data source.

        Returns:
            str: Data source name
        """
        return self.name
