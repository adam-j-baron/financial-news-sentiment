"""Abstract base class for dataset loaders."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from ..exceptions import DatasetError


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    This class defines the interface for loading sentiment analysis datasets.
    Each dataset loader must implement the load method and provide functionality
    for loading and preprocessing the data.

    Example:
        ```python
        class MyDatasetLoader(BaseDatasetLoader):
            def __init__(self, name: str, file_path: Path):
                super().__init__(name)
                self.file_path = file_path

            def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
                texts = ["Sample text 1", "Sample text 2"]
                labels = [1, 0]  # 0=negative, 1=neutral, 2=positive
                return texts, labels
        ```
    """

    def __init__(self, name: str) -> None:
        """Initialize the dataset loader.

        Args:
            name: Name of the dataset
        """
        self.name = name

    @abstractmethod
    def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
        """Load the dataset.

        Args:
            max_samples: Maximum number of samples to load.
                      If None, load the entire dataset.

        Returns:
            Tuple[List[str], List[int]]: Tuple containing lists of texts and their labels
                                    Labels should be integers: 0=negative, 1=neutral, 2=positive

        Raises:
            DatasetError: If there is an error loading the dataset
        """
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Get the total size of the dataset.

        Returns:
            int: Total number of samples in the dataset
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
        """Get string representation of the dataset loader.

        Returns:
            str: Dataset name
        """
        return self.name
