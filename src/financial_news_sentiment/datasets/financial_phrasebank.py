"""Loader for the Financial PhraseBank dataset."""
import io
import random
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from ..exceptions import DatasetError
from .base import BaseDatasetLoader


class FinancialPhraseBankLoader(BaseDatasetLoader):
    """Loader for the Financial PhraseBank dataset."""

    def __init__(
        self,
        name: str = "FinancialPhraseBank",
        url: str = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
        file_path: str = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
        encoding: str = "ISO-8859-1",
        shuffle_seed: Optional[int] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the Financial PhraseBank dataset loader.

        Args:
            name: Name of the dataset
            url: URL to download the dataset from
            file_path: Path to the file within the ZIP archive
            encoding: Character encoding of the file
            shuffle_seed: Random seed for shuffling the dataset
            cache_dir: Directory to cache downloaded files

        Raises:
            DatasetError: If there is an error initializing the loader
        """
        super().__init__(name)
        self.url = url
        self.file_path = file_path
        self.encoding = encoding
        self.shuffle_seed = shuffle_seed
        self.cache_dir = cache_dir or Path(".cache/datasets")
        self._size: Optional[int] = None
        self._data: Optional[Tuple[List[str], List[int]]] = None

    def _download_dataset(self) -> bytes:
        """Download the dataset from the URL.

        Returns:
            bytes: Raw ZIP file content

        Raises:
            DatasetError: If there is an error downloading the dataset
        """
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / "financial_phrasebank.zip"

            # Check if cached file exists
            if cache_path.exists():
                return cache_path.read_bytes()

            # Download file if not cached
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(self.url)
                response.raise_for_status()

            # Cache the downloaded file
            cache_path.write_bytes(response.content)
            return response.content
        except Exception as e:
            raise DatasetError(
                "Failed to download Financial PhraseBank dataset",
                component="financial_phrasebank",
                details=str(e),
            )

    def _parse_file(self, content: str) -> Tuple[List[str], List[int]]:
        """Parse the dataset file content.

        Args:
            content: Raw file content as string

        Returns:
            Tuple[List[str], List[int]]: Tuple of texts and labels

        Raises:
            DatasetError: If there is an error parsing the file
        """
        try:
            data_tuples = []
            lines = content.splitlines()

            for line in lines:
                if "@" in line:
                    sentence, label_str = line.rsplit("@", 1)
                    sentence = sentence.strip()
                    label_str = label_str.strip()
                    if label_str in self.label_map:
                        data_tuples.append((sentence, self.label_map[label_str]))

            if not data_tuples:
                raise DatasetError(
                    "No valid data found in Financial PhraseBank dataset",
                    component="financial_phrasebank",
                )

            # Shuffle data if seed is provided
            if self.shuffle_seed is not None:
                random.seed(self.shuffle_seed)
                random.shuffle(data_tuples)

            # Split into texts and labels
            texts = [t[0] for t in data_tuples]
            labels = [t[1] for t in data_tuples]

            return texts, labels
        except Exception as e:
            raise DatasetError(
                "Failed to parse Financial PhraseBank dataset",
                component="financial_phrasebank",
                details=str(e),
            )

    def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
        """Load the Financial PhraseBank dataset.

        Args:
            max_samples: Maximum number of samples to load

        Returns:
            Tuple[List[str], List[int]]: Tuple containing lists of texts and their labels

        Raises:
            DatasetError: If there is an error loading the dataset
        """
        try:
            # Return cached data if available
            if self._data is not None:
                texts, labels = self._data
                if max_samples:
                    return texts[:max_samples], labels[:max_samples]
                return texts, labels

            # Download and extract the dataset
            zip_content = self._download_dataset()
            zip_file = zipfile.ZipFile(io.BytesIO(zip_content))

            # Read and parse the file
            with zip_file.open(self.file_path) as f:
                content = f.read().decode(self.encoding)

            # Parse the content
            texts, labels = self._parse_file(content)

            # Cache the parsed data
            self._data = (texts, labels)
            self._size = len(texts)

            # Return requested number of samples
            if max_samples:
                return texts[:max_samples], labels[:max_samples]
            return texts, labels
        except Exception as e:
            raise DatasetError(
                "Failed to load Financial PhraseBank dataset",
                component="financial_phrasebank",
                details=str(e),
            )

    @property
    def size(self) -> int:
        """Get the total size of the dataset.

        Returns:
            int: Total number of samples in the dataset

        Raises:
            DatasetError: If the dataset hasn't been loaded yet
        """
        if self._size is None:
            self.load()  # This will set self._size
        if self._size is None:
            raise DatasetError(
                "Dataset size is not set after loading",
                component="financial_phrasebank",
            )
        return self._size
