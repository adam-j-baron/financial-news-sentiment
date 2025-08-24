"""Configuration management for the financial news sentiment analysis system."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Configuration for a machine learning model."""

    name: str
    device: str = Field(default="auto", pattern="^(cpu|cuda|auto)$")
    batch_size: PositiveInt = Field(
        default=32, description="Batch size for model inference"
    )
    cache_dir: Path = Field(default=Path(".cache/models"))

    @field_validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate the device setting."""
        if v not in ["cpu", "cuda", "auto"]:
            raise ValueError("Device must be one of: cpu, cuda, auto")
        return v


class OllamaConfig(BaseModel):
    """Configuration for the Ollama API."""

    url: HttpUrl
    model: str
    system_prompt: str
    timeout: PositiveFloat = Field(default=30.0, description="API timeout in seconds")
    max_retries: PositiveInt = Field(default=3)


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    url: HttpUrl
    file_path: Path
    encoding: str = Field(default="utf-8")
    cache_dir: Path = Field(default=Path(".cache/datasets"))
    shuffle_seed: Optional[int] = Field(default=None)


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    num_sentences: PositiveInt
    metrics: List[str]
    output_format: str = Field(default="text", pattern="^(text|json|csv)$")


class LiveDataConfig(BaseModel):
    """Configuration for live data sources."""

    timeout: PositiveFloat = Field(default=30.0)
    max_retries: PositiveInt = Field(default=3)
    default_num_articles: PositiveInt = Field(default=10)
    cache_ttl: PositiveInt = Field(default=300)


class CacheConfig(BaseModel):
    """Configuration for the caching system."""

    enabled: bool = True
    directory: Path = Field(default=Path(".cache"))
    max_size: str = Field(default="1GB")
    ttl: PositiveInt = Field(default=86400)


class APIConfig(BaseModel):
    """Configuration for the API server."""

    host: str = Field(default="127.0.0.1")
    port: PositiveInt = Field(default=8000)
    workers: PositiveInt = Field(default=4)
    timeout: PositiveInt = Field(default=60)
    cors_origins: List[str] = Field(default=["*"])


class SecurityConfig(BaseModel):
    """Security-related configuration."""

    ssl_verify: bool = True
    input_validation: bool = True
    max_content_length: str = Field(default="10MB")


class MonitoringConfig(BaseModel):
    """Configuration for system monitoring."""

    enabled: bool = True
    metrics_port: PositiveInt = Field(default=9090)
    collect_performance_metrics: bool = True
    prometheus_enabled: bool = False


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(env_prefix="FINS_", env_nested_delimiter="__")

    app_name: str = "financial-news-sentiment"
    app_version: str = "0.1.0"
    environment: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    log_format: str = Field(default="json", pattern="^(json|text)$")

    models: Dict[str, Union[ModelConfig, OllamaConfig]]
    datasets: Dict[str, DatasetConfig]
    evaluation: EvaluationConfig
    live_data: Dict[str, LiveDataConfig]
    cache: CacheConfig
    api: APIConfig
    security: SecurityConfig
    monitoring: MonitoringConfig

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AppConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            AppConfig: Loaded and validated configuration

        Raises:
            ConfigurationError: If the configuration file cannot be loaded or is invalid
        """
        try:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls.model_validate(config_dict)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {path}",
                component="config",
                details=str(e),
            )

    def get_model_config(self, model_name: str) -> Union[ModelConfig, OllamaConfig]:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Union[ModelConfig, OllamaConfig]: Model configuration

        Raises:
            ConfigurationError: If the model configuration is not found
        """
        try:
            return self.models[model_name]
        except KeyError:
            raise ConfigurationError(
                f"Configuration for model '{model_name}' not found",
                component="config",
            )

    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetConfig: Dataset configuration

        Raises:
            ConfigurationError: If the dataset configuration is not found
        """
        try:
            return self.datasets[dataset_name]
        except KeyError:
            raise ConfigurationError(
                f"Configuration for dataset '{dataset_name}' not found",
                component="config",
            )


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load application configuration.

    Args:
        config_path: Path to the configuration file. If None, uses default locations.

    Returns:
        AppConfig: Loaded and validated configuration

    Raises:
        ConfigurationError: If the configuration cannot be loaded
    """
    if config_path is None:
        # Try common configuration locations
        locations = [
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path.home() / ".config" / "financial-news-sentiment" / "config.yaml",
        ]
        for path in locations:
            if path.is_file():
                config_path = path
                break
        else:
            raise ConfigurationError(
                "No configuration file found in default locations",
                component="config",
                details={"searched_locations": [str(p) for p in locations]},
            )

    return AppConfig.from_yaml(config_path)
