"""Configuration for pytest."""
import pytest


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers", "integration: mark tests that require external services or resources"
    )
