"""
Shared pytest fixtures.

Design principle: per-layer conftests inherit from this one for layer-specific fixtures.
This root conftest only provides primitives shared across all layers.
"""
import os
import pytest


@pytest.fixture(autouse=True)
def _mock_mode_env(monkeypatch):
    """
    Default all tests to mock mode so no real API keys are needed.
    Individual tests can override by setting LLM_MODE themselves.
    """
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv(
        "SUPABASE_DB_URL",
        os.environ.get("TEST_SUPABASE_DB_URL", "postgresql+psycopg://test:test@localhost/test"),
    )
