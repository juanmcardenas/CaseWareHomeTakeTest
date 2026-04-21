import os
import pytest
from config import Settings, LLMMode


def test_settings_defaults_with_mock_mode(monkeypatch):
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://u:p@h/db")
    s = Settings()
    assert s.llm_mode == LLMMode.MOCK
    assert s.ocr_timeout_s == 30
    assert s.max_file_size_mb == 10
    assert s.allowed_extensions == {"jpg", "jpeg", "png", "webp"}


def test_settings_real_mode_requires_keys(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://u:p@h/db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings()


def test_allowed_extensions_parses_csv(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "jpg,png")
    s = Settings()
    assert s.allowed_extensions == {"jpg", "png"}
