import pytest
from config import Settings
from fastapi.testclient import TestClient
from infrastructure.http.app import create_app
from tests.fakes.in_memory_repos import InMemoryReportRepository, InMemoryTraceRepository


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://test:test@localhost/test")
    monkeypatch.setenv("ASSETS_DIR", "./tests/fixtures/folder")
    s = Settings()
    rr = InMemoryReportRepository()
    tr = InMemoryTraceRepository()
    return create_app(settings=s, report_repo=rr, trace_repo=tr), rr, tr


@pytest.fixture
def client(app):
    return TestClient(app[0]), app[1], app[2]
