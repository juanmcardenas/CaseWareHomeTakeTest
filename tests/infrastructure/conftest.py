import pytest
from config import Settings
from fastapi.testclient import TestClient
from infrastructure.http.app import create_app
from tests.fakes.in_memory_repos import InMemoryReportRepository, InMemoryTraceRepository
from tests.fakes.fake_chat_model import FakeChatModelAdapter, default_mock_script


def _build_app(monkeypatch, max_receipts: int = 25):
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://test:test@localhost/test")
    monkeypatch.setenv("ASSETS_DIR", "./tests/fixtures/folder")
    s = Settings()
    rr = InMemoryReportRepository()
    tr = InMemoryTraceRepository()
    chat_model_port = FakeChatModelAdapter(default_mock_script(max_receipts=max_receipts))
    return create_app(settings=s, report_repo=rr, trace_repo=tr, chat_model_port=chat_model_port), rr, tr


@pytest.fixture
def app(monkeypatch):
    return _build_app(monkeypatch, max_receipts=25)


@pytest.fixture
def app1(monkeypatch):
    """App pre-loaded with a 1-receipt mock script."""
    return _build_app(monkeypatch, max_receipts=1)


@pytest.fixture
def app2(monkeypatch):
    """App pre-loaded with a 2-receipt mock script."""
    return _build_app(monkeypatch, max_receipts=2)


@pytest.fixture
def client(app):
    return TestClient(app[0]), app[1], app[2]


@pytest.fixture
def client1(app1):
    return TestClient(app1[0]), app1[1], app1[2]


@pytest.fixture
def client2(app2):
    return TestClient(app2[0]), app2[1], app2[2]
