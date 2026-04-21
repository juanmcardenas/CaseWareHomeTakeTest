"""
Wire adapters -> ports -> application. The only module that imports both
infrastructure and application.
"""
import sys
import pathlib

from fastapi import FastAPI
from config import Settings, LLMMode
from application.ports import ChatModelPort
from infrastructure.db.engine import get_engine, session_factory
from infrastructure.db.repositories import SqlReportRepository, SqlTraceRepository
from infrastructure.http.app import create_app
from infrastructure.llm.deepseek_chat_model import DeepSeekChatModelAdapter


def build_chat_model_port(settings: Settings) -> ChatModelPort:
    if settings.llm_mode == LLMMode.REAL:
        return DeepSeekChatModelAdapter(
            api_key=settings.deepseek_api_key or "",
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            timeout_s=settings.llm_timeout_s,
        )
    # Mock mode: import fake from tests/fakes.
    # repo_root is one level above src/ so that `tests.fakes.fake_chat_model` is importable.
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tests.fakes.fake_chat_model import FakeChatModelAdapter, default_mock_script  # type: ignore
    return FakeChatModelAdapter(default_mock_script(max_receipts=settings.max_files_per_run))


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = get_engine(settings.supabase_db_url)
    sm = session_factory(engine)
    report_repo = SqlReportRepository(sm)
    trace_repo = SqlTraceRepository(sm)
    chat_model_port = build_chat_model_port(settings)
    return create_app(
        settings=settings,
        report_repo=report_repo,
        trace_repo=trace_repo,
        chat_model_port=chat_model_port,
    )
