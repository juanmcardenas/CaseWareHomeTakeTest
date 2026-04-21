"""
Wire adapters -> ports -> application. The only module that imports both
infrastructure and application.
"""
from fastapi import FastAPI
from config import Settings
from infrastructure.db.engine import get_engine, session_factory
from infrastructure.db.repositories import SqlReportRepository, SqlTraceRepository
from infrastructure.http.app import create_app


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = get_engine(settings.supabase_db_url)
    sm = session_factory(engine)
    report_repo = SqlReportRepository(sm)
    trace_repo = SqlTraceRepository(sm)
    return create_app(settings=settings, report_repo=report_repo, trace_repo=trace_repo)
