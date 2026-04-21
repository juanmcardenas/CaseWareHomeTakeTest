"""
Repository round-trip against a real Postgres.
Requires TEST_SUPABASE_DB_URL env var pointing to a test database.
Skipped if unreachable.
"""
import os
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from infrastructure.db.models import Base, ReportRow
from infrastructure.db.repositories import SqlReportRepository, SqlTraceRepository


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def session_maker():
    url = os.environ.get("TEST_SUPABASE_DB_URL")
    if not url:
        pytest.skip("TEST_SUPABASE_DB_URL not set")
    engine = create_async_engine(url, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_report_round_trip(session_maker):
    repo = SqlReportRepository(session_maker)
    rid = uuid4()
    await repo.insert_report({
        "id": rid, "started_at": datetime.now(timezone.utc), "status": "running",
        "prompt": None, "input_kind": "folder", "input_ref": "/tmp",
    })
    await repo.update_report(rid, {
        "finished_at": datetime.now(timezone.utc),
        "status": "succeeded", "total_spend": Decimal("0.00"),
        "by_category": {}, "issues": [],
    })
    async with session_maker() as s:
        row = await s.get(ReportRow, rid)
        assert row.status == "succeeded"
