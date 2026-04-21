"""
Async repositories. Each write is its own short transaction.
`TraceRepository.insert_trace` is tolerant of schema/constraint errors — it logs
and swallows, never raising (per the EventBus contract).
"""
import logging
from uuid import UUID
from sqlalchemy.ext.asyncio import async_sessionmaker
from infrastructure.db.models import ReportRow, ReceiptRow, TraceRow
from application.ports import ReportRepositoryPort, TraceRepositoryPort

_log = logging.getLogger(__name__)


class SqlReportRepository(ReportRepositoryPort):
    def __init__(self, session_maker: async_sessionmaker) -> None:
        self._session = session_maker

    async def insert_report(self, row: dict) -> None:
        async with self._session() as s, s.begin():
            s.add(ReportRow(**row))

    async def update_report(self, report_id: UUID, patch: dict) -> None:
        async with self._session() as s, s.begin():
            row = await s.get(ReportRow, report_id)
            if row is None:
                _log.warning("update_report: no row for id=%s", report_id)
                return
            for k, v in patch.items():
                setattr(row, k, v)

    async def insert_receipt(self, row: dict) -> None:
        async with self._session() as s, s.begin():
            s.add(ReceiptRow(**row))


class SqlTraceRepository(TraceRepositoryPort):
    def __init__(self, session_maker: async_sessionmaker) -> None:
        self._session = session_maker

    async def insert_trace(self, row: dict) -> None:
        try:
            async with self._session() as s, s.begin():
                s.add(TraceRow(**row))
        except Exception:
            _log.exception("insert_trace failed (ignored)")
