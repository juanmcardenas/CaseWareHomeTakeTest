from uuid import UUID
from application.event_bus import InMemoryEventBus as _InMemoryEventBus
from application.ports import ReportRepositoryPort, TraceRepositoryPort


class InMemoryEventBus(_InMemoryEventBus):
    """Test-only subclass that records all published events in `published`."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[dict] = []

    async def publish(self, event: dict) -> None:
        self.published.append(event)
        await super().publish(event)


class InMemoryReportRepository(ReportRepositoryPort):
    def __init__(self) -> None:
        self.reports: dict[UUID, dict] = {}
        self.receipts: list[dict] = []

    async def insert_report(self, row: dict) -> None:
        self.reports[row["id"]] = dict(row)

    async def update_report(self, report_id: UUID, patch: dict) -> None:
        self.reports[report_id].update(patch)

    async def insert_receipt(self, row: dict) -> None:
        self.receipts.append(dict(row))


class InMemoryTraceRepository(TraceRepositoryPort):
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def insert_trace(self, row: dict) -> None:
        self.rows.append(dict(row))
