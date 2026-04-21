"""
In-process async event bus.

Contract:
- `publish(event)` never raises. Subscriber errors are swallowed and logged.
- Subscribers are invoked in registration order, awaited sequentially per publish
  (preserves order per subscriber).
- There is no back-pressure. Expected scale: single-digit receipts per run.
"""
import logging
from application.ports import EventBusPort, Subscriber

_log = logging.getLogger(__name__)


class InMemoryEventBus(EventBusPort):
    def __init__(self) -> None:
        self._subs: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subs.append(subscriber)

    async def publish(self, event: dict) -> None:
        for sub in self._subs:
            try:
                await sub(event)
            except Exception:
                _log.exception("event subscriber raised (ignored)")
