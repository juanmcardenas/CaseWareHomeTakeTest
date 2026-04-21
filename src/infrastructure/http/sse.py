"""Thin SSE helper. Emits events from an async iterator of dicts."""
import json
from typing import AsyncIterator
from sse_starlette.sse import EventSourceResponse


def sse_response(source: AsyncIterator[dict]) -> EventSourceResponse:
    async def gen():
        async for event in source:
            yield {"event": event["event_type"], "data": json.dumps(event, default=str)}
    return EventSourceResponse(gen())
