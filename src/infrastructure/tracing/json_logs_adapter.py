import json
import logging
import time
from application.ports import TracerPort, TracerSpan

_log = logging.getLogger("trace")


class JSONLogsTracer(TracerPort):
    def start_span(self, name: str, input: dict | None = None) -> "JSONLogsSpan":
        return JSONLogsSpan(name, input)


class JSONLogsSpan(TracerSpan):
    def __init__(self, name: str, input: dict | None) -> None:
        self._name = name
        self._started = time.perf_counter()
        _log.info(json.dumps({"event": "span_start", "name": name, "input": input or {}}))

    def end(self, output: dict | None = None, error: str | None = None) -> None:
        duration_ms = int((time.perf_counter() - self._started) * 1000)
        _log.info(json.dumps({
            "event": "span_end",
            "name": self._name,
            "duration_ms": duration_ms,
            "output": output or {},
            "error": error,
        }))
