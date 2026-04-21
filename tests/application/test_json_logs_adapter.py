import json
import logging
import pytest
from infrastructure.tracing.json_logs_adapter import JSONLogsTracer


def test_span_logs_name_and_output_on_end(caplog):
    t = JSONLogsTracer()
    with caplog.at_level(logging.INFO, logger="trace"):
        span = t.start_span("load_images", input={"folder_path": "/tmp"})
        span.end(output={"count": 2})

    records = [r for r in caplog.records if r.name == "trace"]
    assert len(records) == 2
    start = json.loads(records[0].message)
    end = json.loads(records[1].message)
    assert start["event"] == "span_start"
    assert start["name"] == "load_images"
    assert end["event"] == "span_end"
    assert end["output"] == {"count": 2}


def test_span_logs_error(caplog):
    t = JSONLogsTracer()
    with caplog.at_level(logging.INFO, logger="trace"):
        span = t.start_span("ocr")
        span.end(error="timeout")
    end = json.loads([r for r in caplog.records if r.name == "trace"][-1].message)
    assert end["error"] == "timeout"
