import json
from pathlib import Path


def _parse_sse(text: str):
    """Parse SSE body into a list of (event, data) tuples."""
    events = []
    current_event = None
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data = line[len("data:"):].strip()
            events.append((current_event, json.loads(data)))
            current_event = None
    return events


def _collect_stream(client, **kwargs):
    with client.stream("POST", "/runs/stream", **kwargs) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    return _parse_sse(body)


def test_multipart_happy_path(client):
    c, report_repo, trace_repo = client
    p1 = Path("tests/fixtures/receipts/fixture_001.png").read_bytes()
    p2 = Path("tests/fixtures/receipts/fixture_002.png").read_bytes()
    events = _collect_stream(
        c,
        files=[
            ("files", ("r1.png", p1, "image/png")),
            ("files", ("r2.png", p2, "image/png")),
        ],
        data={"prompt": "be conservative"},
    )
    kinds = [e[0] for e in events]
    assert kinds[0] == "run_started"
    assert "final_result" in kinds
    assert kinds.count("receipt_result") == 2


def test_folder_happy_path(client):
    c, *_ = client
    events = _collect_stream(
        c,
        json={"folder_path": "./tests/fixtures/folder", "prompt": None},
    )
    kinds = [e[0] for e in events]
    assert kinds[0] == "run_started"
    assert "final_result" in kinds
    assert kinds.count("receipt_result") == 1


def test_traces_written_through(client):
    c, report_repo, trace_repo = client
    _collect_stream(
        c,
        files=[("files", ("r1.png",
                          Path("tests/fixtures/receipts/fixture_001.png").read_bytes(),
                          "image/png"))],
    )
    event_types = [row["event_type"] for row in trace_repo.rows]
    assert "run_started" in event_types
    assert "final_result" in event_types
