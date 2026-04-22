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


def test_multipart_happy_path(client2):
    """Upload 2 receipt images; expect run_started, 2 receipt_result, final_result."""
    c, report_repo, trace_repo = client2
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


def test_folder_happy_path(client1):
    """Point at a folder with 1 image; expect run_started, 1 receipt_result, final_result."""
    c, *_ = client1
    events = _collect_stream(
        c,
        json={"folder_path": "./tests/fixtures/folder", "prompt": None},
    )
    kinds = [e[0] for e in events]
    assert kinds[0] == "run_started"
    assert "final_result" in kinds
    assert kinds.count("receipt_result") == 1


def test_traces_written_through(client1):
    """Traces for run_started and final_result are written to the trace repo."""
    c, report_repo, trace_repo = client1
    _collect_stream(
        c,
        files=[("files", ("r1.png",
                          Path("tests/fixtures/receipts/fixture_001.png").read_bytes(),
                          "image/png"))],
    )
    event_types = [row["event_type"] for row in trace_repo.rows]
    assert "run_started" in event_types
    assert "final_result" in event_types


def test_image_paths_happy_path(client1):
    """POST with a single image_paths entry; expect run_started, receipt_result, final_result."""
    c, *_ = client1
    events = _collect_stream(
        c,
        json={
            "image_paths": ["./tests/fixtures/folder/fixture_a.png"],
            "prompt": None,
        },
    )
    kinds = [e[0] for e in events]
    assert kinds[0] == "run_started"
    assert "final_result" in kinds
    assert kinds.count("receipt_result") == 1


def test_image_paths_bad_path_returns_400(client1):
    """A path that doesn't exist (still under ASSETS_DIR) returns HTTP 400 before streaming."""
    c, *_ = client1
    r = c.post(
        "/runs/stream",
        json={"image_paths": ["./tests/fixtures/folder/does_not_exist.png"]},
    )
    assert r.status_code == 400
    assert "does_not_exist.png" in r.text
    assert "does not exist" in r.text


def test_image_paths_and_folder_path_both_returns_422(client1):
    """Body with both fields is rejected with 422."""
    c, *_ = client1
    r = c.post(
        "/runs/stream",
        json={
            "folder_path": "./tests/fixtures/folder",
            "image_paths": ["./tests/fixtures/folder/fixture_a.png"],
        },
    )
    assert r.status_code == 422
    assert "exactly one" in r.text


def test_image_paths_neither_folder_nor_paths_returns_422(client1):
    """Body with neither field is rejected with 422."""
    c, *_ = client1
    r = c.post("/runs/stream", json={"prompt": "just a prompt"})
    assert r.status_code == 422
