import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from composition_root import build_app


pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_real_run_smoke():
    missing = [k for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "SUPABASE_DB_URL") if not os.environ.get(k)]
    if missing:
        pytest.skip(f"missing env vars: {missing}")
    os.environ["LLM_MODE"] = "real"
    app = build_app()
    client = TestClient(app)
    resp = client.post(
        "/runs/stream",
        files=[("files", ("sample.png", Path("assets/receipt_001.png").read_bytes(), "image/png"))],
        data={"prompt": "be conservative"},
    )
    assert resp.status_code == 200
    assert "final_result" in resp.text
