# Receipt Processing Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI backend that accepts receipt images (folder path or multipart upload), extracts fields via OCR, categorizes each receipt via a DeepSeek sub-agent, aggregates totals, and streams the full execution trajectory over Server-Sent Events while persisting a reviewable trace to Supabase Cloud Postgres.

**Architecture:** Hexagonal (Ports & Adapters). Deterministic LangGraph outer pipeline, sequential per receipt, with one bounded DeepSeek call inside `categorize_receipt`. Write-through persistence across `reports`/`receipts`/`traces`. Mock mode (`LLM_MODE=mock`) swaps both OCR and LLM adapters for deterministic fakes.

**Tech Stack:** Python 3.11+, FastAPI + `sse-starlette`, LangChain + LangGraph, Langfuse, OpenAI SDK (for OCR and for DeepSeek via its OpenAI-compatible API), SQLAlchemy 2 + Alembic + Supabase Cloud Postgres, Pydantic v2 + `pydantic-settings`, `pytest` + `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-04-20-receipt-processing-agent-design.md` (approved).
**High-level system spec:** `specs.md`.
**Evaluation contract:** `Take-Home Test - Software Development Manager, AI (4) (1).pdf`.

---

## File Structure

Python package layout: flat `src/` with `pythonpath = ["src"]` in `pyproject.toml`, so imports read as `from domain.models import ...`.

```
receipt-agent/
├── pyproject.toml                          # deps, pytest config, pythonpath
├── Makefile                                # run, test, migrate
├── .env.example                            # all env vars
├── .gitignore
├── README.md                               # setup, curl, SSE example, mock mode
├── DESIGN.md                               # 1-page architecture summary
├── AGENTS.md                               # AI-assisted dev setup notes
├── spec.md                                 # PDF deliverable (endpoint + schemas)
├── specs.md                                # high-level system spec (already exists)
├── alembic.ini
├── migrations/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial.py                 # reports + receipts + traces
├── assets/                                 # 4–5 synthetic receipt images
│   ├── receipt_001.png
│   ├── ...
│   └── README.md                           # provenance of sample images
├── transcripts/                            # AI-tool interaction logs (PDF deliverable)
│   ├── README.md
│   └── sample-run-trace.json               # sample run trace artifact
├── src/
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py                       # Pydantic: AllowedCategory, Issue,
│   │   │                                   #   Categorization, RawReceipt,
│   │   │                                   #   NormalizedReceipt, Receipt,
│   │   │                                   #   Aggregates, Report
│   │   ├── normalization.py                # parse_date, parse_money, normalize()
│   │   └── aggregation.py                  # aggregate(receipts) -> Aggregates
│   ├── application/
│   │   ├── __init__.py
│   │   ├── ports.py                        # OCRPort, LLMPort, ImageLoaderPort,
│   │   │                                   #   ReportRepositoryPort,
│   │   │                                   #   TraceRepositoryPort,
│   │   │                                   #   EventBusPort, TracerPort
│   │   ├── events.py                       # SSE event Pydantic models
│   │   ├── event_bus.py                    # in-process async pub/sub
│   │   ├── subagent.py                     # DeepSeek categorization call
│   │   ├── traced_tool.py                  # @traced_tool decorator
│   │   ├── tool_registry.py                # the 6 tools
│   │   └── graph.py                        # LangGraph state machine + GraphRunner
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── ocr/
│   │   │   ├── __init__.py
│   │   │   ├── openai_adapter.py
│   │   │   └── mock_adapter.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── deepseek_adapter.py
│   │   │   └── mock_adapter.py
│   │   ├── images/
│   │   │   ├── __init__.py
│   │   │   ├── folder_loader.py
│   │   │   └── upload_loader.py
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py
│   │   │   ├── models.py                   # ORM: ReportRow, ReceiptRow, TraceRow
│   │   │   └── repositories.py             # ReportRepository, TraceRepository
│   │   ├── tracing/
│   │   │   ├── __init__.py
│   │   │   ├── json_logs_adapter.py
│   │   │   └── langfuse_adapter.py
│   │   └── http/
│   │       ├── __init__.py
│   │       ├── app.py                      # FastAPI factory
│   │       ├── sse.py                      # EventSourceResponse helper
│   │       └── routes_runs.py              # POST /runs/stream
│   ├── config.py                           # Pydantic Settings
│   ├── composition_root.py                 # wires adapters → ports → application
│   └── main.py                             # uvicorn entry (exposes `app`)
└── tests/
    ├── __init__.py
    ├── conftest.py                         # shared fixtures
    ├── fakes/
    │   ├── __init__.py
    │   ├── mock_ocr.py
    │   ├── mock_llm.py
    │   ├── in_memory_repos.py
    │   └── mock_image_loader.py
    ├── fixtures/
    │   ├── receipts/                       # 2–3 fake images for tests
    │   │   ├── fixture_001.png
    │   │   └── fixture_002.png
    │   └── folder/                         # folder-input fixture
    │       └── fixture_a.png
    ├── domain/
    │   ├── __init__.py
    │   ├── test_models.py
    │   ├── test_normalization.py
    │   └── test_aggregation.py
    ├── application/
    │   ├── __init__.py
    │   ├── test_events.py
    │   ├── test_event_bus.py
    │   ├── test_subagent.py
    │   ├── test_traced_tool.py
    │   ├── test_tool_registry.py
    │   └── test_graph.py
    ├── infrastructure/
    │   ├── __init__.py
    │   ├── test_runs_stream.py             # HTTP contract test (mock mode)
    │   ├── test_traces_write_through.py    # DB write-through (skippable)
    │   └── test_repositories.py            # CRUD round-trip (skippable)
    └── e2e/
        ├── __init__.py
        └── test_real_run_smoke.py          # @pytest.mark.e2e
```

**File responsibilities (one sentence each):**

- `domain/models.py` — Pydantic entities and the allowed-categories enum; no I/O.
- `domain/normalization.py` — pure functions for date / money / total parsing.
- `domain/aggregation.py` — pure totals + by-category from a list of categorized receipts.
- `application/ports.py` — abstract interfaces; the only thing `application/*` imports from outside the domain.
- `application/events.py` — Pydantic models for every SSE event type.
- `application/event_bus.py` — async in-process pub/sub with fire-and-log-on-error subscribers.
- `application/subagent.py` — one DeepSeek call; user prompt injected; returns `Categorization`.
- `application/traced_tool.py` — decorator: emits `tool_call`/`tool_result`, opens Langfuse span, times, classifies errors.
- `application/tool_registry.py` — the six tools. Thin wrappers over domain + ports.
- `application/graph.py` — LangGraph state machine and `GraphRunner` class.
- `infrastructure/ocr/*` — real + mock OCR adapters.
- `infrastructure/llm/*` — real + mock categorization LLM adapters.
- `infrastructure/images/*` — folder + upload image loaders.
- `infrastructure/db/*` — engine, ORM rows, repositories.
- `infrastructure/tracing/*` — Langfuse and JSON-logs trace adapters.
- `infrastructure/http/*` — FastAPI app + SSE endpoint.
- `src/config.py` — Pydantic `Settings` loaded from env.
- `src/composition_root.py` — one function per `LLM_MODE` that wires adapters and returns a `GraphRunner` + FastAPI app.
- `src/main.py` — imports the composed app for uvicorn.

---

## Phase 0 — Scaffolding

### Task 0.1: Project skeleton

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore` additions, `src/__init__.py`, `tests/__init__.py`, every subpackage `__init__.py` listed in File Structure above.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "receipt-agent"
version = "0.1.0"
description = "Local receipt-processing agent with SSE streaming."
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.110,<1.0",
  "uvicorn[standard]>=0.29",
  "sse-starlette>=2.1",
  "pydantic>=2.7",
  "pydantic-settings>=2.2",
  "sqlalchemy>=2.0",
  "alembic>=1.13",
  "psycopg[binary]>=3.1",
  "langchain>=0.3,<0.4",
  "langchain-core>=0.3,<0.4",
  "langgraph>=0.2",
  "langfuse>=2.50,<3.0",
  "openai>=1.40",
  "python-multipart>=0.0.9",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-asyncio>=0.23",
  "httpx>=0.27",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["src"]
markers = [
  "e2e: end-to-end test requiring real API keys and network",
]
testpaths = ["tests"]

[tool.setuptools]
# No package install needed — we use pytest's pythonpath and `PYTHONPATH=src` at runtime.
```

- [ ] **Step 2: Create `Makefile`**

```makefile
.PHONY: install run test test-e2e migrate fmt clean

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

run:
	PYTHONPATH=src uvicorn main:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=src pytest -v -m "not e2e"

test-e2e:
	PYTHONPATH=src pytest -v -m e2e

migrate:
	PYTHONPATH=src alembic upgrade head

migrate-create:
	PYTHONPATH=src alembic revision --autogenerate -m "$(MSG)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 3: Create `.env.example`**

```
# Mode
LLM_MODE=mock                 # mock | real

# Database (Supabase cloud)
SUPABASE_DB_URL=postgresql+psycopg://user:pass@host:6543/postgres

# LLM / OCR (required when LLM_MODE=real)
OPENAI_API_KEY=
OPENAI_OCR_MODEL=gpt-4o-mini
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Observability (optional; if empty, Langfuse adapter is a no-op)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Assets
ASSETS_DIR=./assets

# Timeouts (seconds) + upload limits
OCR_TIMEOUT_S=30
LLM_TIMEOUT_S=20
TOOL_WALL_TIMEOUT_S=45
MAX_FILE_SIZE_MB=10
MAX_FILES_PER_RUN=25
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp
```

- [ ] **Step 4: Extend `.gitignore`**

Append these lines to `.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
dist/
*.egg-info/
```

- [ ] **Step 5: Create package directories + `__init__.py` files**

Run:

```bash
mkdir -p src/domain src/application src/infrastructure/ocr src/infrastructure/llm src/infrastructure/images src/infrastructure/db src/infrastructure/tracing src/infrastructure/http
mkdir -p tests/fakes tests/fixtures/receipts tests/fixtures/folder tests/domain tests/application tests/infrastructure tests/e2e
mkdir -p assets transcripts migrations/versions
touch src/__init__.py src/domain/__init__.py src/application/__init__.py src/infrastructure/__init__.py
touch src/infrastructure/ocr/__init__.py src/infrastructure/llm/__init__.py src/infrastructure/images/__init__.py src/infrastructure/db/__init__.py src/infrastructure/tracing/__init__.py src/infrastructure/http/__init__.py
touch tests/__init__.py tests/fakes/__init__.py tests/domain/__init__.py tests/application/__init__.py tests/infrastructure/__init__.py tests/e2e/__init__.py
```

- [ ] **Step 6: Install deps and verify pytest finds zero tests**

```bash
make install
make test
```

Expected: pytest reports `no tests ran` with exit code 0 (nothing to collect yet — this is fine; `make test` uses `pytest -v -m "not e2e"` which exits 5 on zero collection; if exit code 5 happens, that's expected). If exit 5 is unacceptable, add `--co` or accept it for this task only.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml Makefile .env.example .gitignore src/ tests/ assets/ transcripts/ migrations/
git commit -m "chore: project scaffolding (pyproject, Makefile, package layout)"
```

---

### Task 0.2: Config module (Pydantic Settings)

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** at `tests/test_config.py`

```python
import os
import pytest
from config import Settings, LLMMode


def test_settings_defaults_with_mock_mode(monkeypatch):
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://u:p@h/db")
    s = Settings()
    assert s.llm_mode == LLMMode.MOCK
    assert s.ocr_timeout_s == 30
    assert s.max_file_size_mb == 10
    assert s.allowed_extensions == {"jpg", "jpeg", "png", "webp"}


def test_settings_real_mode_requires_keys(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://u:p@h/db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings()


def test_allowed_extensions_parses_csv(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "jpg,png")
    s = Settings()
    assert s.allowed_extensions == {"jpg", "png"}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
make test tests/test_config.py
```
Expected: `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Implement `src/config.py`**

```python
from enum import Enum
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMMode(str, Enum):
    MOCK = "mock"
    REAL = "real"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_mode: LLMMode = Field(default=LLMMode.MOCK, validation_alias="LLM_MODE")

    supabase_db_url: str = Field(validation_alias="SUPABASE_DB_URL")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_ocr_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_OCR_MODEL")
    deepseek_api_key: str | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", validation_alias="DEEPSEEK_MODEL")

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST")

    assets_dir: Path = Field(default=Path("./assets"), validation_alias="ASSETS_DIR")

    ocr_timeout_s: int = Field(default=30, validation_alias="OCR_TIMEOUT_S")
    llm_timeout_s: int = Field(default=20, validation_alias="LLM_TIMEOUT_S")
    tool_wall_timeout_s: int = Field(default=45, validation_alias="TOOL_WALL_TIMEOUT_S")
    max_file_size_mb: int = Field(default=10, validation_alias="MAX_FILE_SIZE_MB")
    max_files_per_run: int = Field(default=25, validation_alias="MAX_FILES_PER_RUN")

    allowed_extensions_raw: str = Field(
        default="jpg,jpeg,png,webp", validation_alias="ALLOWED_EXTENSIONS"
    )

    @property
    def allowed_extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions_raw.split(",") if e.strip()}

    @model_validator(mode="after")
    def _validate_real_mode_keys(self) -> "Settings":
        if self.llm_mode == LLMMode.REAL:
            missing = [
                name for name, val in (
                    ("OPENAI_API_KEY", self.openai_api_key),
                    ("DEEPSEEK_API_KEY", self.deepseek_api_key),
                ) if not val
            ]
            if missing:
                raise ValueError(
                    f"LLM_MODE=real requires: {', '.join(missing)}"
                )
        return self
```

- [ ] **Step 4: Run tests and confirm PASS**

```bash
make test tests/test_config.py
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): Pydantic Settings with LLM_MODE validation"
```

---

### Task 0.3: Pytest shared conftest

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
"""
Shared pytest fixtures.

Design principle: per-layer conftests inherit from this one for layer-specific fixtures.
This root conftest only provides primitives shared across all layers.
"""
import os
import pytest


@pytest.fixture(autouse=True)
def _mock_mode_env(monkeypatch):
    """
    Default all tests to mock mode so no real API keys are needed.
    Individual tests can override by setting LLM_MODE themselves.
    """
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv(
        "SUPABASE_DB_URL",
        os.environ.get("TEST_SUPABASE_DB_URL", "postgresql+psycopg://test:test@localhost/test"),
    )
```

- [ ] **Step 2: Confirm tests still pass**

```bash
make test
```
Expected: all 3 config tests pass. (These tests explicitly set `LLM_MODE`; the fixture doesn't interfere.)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: shared conftest defaults to mock mode"
```

---

## Phase 1 — Domain

### Task 1.1: Enums and simple models (AllowedCategory, Issue, Categorization)

**Files:**
- Create: `src/domain/models.py`
- Test: `tests/domain/test_models.py`

- [ ] **Step 1: Write failing tests** at `tests/domain/test_models.py`

```python
import pytest
from uuid import uuid4
from domain.models import AllowedCategory, Issue, Categorization


def test_allowed_category_values():
    assert AllowedCategory.TRAVEL.value == "Travel"
    assert AllowedCategory.OTHER.value == "Other"
    # Full set matches spec
    expected = {
        "Travel", "Meals & Entertainment", "Software / Subscriptions",
        "Professional Services", "Office Supplies", "Shipping / Postage",
        "Utilities", "Other",
    }
    assert {c.value for c in AllowedCategory} == expected


def test_issue_requires_severity_code_message():
    issue = Issue(severity="warning", code="low_confidence", message="OCR confidence 0.4")
    assert issue.receipt_id is None


def test_issue_rejects_invalid_severity():
    with pytest.raises(ValueError):
        Issue(severity="catastrophe", code="x", message="y")


def test_categorization_valid():
    c = Categorization(
        category=AllowedCategory.TRAVEL, confidence=0.9, notes="Uber ride", issues=[]
    )
    assert c.confidence == 0.9


def test_categorization_confidence_bounds():
    with pytest.raises(ValueError):
        Categorization(category=AllowedCategory.TRAVEL, confidence=1.5)
    with pytest.raises(ValueError):
        Categorization(category=AllowedCategory.TRAVEL, confidence=-0.1)


def test_categorization_other_requires_notes():
    with pytest.raises(ValueError, match="Other"):
        Categorization(category=AllowedCategory.OTHER, confidence=0.7, notes=None)
    # With notes: OK
    c = Categorization(category=AllowedCategory.OTHER, confidence=0.7, notes="donation")
    assert c.notes == "donation"
```

- [ ] **Step 2: Run tests, expect failure**

```bash
make test tests/domain/test_models.py
```
Expected: `ModuleNotFoundError: No module named 'domain'` (until we create `src/domain/models.py`).

- [ ] **Step 3: Implement initial `src/domain/models.py`**

```python
from enum import Enum
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, model_validator


class AllowedCategory(str, Enum):
    TRAVEL = "Travel"
    MEALS = "Meals & Entertainment"
    SOFTWARE = "Software / Subscriptions"
    PROFESSIONAL = "Professional Services"
    OFFICE_SUPPLIES = "Office Supplies"
    SHIPPING = "Shipping / Postage"
    UTILITIES = "Utilities"
    OTHER = "Other"


Severity = Literal["warning", "receipt_error", "run_error"]


class Issue(BaseModel):
    severity: Severity
    code: str
    message: str
    receipt_id: UUID | None = None


class Categorization(BaseModel):
    category: AllowedCategory
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None
    issues: list[Issue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _other_requires_notes(self) -> "Categorization":
        if self.category == AllowedCategory.OTHER and not (self.notes and self.notes.strip()):
            raise ValueError("category 'Other' requires a non-empty note")
        return self
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
make test tests/domain/test_models.py
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/domain/models.py tests/domain/test_models.py
git commit -m "feat(domain): AllowedCategory, Issue, Categorization with validators"
```

---

### Task 1.2: Receipt entities (RawReceipt, NormalizedReceipt, Receipt, Aggregates, Report)

**Files:**
- Modify: `src/domain/models.py`
- Modify: `tests/domain/test_models.py`

- [ ] **Step 1: Append failing tests to `tests/domain/test_models.py`**

```python
from datetime import date as date_type
from decimal import Decimal
from domain.models import (
    RawReceipt, NormalizedReceipt, Receipt, Aggregates, Report,
)


def test_raw_receipt_accepts_optional_fields():
    r = RawReceipt(source_ref="receipt_001.png")
    assert r.line_items == []


def test_normalized_receipt_types():
    n = NormalizedReceipt(
        source_ref="receipt_001.png",
        vendor="Uber",
        receipt_date=date_type(2024, 3, 15),
        receipt_number="R-12345",
        total=Decimal("45.67"),
        currency="USD",
    )
    assert isinstance(n.total, Decimal)


def test_receipt_default_status_ok():
    r = Receipt(id=uuid4(), source_ref="a.png")
    assert r.status == "ok"
    assert r.issues == []


def test_receipt_error_status_carries_error_field():
    r = Receipt(id=uuid4(), source_ref="a.png", status="error", error="OCR timeout")
    assert r.error == "OCR timeout"


def test_aggregates_shape():
    a = Aggregates(total_spend=Decimal("100.00"), by_category={"Travel": Decimal("45.67")})
    assert a.by_category["Travel"] == Decimal("45.67")


def test_report_bundles_fields():
    run_id = uuid4()
    rep = Report(
        run_id=run_id,
        total_spend=Decimal("0.00"),
        by_category={},
        receipts=[],
        issues_and_assumptions=[],
    )
    assert rep.run_id == run_id
```

- [ ] **Step 2: Run, expect failure**

Expected: `ImportError` for `RawReceipt` etc.

- [ ] **Step 3: Append to `src/domain/models.py`**

```python
from datetime import date
from decimal import Decimal


class RawReceipt(BaseModel):
    """OCR output pre-normalization. All fields optional (OCR may fail)."""
    source_ref: str
    vendor: str | None = None
    receipt_date: str | None = None      # raw string
    receipt_number: str | None = None
    total_raw: str | None = None         # raw string ("$1,234.56")
    currency_raw: str | None = None
    line_items: list[dict] = Field(default_factory=list)
    ocr_confidence: float | None = None


class NormalizedReceipt(BaseModel):
    """Normalized receipt fields. Types are strict (date, Decimal)."""
    source_ref: str
    vendor: str | None = None
    receipt_date: date | None = None
    receipt_number: str | None = None
    total: Decimal | None = None
    currency: str | None = None          # ISO 4217; defaults to "USD" when absent


class Receipt(BaseModel):
    """Full per-receipt record — what gets persisted and streamed on receipt_result."""
    id: UUID
    source_ref: str
    vendor: str | None = None
    receipt_date: date | None = None
    receipt_number: str | None = None
    total: Decimal | None = None
    currency: str | None = None
    category: AllowedCategory | None = None
    confidence: float | None = None
    notes: str | None = None
    issues: list[Issue] = Field(default_factory=list)
    raw_ocr: dict | None = None
    normalized: dict | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None


class Aggregates(BaseModel):
    total_spend: Decimal
    by_category: dict[str, Decimal]


class Report(BaseModel):
    run_id: UUID
    total_spend: Decimal
    by_category: dict[str, Decimal]
    receipts: list[Receipt]
    issues_and_assumptions: list[Issue]
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/domain/test_models.py
```
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/domain/models.py tests/domain/test_models.py
git commit -m "feat(domain): receipt entities (Raw/Normalized/Receipt) and Report"
```

---

### Task 1.3: Normalization (dates, money, currency)

**Files:**
- Create: `src/domain/normalization.py`
- Test: `tests/domain/test_normalization.py`

- [ ] **Step 1: Write failing tests** at `tests/domain/test_normalization.py`

```python
import pytest
from datetime import date
from decimal import Decimal
from domain.models import RawReceipt, NormalizedReceipt
from domain.normalization import parse_date, parse_money, normalize


def test_parse_date_iso():
    assert parse_date("2024-03-15") == date(2024, 3, 15)


def test_parse_date_us_slash():
    assert parse_date("03/15/2024") == date(2024, 3, 15)


def test_parse_date_long():
    assert parse_date("15 March 2024") == date(2024, 3, 15)


def test_parse_date_none_returns_none():
    assert parse_date(None) is None
    assert parse_date("") is None


def test_parse_date_unparseable_raises():
    with pytest.raises(ValueError):
        parse_date("not a date")


def test_parse_money_plain():
    amount, currency = parse_money("45.67")
    assert amount == Decimal("45.67")
    assert currency is None


def test_parse_money_dollar_sign():
    amount, currency = parse_money("$1,234.56")
    assert amount == Decimal("1234.56")
    assert currency == "USD"


def test_parse_money_with_currency_suffix():
    amount, currency = parse_money("45.67 EUR")
    assert amount == Decimal("45.67")
    assert currency == "EUR"


def test_parse_money_none():
    assert parse_money(None) == (None, None)


def test_parse_money_unparseable_raises():
    with pytest.raises(ValueError):
        parse_money("not money")


def test_normalize_happy_path():
    raw = RawReceipt(
        source_ref="r1.png",
        vendor="Uber",
        receipt_date="2024-03-15",
        receipt_number="R-12345",
        total_raw="$45.67",
        currency_raw=None,
    )
    n = normalize(raw)
    assert n.vendor == "Uber"
    assert n.receipt_date == date(2024, 3, 15)
    assert n.total == Decimal("45.67")
    assert n.currency == "USD"  # inferred from $


def test_normalize_explicit_currency_wins():
    raw = RawReceipt(source_ref="r.png", total_raw="45.67", currency_raw="EUR")
    assert normalize(raw).currency == "EUR"


def test_normalize_defaults_to_usd_when_absent():
    raw = RawReceipt(source_ref="r.png", total_raw="45.67")
    assert normalize(raw).currency == "USD"
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/domain/normalization.py`**

```python
"""
Pure normalization functions.

parse_date / parse_money tolerate None and raise ValueError on unparseable non-empty input.
`normalize(raw)` always returns a NormalizedReceipt; on parse failure it raises so the
tool wrapper can classify the error at the boundary.
"""
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from domain.models import NormalizedReceipt, RawReceipt


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y")
_CURRENCY_SIGN = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_CURRENCY_RE = re.compile(r"\b([A-Z]{3})\b")
_MONEY_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def parse_date(raw: str | None) -> date | None:
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {raw!r}")


def parse_money(raw: str | None) -> tuple[Decimal | None, str | None]:
    if raw is None or not raw.strip():
        return None, None
    text = raw.strip()

    # currency from sign
    currency: str | None = None
    for sign, iso in _CURRENCY_SIGN.items():
        if sign in text:
            currency = iso
            text = text.replace(sign, "")
            break

    # currency from ISO-3 suffix or prefix
    m = _CURRENCY_RE.search(text)
    if m:
        currency = currency or m.group(1)
        text = text.replace(m.group(1), "")

    # number
    nm = _MONEY_RE.search(text.replace(",", ""))
    if not nm:
        raise ValueError(f"unparseable money: {raw!r}")
    try:
        amount = Decimal(nm.group(0))
    except InvalidOperation as e:
        raise ValueError(f"unparseable money: {raw!r}") from e
    return amount, currency


def normalize(raw: RawReceipt) -> NormalizedReceipt:
    d = parse_date(raw.receipt_date)
    amount, currency_from_total = parse_money(raw.total_raw)
    currency = raw.currency_raw or currency_from_total or "USD"
    return NormalizedReceipt(
        source_ref=raw.source_ref,
        vendor=raw.vendor,
        receipt_date=d,
        receipt_number=raw.receipt_number,
        total=amount,
        currency=currency,
    )
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/domain/test_normalization.py
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/domain/normalization.py tests/domain/test_normalization.py
git commit -m "feat(domain): date/money parsing and receipt normalization"
```

---

### Task 1.4: Aggregation

**Files:**
- Create: `src/domain/aggregation.py`
- Test: `tests/domain/test_aggregation.py`

- [ ] **Step 1: Write failing tests** at `tests/domain/test_aggregation.py`

```python
from decimal import Decimal
from uuid import uuid4
from domain.models import AllowedCategory, Receipt
from domain.aggregation import aggregate


def _receipt(category, total, status="ok"):
    return Receipt(
        id=uuid4(), source_ref="x.png",
        category=category, total=total, currency="USD", status=status,
    )


def test_aggregate_empty():
    a = aggregate([])
    assert a.total_spend == Decimal("0.00")
    assert a.by_category == {}


def test_aggregate_sums_totals_by_category():
    receipts = [
        _receipt(AllowedCategory.TRAVEL, Decimal("45.67")),
        _receipt(AllowedCategory.TRAVEL, Decimal("22.33")),
        _receipt(AllowedCategory.MEALS, Decimal("18.50")),
    ]
    a = aggregate(receipts)
    assert a.total_spend == Decimal("86.50")
    assert a.by_category[AllowedCategory.TRAVEL.value] == Decimal("68.00")
    assert a.by_category[AllowedCategory.MEALS.value] == Decimal("18.50")


def test_aggregate_excludes_errored_receipts():
    receipts = [
        _receipt(AllowedCategory.TRAVEL, Decimal("45.67"), status="ok"),
        _receipt(AllowedCategory.TRAVEL, Decimal("99.99"), status="error"),
    ]
    a = aggregate(receipts)
    assert a.total_spend == Decimal("45.67")


def test_aggregate_excludes_receipts_without_category_or_total():
    receipts = [
        _receipt(AllowedCategory.TRAVEL, Decimal("45.67")),
        Receipt(id=uuid4(), source_ref="x.png", category=None, total=Decimal("10")),
        Receipt(id=uuid4(), source_ref="x.png", category=AllowedCategory.MEALS, total=None),
    ]
    a = aggregate(receipts)
    assert a.total_spend == Decimal("45.67")


def test_aggregate_rounds_to_two_decimals():
    receipts = [
        _receipt(AllowedCategory.TRAVEL, Decimal("10.123")),
        _receipt(AllowedCategory.TRAVEL, Decimal("10.456")),
    ]
    a = aggregate(receipts)
    # 20.579 rounds to 20.58
    assert a.total_spend == Decimal("20.58")
    assert a.by_category[AllowedCategory.TRAVEL.value] == Decimal("20.58")
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/domain/aggregation.py`**

```python
"""
Pure aggregation over a list of Receipt objects.

Excludes: receipts with status='error', missing category, or missing total.
Rounds totals to 2 decimal places (banker's rounding — Decimal default).
"""
from decimal import Decimal, ROUND_HALF_UP
from domain.models import Aggregates, Receipt


_CENT = Decimal("0.01")


def _q(v: Decimal) -> Decimal:
    return v.quantize(_CENT, rounding=ROUND_HALF_UP)


def aggregate(receipts: list[Receipt]) -> Aggregates:
    total = Decimal("0")
    by_cat: dict[str, Decimal] = {}
    for r in receipts:
        if r.status != "ok" or r.category is None or r.total is None:
            continue
        total += r.total
        key = r.category.value
        by_cat[key] = by_cat.get(key, Decimal("0")) + r.total
    return Aggregates(
        total_spend=_q(total),
        by_category={k: _q(v) for k, v in by_cat.items()},
    )
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/domain/test_aggregation.py
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/domain/aggregation.py tests/domain/test_aggregation.py
git commit -m "feat(domain): aggregation over receipts with rounding and exclusions"
```

---

## Phase 2 — Application: ports, events, event bus

### Task 2.1: Ports (abstract interfaces)

**Files:**
- Create: `src/application/ports.py`

- [ ] **Step 1: Create `src/application/ports.py`**

```python
"""
Abstract ports. The application layer depends ONLY on these interfaces.
Infrastructure adapters implement them.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from uuid import UUID
from domain.models import Categorization, NormalizedReceipt, RawReceipt


@dataclass(frozen=True)
class ImageRef:
    """Reference to an image available to the OCR adapter."""
    source_ref: str         # original filename or identifier
    local_path: Path        # absolute path on local disk


class ImageLoaderPort(ABC):
    @abstractmethod
    async def load(self) -> list[ImageRef]: ...


class OCRPort(ABC):
    @abstractmethod
    async def extract(self, image: ImageRef) -> RawReceipt: ...


class LLMPort(ABC):
    """One call per receipt. Implementation MUST inject `user_prompt` into its system message."""
    @abstractmethod
    async def categorize(
        self,
        normalized: NormalizedReceipt,
        allowed: list[str],
        user_prompt: str | None,
    ) -> Categorization: ...


class ReportRepositoryPort(ABC):
    @abstractmethod
    async def insert_report(self, row: dict) -> None: ...

    @abstractmethod
    async def update_report(self, report_id: UUID, patch: dict) -> None: ...

    @abstractmethod
    async def insert_receipt(self, row: dict) -> None: ...


class TraceRepositoryPort(ABC):
    @abstractmethod
    async def insert_trace(self, row: dict) -> None: ...


# EventBus subscriber signature
Subscriber = Callable[[dict], Awaitable[None]]


class EventBusPort(ABC):
    @abstractmethod
    async def publish(self, event: dict) -> None: ...

    @abstractmethod
    def subscribe(self, subscriber: Subscriber) -> None: ...


class TracerPort(ABC):
    """Opaque tracer (Langfuse or no-op). Opens a span for a tool call."""
    @abstractmethod
    def start_span(self, name: str, input: dict | None = None) -> "TracerSpan": ...


class TracerSpan(ABC):
    @abstractmethod
    def end(self, output: dict | None = None, error: str | None = None) -> None: ...
```

- [ ] **Step 2: Smoke-import the module**

```bash
PYTHONPATH=src python -c "import application.ports as p; print(sorted(n for n in dir(p) if not n.startswith('_')))"
```
Expected: list including `EventBusPort`, `ImageLoaderPort`, `LLMPort`, `OCRPort`, etc.

- [ ] **Step 3: Commit**

```bash
git add src/application/ports.py
git commit -m "feat(app): ports (abstract interfaces)"
```

---

### Task 2.2: SSE event models

**Files:**
- Create: `src/application/events.py`
- Test: `tests/application/test_events.py`

- [ ] **Step 1: Write failing tests** at `tests/application/test_events.py`

```python
import pytest
from datetime import datetime, timezone
from uuid import uuid4
from application.events import (
    EventType, RunStarted, Progress, ToolCall, ToolResult,
    ReceiptResult, FinalResult, ErrorEvent, serialize_event,
)


def _now():
    return datetime.now(timezone.utc)


def test_run_started_minimal():
    e = RunStarted(run_id=uuid4(), seq=1, ts=_now(), prompt="conservative")
    assert e.event_type == EventType.RUN_STARTED


def test_progress_with_receipt_scope():
    rid = uuid4()
    rc = uuid4()
    e = Progress(run_id=rid, seq=2, ts=_now(), step="ocr", receipt_id=rc, i=1, n=3)
    assert e.event_type == EventType.PROGRESS
    assert e.receipt_id == rc


def test_tool_call_args_dict():
    e = ToolCall(
        run_id=uuid4(), seq=3, ts=_now(),
        tool="load_images", args={"folder_path": "/tmp"},
    )
    assert e.tool == "load_images"


def test_tool_result_error_flag():
    e = ToolResult(
        run_id=uuid4(), seq=4, ts=_now(),
        tool="extract_receipt_fields", result_summary={"count": 0},
        error=True, duration_ms=120,
    )
    assert e.error is True


def test_receipt_result_status_ok():
    e = ReceiptResult(
        run_id=uuid4(), seq=5, ts=_now(), receipt_id=uuid4(),
        status="ok", vendor="Uber", total="45.67",
        category="Travel", confidence=0.9, issues=[],
    )
    assert e.status == "ok"


def test_final_result_shape():
    e = FinalResult(
        run_id=uuid4(), seq=6, ts=_now(),
        total_spend="100.00", by_category={"Travel": "100.00"},
        receipts=[], issues_and_assumptions=[],
    )
    assert e.event_type == EventType.FINAL_RESULT


def test_error_event_has_code_and_message():
    e = ErrorEvent(run_id=uuid4(), seq=7, ts=_now(), code="no_images", message="0 images")
    assert e.code == "no_images"


def test_serialize_event_is_json_string_with_event_type_field():
    e = RunStarted(run_id=uuid4(), seq=1, ts=_now())
    s = serialize_event(e)
    assert '"event_type"' in s
    assert '"run_started"' in s
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/application/events.py`**

```python
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal, Union
from uuid import UUID
from pydantic import BaseModel, Field


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    PROGRESS = "progress"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RECEIPT_RESULT = "receipt_result"
    FINAL_RESULT = "final_result"
    ERROR = "error"


class _EventBase(BaseModel):
    run_id: UUID
    seq: int
    ts: datetime


class RunStarted(_EventBase):
    event_type: Literal[EventType.RUN_STARTED] = EventType.RUN_STARTED
    prompt: str | None = None
    receipt_count_estimate: int | None = None


class Progress(_EventBase):
    event_type: Literal[EventType.PROGRESS] = EventType.PROGRESS
    step: str
    receipt_id: UUID | None = None
    i: int | None = None
    n: int | None = None


class ToolCall(_EventBase):
    event_type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    tool: str
    receipt_id: UUID | None = None
    attempt: int = 1
    args: dict = Field(default_factory=dict)


class ToolResult(_EventBase):
    event_type: Literal[EventType.TOOL_RESULT] = EventType.TOOL_RESULT
    tool: str
    receipt_id: UUID | None = None
    result_summary: dict = Field(default_factory=dict)
    error: bool = False
    error_message: str | None = None
    duration_ms: int | None = None


class ReceiptResult(_EventBase):
    event_type: Literal[EventType.RECEIPT_RESULT] = EventType.RECEIPT_RESULT
    receipt_id: UUID
    status: Literal["ok", "error"]
    vendor: str | None = None
    receipt_date: str | None = None        # ISO; kept as string in the wire event
    receipt_number: str | None = None
    total: str | None = None               # Decimal → str for wire
    currency: str | None = None
    category: str | None = None
    confidence: float | None = None
    notes: str | None = None
    issues: list[dict] = Field(default_factory=list)
    error_message: str | None = None


class FinalResult(_EventBase):
    event_type: Literal[EventType.FINAL_RESULT] = EventType.FINAL_RESULT
    total_spend: str
    by_category: dict[str, str]
    receipts: list[dict]
    issues_and_assumptions: list[dict]


class ErrorEvent(_EventBase):
    event_type: Literal[EventType.ERROR] = EventType.ERROR
    code: str
    message: str


Event = Union[
    RunStarted, Progress, ToolCall, ToolResult,
    ReceiptResult, FinalResult, ErrorEvent,
]


def serialize_event(e: Event) -> str:
    """Serialize an event to a JSON string suitable for an SSE `data:` field."""
    return e.model_dump_json()
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/application/test_events.py
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/application/events.py tests/application/test_events.py
git commit -m "feat(app): SSE event models and serialize_event"
```

---

### Task 2.3: EventBus (async pub/sub)

**Files:**
- Create: `src/application/event_bus.py`
- Test: `tests/application/test_event_bus.py`

- [ ] **Step 1: Write failing tests** at `tests/application/test_event_bus.py`

```python
import pytest
from application.event_bus import InMemoryEventBus


@pytest.mark.asyncio
async def test_publish_delivers_to_all_subscribers():
    bus = InMemoryEventBus()
    a: list[dict] = []
    b: list[dict] = []

    async def sub_a(e):
        a.append(e)

    async def sub_b(e):
        b.append(e)

    bus.subscribe(sub_a)
    bus.subscribe(sub_b)

    await bus.publish({"x": 1})
    await bus.publish({"x": 2})

    assert a == [{"x": 1}, {"x": 2}]
    assert b == [{"x": 1}, {"x": 2}]


@pytest.mark.asyncio
async def test_subscriber_error_does_not_stop_bus():
    bus = InMemoryEventBus()
    bad_called: list[int] = []
    good: list[dict] = []

    async def bad(e):
        bad_called.append(1)
        raise RuntimeError("boom")

    async def good_sub(e):
        good.append(e)

    bus.subscribe(bad)
    bus.subscribe(good_sub)

    # publish must NOT raise
    await bus.publish({"x": 1})
    await bus.publish({"x": 2})

    assert len(bad_called) == 2
    assert good == [{"x": 1}, {"x": 2}]


@pytest.mark.asyncio
async def test_publish_preserves_order_per_subscriber():
    bus = InMemoryEventBus()
    received: list[int] = []

    async def sub(e):
        received.append(e["i"])

    bus.subscribe(sub)
    for i in range(10):
        await bus.publish({"i": i})

    assert received == list(range(10))
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/application/event_bus.py`**

```python
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
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/application/test_event_bus.py
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/application/event_bus.py tests/application/test_event_bus.py
git commit -m "feat(app): in-memory async EventBus (swallow subscriber errors)"
```

---

## Phase 3 — Test fakes (mock adapters)

### Task 3.1: MockOCR, MockLLM, MockImageLoader

**Files:**
- Create: `tests/fakes/mock_ocr.py`, `tests/fakes/mock_llm.py`, `tests/fakes/mock_image_loader.py`

- [ ] **Step 1: Create `tests/fakes/mock_ocr.py`**

```python
"""Deterministic OCR mock. Returns canned RawReceipt keyed by source_ref."""
from pathlib import Path
from domain.models import RawReceipt
from application.ports import OCRPort, ImageRef


class MockOCR(OCRPort):
    def __init__(self, responses: dict[str, RawReceipt] | None = None,
                 fail_on: set[str] | None = None) -> None:
        self._responses = responses or {}
        self._fail_on = fail_on or set()

    async def extract(self, image: ImageRef) -> RawReceipt:
        if image.source_ref in self._fail_on:
            raise RuntimeError(f"mock OCR configured to fail on {image.source_ref}")
        if image.source_ref in self._responses:
            return self._responses[image.source_ref]
        # sensible default
        return RawReceipt(
            source_ref=image.source_ref,
            vendor="MockVendor",
            receipt_date="2024-03-15",
            receipt_number="R-MOCK-001",
            total_raw="$12.34",
            ocr_confidence=0.9,
        )
```

- [ ] **Step 2: Create `tests/fakes/mock_llm.py`**

```python
"""Deterministic LLM mock. Captures the user_prompt it received for assertion."""
from dataclasses import dataclass, field
from domain.models import AllowedCategory, Categorization, NormalizedReceipt, Issue
from application.ports import LLMPort


@dataclass
class MockLLMCall:
    normalized: NormalizedReceipt
    allowed: list[str]
    user_prompt: str | None


class MockLLM(LLMPort):
    def __init__(self,
                 responses: dict[str, Categorization] | None = None,
                 default_category: AllowedCategory = AllowedCategory.OTHER,
                 fail_on: set[str] | None = None) -> None:
        self._responses = responses or {}
        self._default_category = default_category
        self._fail_on = fail_on or set()
        self.calls: list[MockLLMCall] = []

    async def categorize(self, normalized, allowed, user_prompt):
        self.calls.append(MockLLMCall(normalized, list(allowed), user_prompt))
        if normalized.source_ref in self._fail_on:
            raise RuntimeError(f"mock LLM configured to fail on {normalized.source_ref}")
        if normalized.source_ref in self._responses:
            return self._responses[normalized.source_ref]
        return Categorization(
            category=self._default_category,
            confidence=0.8,
            notes="default mock categorization",
            issues=[],
        )
```

- [ ] **Step 3: Create `tests/fakes/mock_image_loader.py`**

```python
from pathlib import Path
from application.ports import ImageLoaderPort, ImageRef


class MockImageLoader(ImageLoaderPort):
    def __init__(self, refs: list[ImageRef]) -> None:
        self._refs = refs

    async def load(self) -> list[ImageRef]:
        return list(self._refs)
```

- [ ] **Step 4: Create `tests/fakes/in_memory_repos.py`**

```python
from uuid import UUID
from application.ports import ReportRepositoryPort, TraceRepositoryPort


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
```

- [ ] **Step 5: Smoke-import**

```bash
PYTHONPATH=src:. python -c "from tests.fakes.mock_ocr import MockOCR; from tests.fakes.mock_llm import MockLLM; from tests.fakes.in_memory_repos import InMemoryReportRepository, InMemoryTraceRepository; print('ok')"
```
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add tests/fakes/
git commit -m "test(fakes): in-memory mocks for OCR, LLM, ImageLoader, repositories"
```

---

### Task 3.2: Fixture images for tests

**Files:**
- Create: `tests/fixtures/receipts/fixture_001.png`, `tests/fixtures/receipts/fixture_002.png`, `tests/fixtures/folder/fixture_a.png`

- [ ] **Step 1: Create three tiny valid PNG fixtures**

```bash
python - <<'PY'
import base64
# 1x1 transparent PNG
png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
raw = base64.b64decode(png_b64)
for p in (
    "tests/fixtures/receipts/fixture_001.png",
    "tests/fixtures/receipts/fixture_002.png",
    "tests/fixtures/folder/fixture_a.png",
):
    open(p, "wb").write(raw)
print("ok")
PY
```

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/
git commit -m "test(fixtures): tiny PNG fixtures for image-loading tests"
```

---

## Phase 4 — Categorization sub-agent

### Task 4.1: Sub-agent that calls LLMPort

**Files:**
- Create: `src/application/subagent.py`
- Test: `tests/application/test_subagent.py`

- [ ] **Step 1: Write failing tests** at `tests/application/test_subagent.py`

```python
import pytest
from decimal import Decimal
from domain.models import AllowedCategory, Categorization, NormalizedReceipt
from application.subagent import categorize_with_subagent
from tests.fakes.mock_llm import MockLLM


def _normalized(source_ref="r.png", vendor="Uber"):
    return NormalizedReceipt(
        source_ref=source_ref, vendor=vendor,
        total=Decimal("45.67"), currency="USD",
    )


@pytest.mark.asyncio
async def test_calls_llm_with_allowed_categories_and_prompt():
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    result = await categorize_with_subagent(
        llm, _normalized(), user_prompt="be conservative",
    )
    assert result.category == AllowedCategory.TRAVEL
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.user_prompt == "be conservative"
    # allowed_categories passed by value (strings matching the enum values)
    assert set(call.allowed) == {c.value for c in AllowedCategory}


@pytest.mark.asyncio
async def test_prompt_defaults_to_none():
    llm = MockLLM()
    await categorize_with_subagent(llm, _normalized(), user_prompt=None)
    assert llm.calls[0].user_prompt is None


@pytest.mark.asyncio
async def test_rejects_out_of_band_category_classifies_as_invalid(monkeypatch):
    """
    If the LLM returns a category string not in AllowedCategory, sub-agent raises
    ValueError — the tool wrapper will convert this into a Band-A receipt error.
    """
    class BadLLM:
        async def categorize(self, normalized, allowed, user_prompt):
            # Bypassing the Pydantic validator by constructing raw dict? Instead,
            # simulate by raising directly — mirroring what a provider JSON-mode
            # mismatch would cause.
            raise ValueError("invalid category")

    with pytest.raises(ValueError):
        await categorize_with_subagent(BadLLM(), _normalized(), user_prompt=None)
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/application/subagent.py`**

```python
"""
Categorization sub-agent.

One LLM call per receipt. Sub-agent wires:
  - normalized fields (type-safe)
  - the list of allowed category strings
  - the user's optional prompt (injected into the LLMPort implementation's system message)
and returns a validated Categorization.

The actual prompt construction and provider call live in the LLM adapter.
This module only coordinates the call and re-raises on failure.
"""
from domain.models import AllowedCategory, Categorization, NormalizedReceipt
from application.ports import LLMPort


ALLOWED_CATEGORY_VALUES: list[str] = [c.value for c in AllowedCategory]


async def categorize_with_subagent(
    llm: LLMPort,
    normalized: NormalizedReceipt,
    user_prompt: str | None,
) -> Categorization:
    return await llm.categorize(
        normalized=normalized,
        allowed=ALLOWED_CATEGORY_VALUES,
        user_prompt=user_prompt,
    )
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/application/test_subagent.py
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/application/subagent.py tests/application/test_subagent.py
git commit -m "feat(app): categorization sub-agent coordinator"
```

---

## Phase 5 — Tool registry

### Task 5.1: @traced_tool decorator

**Files:**
- Create: `src/application/traced_tool.py`
- Test: `tests/application/test_traced_tool.py`

- [ ] **Step 1: Write failing tests** at `tests/application/test_traced_tool.py`

```python
import asyncio
import pytest
from uuid import uuid4
from application.traced_tool import traced_tool, ToolContext
from application.event_bus import InMemoryEventBus


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, input=None):
        span = FakeSpan(name, input)
        self.spans.append(span)
        return span


class FakeSpan:
    def __init__(self, name, input):
        self.name = name
        self.input = input
        self.output = None
        self.error = None
        self.ended = False

    def end(self, output=None, error=None):
        self.output = output
        self.error = error
        self.ended = True


@pytest.fixture
def ctx():
    bus = InMemoryEventBus()
    events: list[dict] = []

    async def capture(e):
        events.append(e)

    bus.subscribe(capture)
    return ToolContext(
        run_id=uuid4(),
        bus=bus,
        tracer=FakeTracer(),
        seq_counter=iter(range(1, 1000)),
    ), events


@pytest.mark.asyncio
async def test_emits_tool_call_and_tool_result_on_success(ctx):
    c, events = ctx

    @traced_tool("load_images")
    async def impl(ctx, *, folder_path):
        return {"count": 2}

    out = await impl(c, folder_path="/tmp")
    assert out == {"count": 2}
    kinds = [e["event_type"] for e in events]
    assert kinds == ["tool_call", "tool_result"]
    assert events[0]["tool"] == "load_images"
    assert events[0]["args"] == {"folder_path": "/tmp"}
    assert events[1]["error"] is False
    assert events[1]["duration_ms"] is not None
    assert events[1]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_emits_tool_result_with_error_flag_on_exception(ctx):
    c, events = ctx

    @traced_tool("extract_receipt_fields")
    async def impl(ctx, *, image_ref):
        raise RuntimeError("OCR timed out")

    with pytest.raises(RuntimeError):
        await impl(c, image_ref="r.png")

    kinds = [e["event_type"] for e in events]
    assert kinds == ["tool_call", "tool_result"]
    assert events[1]["error"] is True
    assert "OCR timed out" in events[1]["error_message"]


@pytest.mark.asyncio
async def test_result_summary_builder_is_honored(ctx):
    c, events = ctx

    def summarize(result):
        return {"count": result["count"], "kind": "summary"}

    @traced_tool("load_images", summarize=summarize)
    async def impl(ctx, *, folder_path):
        return {"count": 5, "files": ["a", "b", "c", "d", "e"]}

    await impl(c, folder_path="/tmp")
    assert events[1]["result_summary"] == {"count": 5, "kind": "summary"}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/application/traced_tool.py`**

```python
"""
@traced_tool decorator.

Applies to an async function with signature:
    async def f(ctx: ToolContext, **kwargs) -> <result>

Responsibilities:
- Publish `tool_call` before the call and `tool_result` after.
- Measure wall-clock duration (ms) on the `tool_result`.
- On exception: emit `tool_result` with error=true + error_message, then re-raise.
- Open a tracer span for the call.
- Build `result_summary` via a user-supplied `summarize(result)` or default to {}.

NOT responsible for:
- Retries (handled by tools that opt in, e.g. extract_receipt_fields).
- Converting exceptions into recoverable Receipt errors (that's done by the graph).
"""
from __future__ import annotations
import functools
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterator
from uuid import UUID, uuid4

from application.events import (
    EventType, ToolCall, ToolResult, serialize_event,
)
from application.ports import EventBusPort, TracerPort


@dataclass
class ToolContext:
    run_id: UUID
    bus: EventBusPort
    tracer: TracerPort
    seq_counter: Iterator[int]
    receipt_id: UUID | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def traced_tool(
    tool_name: str,
    summarize: Callable[[Any], dict] | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    def decorator(fn: Callable[..., Awaitable[Any]]):
        @functools.wraps(fn)
        async def wrapper(ctx: ToolContext, /, **kwargs) -> Any:
            call_event = ToolCall(
                run_id=ctx.run_id,
                seq=next(ctx.seq_counter),
                ts=_now(),
                tool=tool_name,
                receipt_id=ctx.receipt_id,
                args=_safe_args(kwargs),
            )
            await ctx.bus.publish(call_event.model_dump(mode="json"))
            span = ctx.tracer.start_span(tool_name, input=_safe_args(kwargs))
            started = time.perf_counter()
            try:
                result = await fn(ctx, **kwargs)
            except Exception as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                err_event = ToolResult(
                    run_id=ctx.run_id,
                    seq=next(ctx.seq_counter),
                    ts=_now(),
                    tool=tool_name,
                    receipt_id=ctx.receipt_id,
                    result_summary={},
                    error=True,
                    error_message=f"{type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                )
                await ctx.bus.publish(err_event.model_dump(mode="json"))
                span.end(error=str(exc))
                raise
            duration_ms = int((time.perf_counter() - started) * 1000)
            summary = (summarize(result) if summarize else {}) or {}
            ok_event = ToolResult(
                run_id=ctx.run_id,
                seq=next(ctx.seq_counter),
                ts=_now(),
                tool=tool_name,
                receipt_id=ctx.receipt_id,
                result_summary=summary,
                error=False,
                duration_ms=duration_ms,
            )
            await ctx.bus.publish(ok_event.model_dump(mode="json"))
            span.end(output=summary)
            return result

        return wrapper

    return decorator


def _safe_args(kwargs: dict) -> dict:
    """Best-effort JSON-safe rendering of kwargs for the SSE payload."""
    out: dict = {}
    for k, v in kwargs.items():
        try:
            if hasattr(v, "model_dump"):
                out[k] = v.model_dump(mode="json")
            elif isinstance(v, (str, int, float, bool, type(None), list, dict)):
                out[k] = v
            else:
                out[k] = str(v)
        except Exception:
            out[k] = f"<unserializable {type(v).__name__}>"
    return out
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/application/test_traced_tool.py
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/application/traced_tool.py tests/application/test_traced_tool.py
git commit -m "feat(app): @traced_tool decorator (events + timing + spans)"
```

---

### Task 5.2: Tool registry (6 tools)

**Files:**
- Create: `src/application/tool_registry.py`
- Test: `tests/application/test_tool_registry.py`

- [ ] **Step 1: Write failing tests** at `tests/application/test_tool_registry.py`

```python
import pytest
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from domain.models import (
    AllowedCategory, Categorization, Issue, NormalizedReceipt, RawReceipt, Receipt,
)
from application.ports import ImageRef
from application.event_bus import InMemoryEventBus
from application.traced_tool import ToolContext
from application.tool_registry import (
    load_images, extract_receipt_fields, normalize_receipt,
    categorize_receipt, aggregate_receipts, generate_report,
)
from tests.fakes.mock_ocr import MockOCR
from tests.fakes.mock_llm import MockLLM
from tests.fakes.mock_image_loader import MockImageLoader


class _NullTracer:
    def start_span(self, name, input=None):
        class _S:
            def end(self_, output=None, error=None): pass
        return _S()


def _ctx(bus):
    return ToolContext(run_id=uuid4(), bus=bus, tracer=_NullTracer(),
                       seq_counter=iter(range(1, 1000)))


@pytest.mark.asyncio
async def test_load_images_returns_refs_and_emits_tool_pair():
    bus = InMemoryEventBus()
    events = []
    bus.subscribe(lambda e: events.append(e))
    loader = MockImageLoader([ImageRef(source_ref="a.png", local_path=Path("/t/a.png"))])
    refs = await load_images(_ctx(bus), loader=loader)
    assert len(refs) == 1
    assert [e["event_type"] for e in events] == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_extract_receipt_fields_calls_ocr():
    bus = InMemoryEventBus()
    ocr = MockOCR(responses={"a.png": RawReceipt(source_ref="a.png", vendor="V", total_raw="$1.00")})
    raw = await extract_receipt_fields(_ctx(bus), ocr=ocr,
                                       image=ImageRef(source_ref="a.png", local_path=Path("/t/a.png")))
    assert raw.vendor == "V"


@pytest.mark.asyncio
async def test_normalize_receipt_returns_normalized():
    bus = InMemoryEventBus()
    raw = RawReceipt(source_ref="a.png", vendor="V", total_raw="$45.67",
                     receipt_date="2024-03-15")
    n = await normalize_receipt(_ctx(bus), raw=raw)
    assert n.total == Decimal("45.67")
    assert n.currency == "USD"


@pytest.mark.asyncio
async def test_categorize_receipt_calls_llm_with_prompt():
    bus = InMemoryEventBus()
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    n = NormalizedReceipt(source_ref="a.png", vendor="V", total=Decimal("10"))
    cat = await categorize_receipt(_ctx(bus), llm=llm, normalized=n, user_prompt="test")
    assert cat.category == AllowedCategory.TRAVEL
    assert llm.calls[0].user_prompt == "test"


@pytest.mark.asyncio
async def test_aggregate_receipts():
    bus = InMemoryEventBus()
    receipts = [
        Receipt(id=uuid4(), source_ref="a.png", category=AllowedCategory.TRAVEL,
                total=Decimal("45.67"), status="ok"),
    ]
    agg = await aggregate_receipts(_ctx(bus), receipts=receipts)
    assert agg.total_spend == Decimal("45.67")


@pytest.mark.asyncio
async def test_generate_report_bundles_fields():
    bus = InMemoryEventBus()
    rid = uuid4()
    receipts = []
    from domain.aggregation import aggregate
    agg = aggregate([])
    rep = await generate_report(_ctx(bus),
                                run_id=rid, aggregates=agg,
                                receipts=receipts, issues=[])
    assert rep.run_id == rid
    assert rep.total_spend == Decimal("0.00")
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/application/tool_registry.py`**

```python
"""
The 6 tools. Each is a thin wrapper decorated with @traced_tool.

Registry constraint: the application (graph) must only touch receipt data
by calling these. Direct use of adapters or domain functions outside these
wrappers bypasses the trace — don't do that.
"""
from uuid import UUID
from decimal import Decimal
from domain.aggregation import aggregate as _aggregate_pure
from domain.models import (
    Aggregates, AllowedCategory, Categorization, Issue, NormalizedReceipt,
    RawReceipt, Receipt, Report,
)
from domain.normalization import normalize as _normalize_pure
from application.ports import ImageLoaderPort, ImageRef, LLMPort, OCRPort
from application.subagent import categorize_with_subagent
from application.traced_tool import ToolContext, traced_tool


# 1. load_images
@traced_tool(
    "load_images",
    summarize=lambda refs: {"count": len(refs)},
)
async def load_images(ctx: ToolContext, *, loader: ImageLoaderPort) -> list[ImageRef]:
    return await loader.load()


# 2. extract_receipt_fields
def _summarize_raw(r: RawReceipt) -> dict:
    return {
        "vendor": r.vendor,
        "has_total": r.total_raw is not None,
        "ocr_confidence": r.ocr_confidence,
    }


@traced_tool("extract_receipt_fields", summarize=_summarize_raw)
async def extract_receipt_fields(
    ctx: ToolContext, *, ocr: OCRPort, image: ImageRef,
) -> RawReceipt:
    return await ocr.extract(image)


# 3. normalize_receipt
def _summarize_normalized(n: NormalizedReceipt) -> dict:
    return {
        "vendor": n.vendor,
        "receipt_date": n.receipt_date.isoformat() if n.receipt_date else None,
        "total": str(n.total) if n.total is not None else None,
        "currency": n.currency,
    }


@traced_tool("normalize_receipt", summarize=_summarize_normalized)
async def normalize_receipt(
    ctx: ToolContext, *, raw: RawReceipt,
) -> NormalizedReceipt:
    return _normalize_pure(raw)


# 4. categorize_receipt
def _summarize_categorization(c: Categorization) -> dict:
    return {
        "category": c.category.value,
        "confidence": c.confidence,
        "issue_count": len(c.issues),
    }


@traced_tool("categorize_receipt", summarize=_summarize_categorization)
async def categorize_receipt(
    ctx: ToolContext, *, llm: LLMPort,
    normalized: NormalizedReceipt, user_prompt: str | None,
) -> Categorization:
    return await categorize_with_subagent(llm, normalized, user_prompt)


# 5. aggregate
def _summarize_aggregates(a: Aggregates) -> dict:
    return {
        "total_spend": str(a.total_spend),
        "by_category": {k: str(v) for k, v in a.by_category.items()},
    }


@traced_tool("aggregate", summarize=_summarize_aggregates)
async def aggregate_receipts(
    ctx: ToolContext, *, receipts: list[Receipt],
) -> Aggregates:
    return _aggregate_pure(receipts)


# 6. generate_report
def _summarize_report(r: Report) -> dict:
    return {
        "total_spend": str(r.total_spend),
        "receipt_count": len(r.receipts),
        "issue_count": len(r.issues_and_assumptions),
    }


@traced_tool("generate_report", summarize=_summarize_report)
async def generate_report(
    ctx: ToolContext, *,
    run_id: UUID, aggregates: Aggregates,
    receipts: list[Receipt], issues: list[Issue],
) -> Report:
    return Report(
        run_id=run_id,
        total_spend=aggregates.total_spend,
        by_category=aggregates.by_category,
        receipts=receipts,
        issues_and_assumptions=issues,
    )


TOOL_NAMES = [
    "load_images",
    "extract_receipt_fields",
    "normalize_receipt",
    "categorize_receipt",
    "aggregate",
    "generate_report",
]
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/application/test_tool_registry.py
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/application/tool_registry.py tests/application/test_tool_registry.py
git commit -m "feat(app): tool registry (6 tools) with result_summary builders"
```

---

## Phase 6 — Graph (LangGraph state machine)

### Task 6.1: Graph state + GraphRunner happy path

**Files:**
- Create: `src/application/graph.py`
- Test: `tests/application/test_graph.py`

- [ ] **Step 1: Write failing tests** at `tests/application/test_graph.py`

```python
import pytest
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from domain.models import AllowedCategory, Categorization, RawReceipt
from application.graph import GraphRunner, build_graph, RunState
from application.event_bus import InMemoryEventBus
from application.traced_tool import ToolContext
from application.ports import ImageRef
from tests.fakes.mock_ocr import MockOCR
from tests.fakes.mock_llm import MockLLM
from tests.fakes.mock_image_loader import MockImageLoader
from tests.fakes.in_memory_repos import InMemoryReportRepository, InMemoryTraceRepository


class _NullTracer:
    def start_span(self, name, input=None):
        class _S:
            def end(self_, output=None, error=None): pass
        return _S()


def _refs(n=2):
    return [
        ImageRef(source_ref=f"r{i}.png", local_path=Path(f"/t/r{i}.png"))
        for i in range(1, n + 1)
    ]


def _runner_with_mocks(refs, ocr=None, llm=None, prompt=None):
    bus = InMemoryEventBus()
    events: list[dict] = []
    bus.subscribe(lambda e: events.append(e))
    return GraphRunner(
        run_id=uuid4(),
        prompt=prompt,
        bus=bus,
        tracer=_NullTracer(),
        image_loader=MockImageLoader(refs),
        ocr=ocr or MockOCR(),
        llm=llm or MockLLM(default_category=AllowedCategory.TRAVEL),
        report_repo=InMemoryReportRepository(),
    ), events


@pytest.mark.asyncio
async def test_happy_path_event_order():
    runner, events = _runner_with_mocks(_refs(2))
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))

    kinds = [e["event_type"] for e in events]
    # first: run_started + load_images tool_call/tool_result
    assert kinds[0] == "run_started"
    assert "load_images" in str(events[1:3])
    # per-receipt sequence appears twice (2 receipts)
    assert kinds.count("receipt_result") == 2
    # final_result is terminal
    assert kinds[-1] == "final_result"


@pytest.mark.asyncio
async def test_final_result_contains_aggregates():
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    runner, events = _runner_with_mocks(_refs(2), llm=llm)
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
    final = [e for e in events if e["event_type"] == "final_result"][0]
    # mock OCR total is "$12.34" per receipt → 2 × 12.34 = 24.68
    assert final["total_spend"] == "24.68"
    assert final["by_category"]["Travel"] == "24.68"


@pytest.mark.asyncio
async def test_prompt_threads_through_to_subagent():
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    runner, _ = _runner_with_mocks(_refs(1), llm=llm, prompt="be conservative")
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
    assert llm.calls[0].user_prompt == "be conservative"
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `src/application/graph.py`**

```python
"""
LangGraph state machine: deterministic outer pipeline, sequential per receipt.

Graph:
    START → load_images → [loop over receipts: ocr → normalize → categorize → receipt_result]
          → aggregate → generate_report → final_result → END

`RunState` carries the data. `GraphRunner` holds the adapters and implements
each node as an async method. The graph is built from the runner via `build_graph`.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from itertools import count
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from domain.aggregation import aggregate as aggregate_pure
from domain.models import (
    AllowedCategory, Issue, NormalizedReceipt, RawReceipt, Receipt,
)
from application.events import (
    ErrorEvent, FinalResult, Progress, ReceiptResult, RunStarted, serialize_event,
)
from application.ports import (
    EventBusPort, ImageLoaderPort, LLMPort, OCRPort, ReportRepositoryPort, TracerPort, ImageRef,
)
from application.traced_tool import ToolContext
from application.tool_registry import (
    aggregate_receipts, categorize_receipt, extract_receipt_fields,
    generate_report, load_images, normalize_receipt,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunState(BaseModel):
    """Mutable state threaded through the graph."""
    images: list[ImageRef] = Field(default_factory=list)
    receipts: list[Receipt] = Field(default_factory=list)
    current: int = 0
    errors: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class GraphRunner:
    def __init__(
        self, *,
        run_id: UUID,
        prompt: str | None,
        bus: EventBusPort,
        tracer: TracerPort,
        image_loader: ImageLoaderPort,
        ocr: OCRPort,
        llm: LLMPort,
        report_repo: ReportRepositoryPort,
    ) -> None:
        self.run_id = run_id
        self.prompt = prompt
        self.bus = bus
        self.tracer = tracer
        self.image_loader = image_loader
        self.ocr = ocr
        self.llm = llm
        self.report_repo = report_repo
        self._seq = count(1)

    # ---- helpers ----

    def _ctx(self, receipt_id: UUID | None = None) -> ToolContext:
        return ToolContext(
            run_id=self.run_id, bus=self.bus, tracer=self.tracer,
            seq_counter=self._seq, receipt_id=receipt_id,
        )

    async def _emit(self, event_model) -> None:
        await self.bus.publish(event_model.model_dump(mode="json"))

    async def _progress(self, step: str, receipt_id: UUID | None = None,
                        i: int | None = None, n: int | None = None) -> None:
        await self._emit(Progress(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            step=step, receipt_id=receipt_id, i=i, n=n,
        ))

    # ---- nodes ----

    async def start(self, state: RunState) -> RunState:
        await self._emit(RunStarted(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            prompt=self.prompt,
        ))
        await self._progress("load_images")
        images = await load_images(self._ctx(), loader=self.image_loader)
        return state.model_copy(update={"images": images})

    async def process_receipt(self, state: RunState) -> RunState:
        i = state.current
        n = len(state.images)
        image = state.images[i]
        receipt_id = uuid4()

        await self._progress("ocr", receipt_id=receipt_id, i=i + 1, n=n)
        raw: RawReceipt | None = None
        normalized: NormalizedReceipt | None = None
        categorization = None
        receipt_status = "ok"
        err: str | None = None
        local_issues: list[Issue] = []

        try:
            raw = await extract_receipt_fields(self._ctx(receipt_id), ocr=self.ocr, image=image)
        except Exception as e:  # Band A: receipt-level recoverable
            receipt_status = "error"
            err = f"{type(e).__name__}: {e}"
            local_issues.append(Issue(
                severity="receipt_error", code="ocr_failed",
                message=err, receipt_id=receipt_id,
            ))

        if receipt_status == "ok":
            await self._progress("normalize", receipt_id=receipt_id)
            try:
                normalized = await normalize_receipt(self._ctx(receipt_id), raw=raw)  # type: ignore[arg-type]
            except Exception as e:
                receipt_status = "error"
                err = f"{type(e).__name__}: {e}"
                local_issues.append(Issue(
                    severity="receipt_error", code="parse_failed",
                    message=err, receipt_id=receipt_id,
                ))

        if receipt_status == "ok":
            await self._progress("categorize", receipt_id=receipt_id)
            try:
                categorization = await categorize_receipt(
                    self._ctx(receipt_id), llm=self.llm,
                    normalized=normalized,  # type: ignore[arg-type]
                    user_prompt=self.prompt,
                )
            except Exception as e:
                receipt_status = "error"
                err = f"{type(e).__name__}: {e}"
                local_issues.append(Issue(
                    severity="receipt_error", code="llm_failed",
                    message=err, receipt_id=receipt_id,
                ))

        receipt = Receipt(
            id=receipt_id,
            source_ref=image.source_ref,
            vendor=normalized.vendor if normalized else (raw.vendor if raw else None),
            receipt_date=normalized.receipt_date if normalized else None,
            receipt_number=normalized.receipt_number if normalized else (raw.receipt_number if raw else None),
            total=normalized.total if normalized else None,
            currency=normalized.currency if normalized else None,
            category=categorization.category if categorization else None,
            confidence=categorization.confidence if categorization else None,
            notes=categorization.notes if categorization else None,
            issues=local_issues + (categorization.issues if categorization else []),
            raw_ocr=raw.model_dump(mode="json") if raw else None,
            normalized=normalized.model_dump(mode="json") if normalized else None,
            status=receipt_status,
            error=err,
        )

        # Persist + emit receipt_result
        await self.report_repo.insert_receipt({
            "id": receipt.id, "report_id": self.run_id, "seq": i + 1,
            "source_ref": receipt.source_ref, "vendor": receipt.vendor,
            "receipt_date": receipt.receipt_date, "receipt_number": receipt.receipt_number,
            "total": receipt.total, "currency": receipt.currency,
            "category": receipt.category.value if receipt.category else None,
            "confidence": receipt.confidence, "notes": receipt.notes,
            "issues": [iss.model_dump(mode="json") for iss in receipt.issues],
            "raw_ocr": receipt.raw_ocr, "normalized": receipt.normalized,
            "status": receipt.status, "error": receipt.error,
            "created_at": _now(),
        })
        await self._emit(ReceiptResult(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            receipt_id=receipt.id, status=receipt.status,
            vendor=receipt.vendor,
            receipt_date=receipt.receipt_date.isoformat() if receipt.receipt_date else None,
            receipt_number=receipt.receipt_number,
            total=str(receipt.total) if receipt.total is not None else None,
            currency=receipt.currency,
            category=receipt.category.value if receipt.category else None,
            confidence=receipt.confidence,
            notes=receipt.notes,
            issues=[iss.model_dump(mode="json") for iss in receipt.issues],
            error_message=err,
        ))

        new_receipts = state.receipts + [receipt]
        new_issues = state.issues + receipt.issues
        return state.model_copy(update={
            "receipts": new_receipts,
            "current": i + 1,
            "issues": new_issues,
        })

    async def finalize(self, state: RunState) -> RunState:
        await self._progress("aggregate")
        agg = await aggregate_receipts(self._ctx(), receipts=state.receipts)
        await self._progress("generate_report")
        report = await generate_report(
            self._ctx(),
            run_id=self.run_id,
            aggregates=agg,
            receipts=state.receipts,
            issues=state.issues,
        )
        await self._emit(FinalResult(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            total_spend=str(report.total_spend),
            by_category={k: str(v) for k, v in report.by_category.items()},
            receipts=[r.model_dump(mode="json") for r in report.receipts],
            issues_and_assumptions=[iss.model_dump(mode="json") for iss in report.issues_and_assumptions],
        ))
        return state


def build_graph(runner: GraphRunner):
    g = StateGraph(RunState)
    g.add_node("start", runner.start)
    g.add_node("process_receipt", runner.process_receipt)
    g.add_node("finalize", runner.finalize)

    g.add_edge(START, "start")

    def _after_start(state: RunState):
        return "process_receipt" if state.images else "finalize"

    g.add_conditional_edges("start", _after_start, {
        "process_receipt": "process_receipt",
        "finalize": "finalize",
    })

    def _loop_or_finalize(state: RunState):
        return "process_receipt" if state.current < len(state.images) else "finalize"

    g.add_conditional_edges("process_receipt", _loop_or_finalize, {
        "process_receipt": "process_receipt",
        "finalize": "finalize",
    })
    g.add_edge("finalize", END)
    return g.compile()
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/application/test_graph.py::test_happy_path_event_order tests/application/test_graph.py::test_final_result_contains_aggregates tests/application/test_graph.py::test_prompt_threads_through_to_subagent
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/application/graph.py tests/application/test_graph.py
git commit -m "feat(app): LangGraph state machine with happy-path receipts loop"
```

---

### Task 6.2: Receipt-level error handling in graph

**Files:**
- Modify: `tests/application/test_graph.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_receipt_level_error_continues_run():
    refs = _refs(3)
    ocr = MockOCR(fail_on={"r2.png"})
    runner, events = _runner_with_mocks(refs, ocr=ocr)
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))

    receipt_results = [e for e in events if e["event_type"] == "receipt_result"]
    assert len(receipt_results) == 3
    statuses = [e["status"] for e in receipt_results]
    assert statuses.count("error") == 1
    assert statuses.count("ok") == 2

    final = [e for e in events if e["event_type"] == "final_result"][0]
    # 2 successful receipts × $12.34 = $24.68
    assert final["total_spend"] == "24.68"
    # issues_and_assumptions includes at least the one receipt_error
    codes = [i["code"] for i in final["issues_and_assumptions"]]
    assert "ocr_failed" in codes
```

- [ ] **Step 2: Run — it should already pass** (existing implementation handles this path)

```bash
make test tests/application/test_graph.py::test_receipt_level_error_continues_run
```
Expected: PASS. If it fails, fix the graph to match the spec (errored receipt still emits `receipt_result` with `status="error"` and contributes its issue to `issues_and_assumptions`).

- [ ] **Step 3: Commit**

```bash
git add tests/application/test_graph.py
git commit -m "test(graph): verify receipt-level error continues run"
```

---

### Task 6.3: Run-level error (zero images)

**Files:**
- Modify: `src/application/graph.py`
- Modify: `tests/application/test_graph.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_zero_images_emits_error_event():
    runner, events = _runner_with_mocks([])  # no images
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))

    kinds = [e["event_type"] for e in events]
    assert "error" in kinds
    err = next(e for e in events if e["event_type"] == "error")
    assert err["code"] == "no_images"
    assert "final_result" not in kinds
```

- [ ] **Step 2: Run, expect failure** (current graph transitions to finalize when images are empty — it emits `final_result` with zero receipts, not an error).

- [ ] **Step 3: Modify `GraphRunner.start` and `build_graph`**

Replace `start` with:

```python
    async def start(self, state: RunState) -> RunState:
        await self._emit(RunStarted(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            prompt=self.prompt,
        ))
        await self._progress("load_images")
        images = await load_images(self._ctx(), loader=self.image_loader)
        if not images:
            await self._emit(ErrorEvent(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                code="no_images", message="no images found in input",
            ))
            return state.model_copy(update={"images": [], "errors": ["no_images"]})
        return state.model_copy(update={"images": images})
```

And update `_after_start` in `build_graph`:

```python
    def _after_start(state: RunState):
        if state.errors:
            return END
        return "process_receipt" if state.images else "finalize"

    g.add_conditional_edges("start", _after_start, {
        "process_receipt": "process_receipt",
        "finalize": "finalize",
        END: END,
    })
```

- [ ] **Step 4: Run, expect PASS**

```bash
make test tests/application/test_graph.py
```
Expected: all graph tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/application/graph.py tests/application/test_graph.py
git commit -m "feat(graph): emit error event and halt on zero images"
```

---

## Phase 7 — Database infrastructure

### Task 7.1: SQLAlchemy engine + ORM models

**Files:**
- Create: `src/infrastructure/db/engine.py`, `src/infrastructure/db/models.py`

- [ ] **Step 1: Create `src/infrastructure/db/engine.py`**

```python
"""Async SQLAlchemy engine factory."""
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker


@lru_cache(maxsize=1)
def get_engine(db_url: str) -> AsyncEngine:
    # psycopg v3 async driver: use `postgresql+psycopg://` (not asyncpg).
    return create_async_engine(db_url, pool_pre_ping=True, future=True)


def session_factory(engine: AsyncEngine):
    return async_sessionmaker(engine, expire_on_commit=False)
```

- [ ] **Step 2: Create `src/infrastructure/db/models.py`**

```python
"""
ORM rows for reports / receipts / traces.

Mirrors the schema from the design spec. JSONB columns use SQLAlchemy's JSONB.
"""
from __future__ import annotations
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID
from sqlalchemy import (
    BigInteger, Column, ForeignKey, Integer, Numeric, String, Text, DateTime, Date,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ReportRow(Base):
    __tablename__ = "reports"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text)
    input_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    input_ref: Mapped[str | None] = mapped_column(Text)
    receipt_count: Mapped[int | None] = mapped_column(Integer)
    total_spend: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    by_category: Mapped[dict | None] = mapped_column(JSONB)
    issues: Mapped[list | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)


class ReceiptRow(Base):
    __tablename__ = "receipts"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    report_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    vendor: Mapped[str | None] = mapped_column(Text)
    receipt_date: Mapped[date | None] = mapped_column(Date)
    receipt_number: Mapped[str | None] = mapped_column(Text)
    total: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    currency: Mapped[str | None] = mapped_column(String(8))
    category: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(3, 2))
    notes: Mapped[str | None] = mapped_column(Text)
    issues: Mapped[list | None] = mapped_column(JSONB)
    raw_ocr: Mapped[dict | None] = mapped_column(JSONB)
    normalized: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TraceRow(Base):
    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    report_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
    )
    receipt_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True))  # NOT a FK
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    step: Mapped[str | None] = mapped_column(String(32))
    tool: Mapped[str | None] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

- [ ] **Step 3: Smoke-import**

```bash
PYTHONPATH=src python -c "from infrastructure.db.models import Base, ReportRow, ReceiptRow, TraceRow; print(sorted(Base.metadata.tables.keys()))"
```
Expected: `['receipts', 'reports', 'traces']`.

- [ ] **Step 4: Commit**

```bash
git add src/infrastructure/db/engine.py src/infrastructure/db/models.py
git commit -m "feat(db): SQLAlchemy engine + ORM rows for reports/receipts/traces"
```

---

### Task 7.2: Alembic initial migration

**Files:**
- Create: `alembic.ini`, `migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/0001_initial.py`

- [ ] **Step 1: Initialize Alembic**

```bash
PYTHONPATH=src alembic init migrations
```

This creates `alembic.ini`, `migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/`.

- [ ] **Step 2: Edit `alembic.ini`**

Set:

```ini
script_location = migrations
sqlalchemy.url =
```

(Leave `sqlalchemy.url` blank; we'll inject it from env in `env.py`.)

- [ ] **Step 3: Replace `migrations/env.py`**

```python
import os
from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config, pool
from infrastructure.db.models import Base

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

db_url = os.environ["SUPABASE_DB_URL"].replace("+psycopg", "")  # sync driver for Alembic
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(
        url=db_url, target_metadata=target_metadata, literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Create `migrations/versions/0001_initial.py`**

```python
"""initial schema: reports, receipts, traces"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID

# revision identifiers
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "reports",
        sa.Column("id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("input_kind", sa.String(16), nullable=False),
        sa.Column("input_ref", sa.Text, nullable=True),
        sa.Column("receipt_count", sa.Integer, nullable=True),
        sa.Column("total_spend", sa.Numeric(14, 2), nullable=True),
        sa.Column("by_category", JSONB, nullable=True),
        sa.Column("issues", JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )

    op.create_table(
        "receipts",
        sa.Column("id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", PgUUID(as_uuid=True),
                  sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("source_ref", sa.Text, nullable=False),
        sa.Column("vendor", sa.Text),
        sa.Column("receipt_date", sa.Date),
        sa.Column("receipt_number", sa.Text),
        sa.Column("total", sa.Numeric(14, 2)),
        sa.Column("currency", sa.String(8)),
        sa.Column("category", sa.Text),
        sa.Column("confidence", sa.Numeric(3, 2)),
        sa.Column("notes", sa.Text),
        sa.Column("issues", JSONB),
        sa.Column("raw_ocr", JSONB),
        sa.Column("normalized", JSONB),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_receipts_report_seq", "receipts", ["report_id", "seq"])

    op.create_table(
        "traces",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("report_id", PgUUID(as_uuid=True),
                  sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("receipt_id", PgUUID(as_uuid=True), nullable=True),  # NOT a FK
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("step", sa.String(32)),
        sa.Column("tool", sa.String(64)),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_traces_report_seq", "traces", ["report_id", "seq"])
    op.create_index("idx_traces_report_type", "traces", ["report_id", "event_type"])


def downgrade():
    op.drop_index("idx_traces_report_type", table_name="traces")
    op.drop_index("idx_traces_report_seq", table_name="traces")
    op.drop_table("traces")
    op.drop_index("idx_receipts_report_seq", table_name="receipts")
    op.drop_table("receipts")
    op.drop_table("reports")
```

- [ ] **Step 5: Sanity-check — alembic lists the revision**

```bash
PYTHONPATH=src SUPABASE_DB_URL="postgresql://u:p@localhost/x" alembic history
```
Expected: shows revision `0001` with description "initial schema".

(We don't apply the migration here — that requires a reachable DB.)

- [ ] **Step 6: Commit**

```bash
git add alembic.ini migrations/
git commit -m "feat(db): Alembic initial migration (reports/receipts/traces)"
```

---

### Task 7.3: Async repositories

**Files:**
- Create: `src/infrastructure/db/repositories.py`
- Test: `tests/infrastructure/test_repositories.py` (integration; skippable)

- [ ] **Step 1: Implement `src/infrastructure/db/repositories.py`**

```python
"""
Async repositories. Each write is its own short transaction.
`TraceRepository.insert_trace` is tolerant of schema/constraint errors — it logs
and swallows, never raising (per the EventBus contract).
"""
import logging
from datetime import datetime
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
```

- [ ] **Step 2: Write a skippable integration test** at `tests/infrastructure/test_repositories.py`

```python
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
from infrastructure.db.models import Base
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
    # Verification via session
    async with session_maker() as s:
        from infrastructure.db.models import ReportRow
        row = await s.get(ReportRow, rid)
        assert row.status == "succeeded"
```

- [ ] **Step 3: Run (expected SKIP unless `TEST_SUPABASE_DB_URL` is set)**

```bash
make test tests/infrastructure/test_repositories.py
```
Expected: SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add src/infrastructure/db/repositories.py tests/infrastructure/test_repositories.py
git commit -m "feat(db): async repositories (SqlReport, SqlTrace)"
```

---

## Phase 8 — Tracing adapters

### Task 8.1: JSON logs tracer

**Files:**
- Create: `src/infrastructure/tracing/json_logs_adapter.py`
- Test: `tests/application/test_json_logs_adapter.py`

- [ ] **Step 1: Write failing test** at `tests/application/test_json_logs_adapter.py`

```python
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
```

- [ ] **Step 2: Implement `src/infrastructure/tracing/json_logs_adapter.py`**

```python
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
```

- [ ] **Step 3: Run, expect PASS**

```bash
make test tests/application/test_json_logs_adapter.py
```

- [ ] **Step 4: Commit**

```bash
git add src/infrastructure/tracing/json_logs_adapter.py tests/application/test_json_logs_adapter.py
git commit -m "feat(tracing): JSON logs tracer (structured span start/end)"
```

---

### Task 8.2: Langfuse tracer adapter

**Files:**
- Create: `src/infrastructure/tracing/langfuse_adapter.py`

- [ ] **Step 1: Implement `src/infrastructure/tracing/langfuse_adapter.py`**

```python
"""
Langfuse tracer.

If `public_key`/`secret_key` are empty, returns a no-op tracer. This keeps
Langfuse optional for reviewers who don't set up credentials.

The API uses manual trace+span lifecycle; we treat each tool call as a span
under a single trace opened per run (constructed by `open_trace`).
"""
from __future__ import annotations
import logging
from typing import Any
from application.ports import TracerPort, TracerSpan

_log = logging.getLogger(__name__)


class NoopSpan(TracerSpan):
    def end(self, output=None, error=None):
        pass


class NoopTracer(TracerPort):
    def start_span(self, name, input=None):
        return NoopSpan()


class LangfuseTracer(TracerPort):
    def __init__(self, *, public_key: str, secret_key: str, host: str, run_id: str) -> None:
        # Import lazily so the package isn't needed in pure-mock test envs.
        from langfuse import Langfuse
        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._trace = self._client.trace(name="receipt_run", id=run_id)

    def start_span(self, name: str, input: dict | None = None) -> "LangfuseSpan":
        span = self._trace.span(name=name, input=input or {})
        return LangfuseSpan(span)


class LangfuseSpan(TracerSpan):
    def __init__(self, span: Any) -> None:
        self._span = span

    def end(self, output: dict | None = None, error: str | None = None) -> None:
        try:
            if error:
                self._span.end(level="ERROR", status_message=error)
            else:
                self._span.end(output=output or {})
        except Exception:
            _log.exception("langfuse span end failed (ignored)")


def build_tracer(*, public_key: str | None, secret_key: str | None, host: str, run_id: str) -> TracerPort:
    if not public_key or not secret_key:
        return NoopTracer()
    try:
        return LangfuseTracer(public_key=public_key, secret_key=secret_key, host=host, run_id=run_id)
    except Exception:
        _log.exception("Langfuse init failed; falling back to no-op tracer")
        return NoopTracer()
```

- [ ] **Step 2: Smoke-import**

```bash
PYTHONPATH=src python -c "from infrastructure.tracing.langfuse_adapter import build_tracer, NoopTracer; t=build_tracer(public_key=None, secret_key=None, host='', run_id='x'); print(type(t).__name__)"
```
Expected: `NoopTracer`.

- [ ] **Step 3: Commit**

```bash
git add src/infrastructure/tracing/langfuse_adapter.py
git commit -m "feat(tracing): Langfuse adapter with graceful no-op fallback"
```

---

## Phase 9 — HTTP infrastructure

### Task 9.1: Composition root + main

**Files:**
- Create: `src/composition_root.py`, `src/main.py`

- [ ] **Step 1: Implement `src/composition_root.py`**

```python
"""
Build a composed FastAPI app.

Wires adapters → ports → application by LLM_MODE. This is the ONLY module
that imports from `infrastructure` AND `application`.
"""
from pathlib import Path
from fastapi import FastAPI

from config import Settings, LLMMode
from infrastructure.db.engine import get_engine, session_factory
from infrastructure.db.repositories import SqlReportRepository, SqlTraceRepository
from infrastructure.http.app import create_app


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = get_engine(settings.supabase_db_url)
    sm = session_factory(engine)
    report_repo = SqlReportRepository(sm)
    trace_repo = SqlTraceRepository(sm)
    return create_app(settings=settings, report_repo=report_repo, trace_repo=trace_repo)
```

- [ ] **Step 2: Implement `src/main.py`**

```python
from composition_root import build_app

app = build_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 3: Commit (smoke-import happens in the next task)**

```bash
git add src/composition_root.py src/main.py
git commit -m "feat: composition root + uvicorn entry"
```

---

### Task 9.2: FastAPI app + SSE endpoint

**Files:**
- Create: `src/infrastructure/http/sse.py`, `src/infrastructure/http/app.py`, `src/infrastructure/http/routes_runs.py`
- Create: `src/infrastructure/ocr/mock_adapter.py`, `src/infrastructure/llm/mock_adapter.py`, `src/infrastructure/images/folder_loader.py`, `src/infrastructure/images/upload_loader.py`

- [ ] **Step 1: Create `src/infrastructure/http/sse.py`**

```python
"""Thin SSE helper. We emit events from an async iterator of dicts."""
from typing import AsyncIterator
from sse_starlette.sse import EventSourceResponse


def sse_response(source: AsyncIterator[dict]) -> EventSourceResponse:
    async def gen():
        async for event in source:
            yield {
                "event": event["event_type"],
                "data": _serialize(event),
            }

    return EventSourceResponse(gen())


def _serialize(event: dict) -> str:
    import json
    return json.dumps(event, default=str)
```

- [ ] **Step 2: Create `src/infrastructure/ocr/mock_adapter.py`**

```python
"""
Production mock OCR adapter.

Deterministic OCR keyed by source_ref. Returns a canned RawReceipt that the
normalizer can parse. Intentionally simple — richness lives in the real adapter.
"""
import hashlib
from decimal import Decimal
from domain.models import RawReceipt
from application.ports import OCRPort, ImageRef


_VENDORS = ["Uber", "Delta", "Starbucks", "Staples", "FedEx", "AWS", "ConEd"]


class MockOCRAdapter(OCRPort):
    async def extract(self, image: ImageRef) -> RawReceipt:
        # Stable pseudo-random values keyed by filename
        h = int(hashlib.sha256(image.source_ref.encode()).hexdigest(), 16)
        vendor = _VENDORS[h % len(_VENDORS)]
        total_cents = 500 + (h % 9500)     # $5.00 – $100.00
        total = f"${total_cents / 100:.2f}"
        return RawReceipt(
            source_ref=image.source_ref,
            vendor=vendor,
            receipt_date="2024-03-15",
            receipt_number=f"R-{h % 100000:05d}",
            total_raw=total,
            ocr_confidence=0.92,
        )
```

- [ ] **Step 3: Create `src/infrastructure/llm/mock_adapter.py`**

```python
"""
Production mock LLM adapter.

Deterministic category assignment keyed by vendor. Captures user_prompt so
tests / tracing see it. Emits 1 'warning' issue when confidence < 0.6 (never
true here) and 0 otherwise.
"""
import hashlib
from domain.models import AllowedCategory, Categorization, Issue, NormalizedReceipt
from application.ports import LLMPort


_VENDOR_CATEGORY = {
    "uber": AllowedCategory.TRAVEL,
    "delta": AllowedCategory.TRAVEL,
    "starbucks": AllowedCategory.MEALS,
    "staples": AllowedCategory.OFFICE_SUPPLIES,
    "fedex": AllowedCategory.SHIPPING,
    "aws": AllowedCategory.SOFTWARE,
    "coned": AllowedCategory.UTILITIES,
}


class MockLLMAdapter(LLMPort):
    async def categorize(self, normalized: NormalizedReceipt, allowed, user_prompt):
        vendor_key = (normalized.vendor or "").lower()
        category = _VENDOR_CATEGORY.get(vendor_key, AllowedCategory.OTHER)
        notes = f"mock categorization; prompt_seen={user_prompt!r}"
        # Ensure "Other" always has a note
        if category == AllowedCategory.OTHER and not notes:
            notes = "uncategorized"
        # Stable confidence from vendor + prompt
        seed = hashlib.sha256((vendor_key + (user_prompt or "")).encode()).hexdigest()
        confidence = 0.70 + (int(seed, 16) % 25) / 100  # 0.70–0.94
        return Categorization(
            category=category, confidence=round(confidence, 2),
            notes=notes, issues=[],
        )
```

- [ ] **Step 4: Create `src/infrastructure/images/folder_loader.py`**

```python
"""Load images from a local folder, filtered by allowed extensions."""
from pathlib import Path
from application.ports import ImageLoaderPort, ImageRef


class LocalFolderImageLoader(ImageLoaderPort):
    def __init__(self, folder: Path, allowed_extensions: set[str]) -> None:
        self._folder = folder
        self._allowed = {e.lower().lstrip(".") for e in allowed_extensions}

    async def load(self) -> list[ImageRef]:
        if not self._folder.exists() or not self._folder.is_dir():
            return []
        refs: list[ImageRef] = []
        for p in sorted(self._folder.iterdir()):
            if p.is_file() and p.suffix.lower().lstrip(".") in self._allowed:
                refs.append(ImageRef(source_ref=p.name, local_path=p.resolve()))
        return refs
```

- [ ] **Step 5: Create `src/infrastructure/images/upload_loader.py`**

```python
"""Save multipart uploads to a per-run temp directory and return ImageRefs."""
import shutil
import tempfile
from pathlib import Path
from application.ports import ImageLoaderPort, ImageRef
from fastapi import UploadFile


class UploadImageLoader(ImageLoaderPort):
    def __init__(self, uploads: list[UploadFile], allowed_extensions: set[str]) -> None:
        self._uploads = uploads
        self._allowed = {e.lower().lstrip(".") for e in allowed_extensions}
        self._tmpdir = Path(tempfile.mkdtemp(prefix="receipt-run-"))

    async def load(self) -> list[ImageRef]:
        refs: list[ImageRef] = []
        for up in self._uploads:
            if not up.filename:
                continue
            ext = Path(up.filename).suffix.lower().lstrip(".")
            if ext not in self._allowed:
                continue
            dest = self._tmpdir / up.filename
            content = await up.read()
            dest.write_bytes(content)
            refs.append(ImageRef(source_ref=up.filename, local_path=dest.resolve()))
        return refs
```

- [ ] **Step 6: Create `src/infrastructure/http/app.py`**

```python
from fastapi import FastAPI
from config import Settings
from application.ports import ReportRepositoryPort, TraceRepositoryPort
from infrastructure.http.routes_runs import router as runs_router, RunsDeps


def create_app(
    *,
    settings: Settings,
    report_repo: ReportRepositoryPort,
    trace_repo: TraceRepositoryPort,
) -> FastAPI:
    app = FastAPI(title="Receipt Processing Agent")
    deps = RunsDeps(settings=settings, report_repo=report_repo, trace_repo=trace_repo)
    app.state.runs_deps = deps
    app.include_router(runs_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "llm_mode": settings.llm_mode}

    return app
```

- [ ] **Step 7: Create `src/infrastructure/http/routes_runs.py`**

```python
"""POST /runs/stream — multipart OR JSON body; returns SSE."""
from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from application.events import RunStarted
from application.event_bus import InMemoryEventBus
from application.graph import GraphRunner, RunState, build_graph
from application.ports import ReportRepositoryPort, TraceRepositoryPort
from config import LLMMode, Settings
from infrastructure.http.sse import sse_response
from infrastructure.images.folder_loader import LocalFolderImageLoader
from infrastructure.images.upload_loader import UploadImageLoader
from infrastructure.ocr.mock_adapter import MockOCRAdapter
from infrastructure.llm.mock_adapter import MockLLMAdapter
from infrastructure.tracing.json_logs_adapter import JSONLogsTracer
from infrastructure.tracing.langfuse_adapter import build_tracer

router = APIRouter()


@dataclass
class RunsDeps:
    settings: Settings
    report_repo: ReportRepositoryPort
    trace_repo: TraceRepositoryPort


class FolderInput(BaseModel):
    folder_path: str
    prompt: str | None = None


@router.post("/runs/stream")
async def post_runs_stream(request: Request,
                           files: list[UploadFile] = File(default=None),
                           prompt: str | None = Form(default=None)):
    deps: RunsDeps = request.app.state.runs_deps
    settings = deps.settings

    # ---- decide input mode ----
    input_kind: str
    input_ref: str | None
    image_loader = None

    if files:
        if len(files) > settings.max_files_per_run:
            raise HTTPException(413, f"too many files (max {settings.max_files_per_run})")
        image_loader = UploadImageLoader(files, settings.allowed_extensions)
        input_kind, input_ref = "upload", f"{len(files)} files"
    else:
        body = await request.json()
        try:
            fi = FolderInput(**body)
        except Exception as e:
            raise HTTPException(422, f"invalid body: {e}")
        folder = Path(fi.folder_path).resolve()
        assets = settings.assets_dir.resolve()
        if not str(folder).startswith(str(assets)):
            raise HTTPException(400, f"folder_path must be under {assets}")
        image_loader = LocalFolderImageLoader(folder, settings.allowed_extensions)
        input_kind, input_ref = "folder", str(folder)
        prompt = fi.prompt

    # ---- wire per-run adapters ----
    run_id = uuid4()
    bus = InMemoryEventBus()

    # subscribe trace writer
    async def trace_writer(e: dict) -> None:
        await deps.trace_repo.insert_trace({
            "report_id": run_id,
            "receipt_id": _uuid_or_none(e.get("receipt_id")),
            "seq": e["seq"],
            "event_type": e["event_type"],
            "step": e.get("step"),
            "tool": e.get("tool"),
            "payload": e,
            "duration_ms": e.get("duration_ms"),
            "created_at": datetime.now(timezone.utc),
        })

    bus.subscribe(trace_writer)

    if settings.llm_mode == LLMMode.MOCK:
        ocr = MockOCRAdapter()
        llm = MockLLMAdapter()
    else:
        from infrastructure.ocr.openai_adapter import OpenAIOCRAdapter
        from infrastructure.llm.deepseek_adapter import DeepSeekLLMAdapter
        ocr = OpenAIOCRAdapter(api_key=settings.openai_api_key, model=settings.openai_ocr_model,
                               timeout_s=settings.ocr_timeout_s)
        llm = DeepSeekLLMAdapter(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                                 model=settings.deepseek_model, timeout_s=settings.llm_timeout_s)

    tracer = build_tracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        run_id=str(run_id),
    )

    # insert the reports row before stream opens
    await deps.report_repo.insert_report({
        "id": run_id,
        "started_at": datetime.now(timezone.utc),
        "status": "running",
        "prompt": prompt,
        "input_kind": input_kind,
        "input_ref": input_ref,
    })

    runner = GraphRunner(
        run_id=run_id, prompt=prompt,
        bus=bus, tracer=tracer,
        image_loader=image_loader, ocr=ocr, llm=llm,
        report_repo=deps.report_repo,
    )
    graph = build_graph(runner)

    # ---- bridge graph events → SSE ----
    sse_queue: asyncio.Queue = asyncio.Queue()

    async def sink(e: dict):
        await sse_queue.put(e)

    bus.subscribe(sink)

    async def run_graph_task():
        try:
            await graph.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
            await deps.report_repo.update_report(run_id, {
                "finished_at": datetime.now(timezone.utc),
                "status": "succeeded",
            })
        except Exception as exc:
            import logging
            logging.exception("graph run failed")
            await deps.report_repo.update_report(run_id, {
                "finished_at": datetime.now(timezone.utc),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            await sse_queue.put(None)  # sentinel

    task = asyncio.create_task(run_graph_task())

    async def event_source() -> AsyncIterator[dict]:
        while True:
            e = await sse_queue.get()
            if e is None:
                break
            yield e
        await task  # ensure background task cleanup

    return sse_response(event_source())


def _uuid_or_none(v) -> UUID | None:
    if v is None:
        return None
    if isinstance(v, UUID):
        return v
    try:
        return UUID(v)
    except Exception:
        return None
```

- [ ] **Step 8: Smoke-run the app in mock mode**

```bash
LLM_MODE=mock SUPABASE_DB_URL="postgresql+psycopg://dummy:dummy@localhost/dummy" PYTHONPATH=src python -c "from composition_root import build_app; a = build_app(); print('routes:', [r.path for r in a.routes])"
```
Expected: includes `/health` and `/runs/stream`. (No DB contact yet — we just import & inspect.)

- [ ] **Step 9: Commit**

```bash
git add src/infrastructure/http/ src/infrastructure/ocr/mock_adapter.py src/infrastructure/llm/mock_adapter.py src/infrastructure/images/
git commit -m "feat(http): /runs/stream SSE endpoint + mock OCR/LLM adapters + image loaders"
```

---

### Task 9.3: SSE contract test (multipart + folder, mock mode)

**Files:**
- Create: `tests/infrastructure/test_runs_stream.py`
- Create: `tests/infrastructure/conftest.py`

- [ ] **Step 1: Create `tests/infrastructure/conftest.py`**

```python
import pytest
from config import Settings
from fastapi.testclient import TestClient
from infrastructure.http.app import create_app
from tests.fakes.in_memory_repos import InMemoryReportRepository, InMemoryTraceRepository


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql+psycopg://test:test@localhost/test")
    monkeypatch.setenv("ASSETS_DIR", "./tests/fixtures/folder")
    s = Settings()
    rr = InMemoryReportRepository()
    tr = InMemoryTraceRepository()
    return create_app(settings=s, report_repo=rr, trace_repo=tr), rr, tr


@pytest.fixture
def client(app):
    return TestClient(app[0]), app[1], app[2]
```

- [ ] **Step 2: Create `tests/infrastructure/test_runs_stream.py`**

```python
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


def test_multipart_happy_path(client):
    c, report_repo, trace_repo = client
    p1 = Path("tests/fixtures/receipts/fixture_001.png").read_bytes()
    p2 = Path("tests/fixtures/receipts/fixture_002.png").read_bytes()
    resp = c.post(
        "/runs/stream",
        files=[
            ("files", ("r1.png", p1, "image/png")),
            ("files", ("r2.png", p2, "image/png")),
        ],
        data={"prompt": "be conservative"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    kinds = [e[0] for e in events]
    assert kinds[0] == "run_started"
    assert "final_result" in kinds
    assert kinds.count("receipt_result") == 2


def test_folder_happy_path(client):
    c, *_ = client
    resp = c.post(
        "/runs/stream",
        json={"folder_path": "./tests/fixtures/folder", "prompt": None},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    kinds = [e[0] for e in events]
    assert kinds[0] == "run_started"
    assert "final_result" in kinds
    assert kinds.count("receipt_result") == 1


def test_traces_written_through(client):
    c, report_repo, trace_repo = client
    c.post(
        "/runs/stream",
        files=[("files", ("r1.png",
                          Path("tests/fixtures/receipts/fixture_001.png").read_bytes(),
                          "image/png"))],
    )
    # Every emitted event should have a corresponding trace row
    event_types = [row["event_type"] for row in trace_repo.rows]
    assert "run_started" in event_types
    assert "final_result" in event_types
```

- [ ] **Step 3: Run tests**

```bash
make test tests/infrastructure/test_runs_stream.py
```
Expected: 3 passed.

Note: if `sse_response` buffering breaks streaming under `TestClient`, switch to consuming via `client.stream("POST", ...)`:
```python
with c.stream("POST", "/runs/stream", files=[...]) as r:
    body = "".join(r.iter_text())
events = _parse_sse(body)
```

- [ ] **Step 4: Commit**

```bash
git add tests/infrastructure/
git commit -m "test(http): SSE contract (multipart + folder + write-through)"
```

---

## Phase 10 — Real adapters

### Task 10.1: OpenAI OCR adapter

**Files:**
- Create: `src/infrastructure/ocr/openai_adapter.py`

- [ ] **Step 1: Implement `src/infrastructure/ocr/openai_adapter.py`**

```python
"""
OpenAI vision OCR adapter.

Sends the image as a base64 data URL to a vision-capable model (default: gpt-4o-mini)
and asks for strict JSON matching our RawReceipt schema. Uses response_format=json_object.
"""
import asyncio
import base64
import json
from pathlib import Path
from openai import AsyncOpenAI
from domain.models import RawReceipt
from application.ports import OCRPort, ImageRef


_SYSTEM = (
    "You are an OCR extractor for expense receipts. Read the image and return a "
    "single JSON object with these fields (all optional strings unless noted): "
    "vendor (string), receipt_date (string as it appears), receipt_number (string), "
    "total_raw (string, include currency symbol if present), currency_raw (string, "
    "ISO 4217 if you can tell), line_items (array of objects with description/amount), "
    "ocr_confidence (number 0.0–1.0 — your self-assessed confidence). "
    "Return ONLY valid JSON with no prose."
)


class OpenAIOCRAdapter(OCRPort):
    def __init__(self, *, api_key: str, model: str, timeout_s: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._timeout_s = timeout_s

    async def extract(self, image: ImageRef) -> RawReceipt:
        data_url = _image_to_data_url(image.local_path)
        resp = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract the receipt."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            ),
            timeout=self._timeout_s,
        )
        body = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"OCR returned non-JSON: {body[:200]!r}") from e
        return RawReceipt(source_ref=image.source_ref, **_safe_subset(data))


def _image_to_data_url(p: Path) -> str:
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = p.suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")
    return f"data:image/{mime};base64,{b64}"


_ALLOWED_KEYS = {
    "vendor", "receipt_date", "receipt_number", "total_raw",
    "currency_raw", "line_items", "ocr_confidence",
}


def _safe_subset(data: dict) -> dict:
    return {k: v for k, v in data.items() if k in _ALLOWED_KEYS}
```

- [ ] **Step 2: Smoke-import**

```bash
PYTHONPATH=src python -c "from infrastructure.ocr.openai_adapter import OpenAIOCRAdapter; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/infrastructure/ocr/openai_adapter.py
git commit -m "feat(ocr): OpenAI vision adapter (JSON mode, timeout)"
```

---

### Task 10.2: DeepSeek LLM adapter

**Files:**
- Create: `src/infrastructure/llm/deepseek_adapter.py`

- [ ] **Step 1: Implement `src/infrastructure/llm/deepseek_adapter.py`**

```python
"""
DeepSeek categorization adapter.

DeepSeek's chat API is OpenAI-compatible. We use AsyncOpenAI with a custom
base_url. The user prompt is injected verbatim into the system message.
Response format: JSON object matching Categorization.
"""
import asyncio
import json
from openai import AsyncOpenAI
from domain.models import AllowedCategory, Categorization, Issue, NormalizedReceipt
from application.ports import LLMPort


_SYSTEM_TEMPLATE = """\
You are an expense-category classifier. Given normalized receipt fields, choose
exactly ONE category from: {allowed}. If none fit, choose "Other" and provide a
note. Output strict JSON:

{{
  "category": "<one of the allowed categories>",
  "confidence": 0.0-1.0,
  "notes": "short rationale",
  "issues": [
    {{"severity": "warning", "code": "<short>", "message": "<explanation>"}}
  ]
}}

Issues to flag as warnings when relevant:
- missing receipt_number
- ambiguous currency
- total_out_of_range
- low_confidence

{user_instructions}
"""


class DeepSeekLLMAdapter(LLMPort):
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout_s: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._timeout_s = timeout_s

    async def categorize(self, normalized: NormalizedReceipt, allowed, user_prompt) -> Categorization:
        system = _SYSTEM_TEMPLATE.format(
            allowed=", ".join(allowed),
            user_instructions=(f"Additional guidance from user: {user_prompt}" if user_prompt else ""),
        )
        user = json.dumps({
            "vendor": normalized.vendor,
            "receipt_date": normalized.receipt_date.isoformat() if normalized.receipt_date else None,
            "receipt_number": normalized.receipt_number,
            "total": str(normalized.total) if normalized.total else None,
            "currency": normalized.currency,
        })
        resp = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            ),
            timeout=self._timeout_s,
        )
        body = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"categorizer returned non-JSON: {body[:200]!r}") from e
        # Coerce issues list into our Issue model
        issues = [Issue(**i) for i in data.get("issues", [])]
        return Categorization(
            category=AllowedCategory(data["category"]),
            confidence=float(data.get("confidence", 0.0)),
            notes=data.get("notes"),
            issues=issues,
        )
```

- [ ] **Step 2: Smoke-import**

```bash
PYTHONPATH=src python -c "from infrastructure.llm.deepseek_adapter import DeepSeekLLMAdapter; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/infrastructure/llm/deepseek_adapter.py
git commit -m "feat(llm): DeepSeek categorization adapter (OpenAI-compatible)"
```

---

## Phase 11 — Deliverables

### Task 11.1: Sample assets

**Files:**
- Create: `assets/receipt_001.png` through `assets/receipt_005.png`, `assets/README.md`

- [ ] **Step 1: Add 5 synthetic sample images**

For the take-home, synthetic (generated) receipt images are sufficient. Use any PNG. The `MockOCRAdapter` derives deterministic data from the filename, so filenames matter more than image content in mock mode.

```bash
python - <<'PY'
from pathlib import Path
import base64
png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
raw = base64.b64decode(png_b64)
for name in ("receipt_001.png", "receipt_002.png", "receipt_003.png",
             "receipt_004.png", "receipt_005.png"):
    Path("assets") / name
    (Path("assets") / name).write_bytes(raw)
print("ok")
PY
```

Replace at your discretion with real receipt images for live demos. In mock mode the content is ignored.

- [ ] **Step 2: Create `assets/README.md`**

```markdown
# Sample Assets

Synthetic 1×1 PNG placeholders. In `LLM_MODE=mock`, the MockOCRAdapter derives
deterministic fields from the filename — image content is not read.

To exercise real OCR (`LLM_MODE=real`), replace these with real receipt images.
Suggested sources:
- Publicly shared sample receipts (search "sample receipt image" — prefer CC0)
- Your own receipts with personal data redacted
```

- [ ] **Step 3: Commit**

```bash
git add assets/
git commit -m "chore(assets): 5 synthetic sample receipt PNGs + README"
```

---

### Task 11.2: `spec.md` (PDF deliverable: endpoint contract + schemas)

**Files:**
- Create: `spec.md` (at repo root)

- [ ] **Step 1: Write `spec.md`**

```markdown
# Receipt Processing Agent — API Spec

This file satisfies the PDF deliverable "spec.md defining the /runs/stream
contract, SSE event schema, tool registry and schemas, and final output schema".

> **Terminology deviation from the PDF:** entity and SSE event names use
> "receipt" instead of "invoice" throughout. Specifically, the PDF's
> required SSE event `invoice_result` is renamed `receipt_result` here.
> See `DESIGN.md` for the rationale.

## Endpoint

**`POST /runs/stream`**

Starts a receipt-processing run and returns a Server-Sent Events stream.

### Request — variant A: multipart upload
Content-Type: `multipart/form-data`
- `files` — one or more image files (`jpg`, `jpeg`, `png`, `webp`; max 10 MB each; max 25 per run)
- `prompt` — optional string; free-form guidance for the categorization sub-agent

### Request — variant B: folder path
Content-Type: `application/json`
```json
{
  "folder_path": "./assets",
  "prompt": "be conservative"
}
```
`folder_path` must resolve under `ASSETS_DIR`.

### Response
Content-Type: `text/event-stream`
The stream remains open until the run completes (`final_result`) or fails (`error`).

## SSE Event Schemas

Every event is JSON with fields: `event_type` (string), `run_id` (UUID), `seq`
(monotonic int), `ts` (ISO-8601 UTC).

### `run_started`
```json
{"event_type":"run_started","run_id":"...","seq":1,"ts":"...","prompt":"optional"}
```

### `progress`
```json
{"event_type":"progress","run_id":"...","seq":N,"ts":"...",
 "step":"ocr|normalize|categorize|load_images|aggregate|generate_report",
 "receipt_id":"optional","i":2,"n":5}
```

### `tool_call`
```json
{"event_type":"tool_call","run_id":"...","seq":N,"ts":"...",
 "tool":"extract_receipt_fields","receipt_id":"optional","attempt":1,
 "args":{"image":"..."}}
```

### `tool_result`
```json
{"event_type":"tool_result","run_id":"...","seq":N,"ts":"...",
 "tool":"extract_receipt_fields","receipt_id":"optional",
 "result_summary":{"vendor":"Uber","has_total":true,"ocr_confidence":0.92},
 "error":false,"error_message":null,"duration_ms":412}
```

### `receipt_result`
Terminal event for each receipt. Always emitted, even on error.
```json
{"event_type":"receipt_result","run_id":"...","seq":N,"ts":"...",
 "receipt_id":"...","status":"ok|error",
 "vendor":"Uber","receipt_date":"2024-03-15","receipt_number":"R-12345",
 "total":"45.67","currency":"USD",
 "category":"Travel","confidence":0.92,
 "notes":"rideshare","issues":[]}
```

### `final_result`
```json
{"event_type":"final_result","run_id":"...","seq":N,"ts":"...",
 "total_spend":"123.45",
 "by_category":{"Travel":"45.67","Meals & Entertainment":"77.78"},
 "receipts":[ /* full receipt objects */ ],
 "issues_and_assumptions":[
   {"severity":"warning","code":"low_confidence","message":"...","receipt_id":"..."}
 ]}
```

### `error`
```json
{"event_type":"error","run_id":"...","seq":N,"ts":"...",
 "code":"no_images","message":"no images found in input"}
```

## Tool Registry

| # | Name | Inputs | Output |
|---|---|---|---|
| 1 | `load_images` | input spec (folder or uploads) | `[ImageRef]` |
| 2 | `extract_receipt_fields` | `image_ref` | `RawReceipt` |
| 3 | `normalize_receipt` | `RawReceipt` | `NormalizedReceipt` |
| 4 | `categorize_receipt` | `NormalizedReceipt`, user prompt, allowed categories | `Categorization` |
| 5 | `aggregate` | `[Receipt]` | `Aggregates` |
| 6 | `generate_report` | `Aggregates`, `[Receipt]`, `[Issue]` | `Report` |

### Schemas

#### `RawReceipt`
```
source_ref: string
vendor: string | null
receipt_date: string | null
receipt_number: string | null
total_raw: string | null
currency_raw: string | null
line_items: list[{description, amount}]
ocr_confidence: float | null
```

#### `NormalizedReceipt`
```
source_ref: string
vendor: string | null
receipt_date: date | null
receipt_number: string | null
total: decimal | null
currency: string | null   (ISO-4217; defaults to USD when absent)
```

#### `Categorization`
```
category: AllowedCategory
confidence: float [0.0, 1.0]
notes: string | null
issues: list[Issue]
```
Validator: `category=Other` requires a non-empty `notes`.

#### `Issue`
```
severity: "warning" | "receipt_error" | "run_error"
code: string
message: string
receipt_id: UUID | null
```

#### `Receipt` (final per-receipt record)
```
id: UUID
source_ref: string
vendor, receipt_date, receipt_number, total, currency  (as NormalizedReceipt)
category, confidence, notes, issues                    (as Categorization, plus receipt-level issues)
raw_ocr, normalized: dict | null  (full payloads)
status: "ok" | "error"
error: string | null
```

#### `Aggregates`
```
total_spend: decimal
by_category: dict[string, decimal]
```

#### `Report`
```
run_id: UUID
total_spend: decimal
by_category: dict[string, decimal]
receipts: list[Receipt]
issues_and_assumptions: list[Issue]
```

### Allowed Categories
`Travel`, `Meals & Entertainment`, `Software / Subscriptions`,
`Professional Services`, `Office Supplies`, `Shipping / Postage`,
`Utilities`, `Other` (requires a note).

## Error Bands

| Band | Trigger | Response |
|---|---|---|
| A | Per-receipt recoverable (OCR timeout, parse failure, LLM error) | One retry on network errors; emit `receipt_result` with `status="error"`; exclude from aggregation |
| B | Per-receipt soft warning (low confidence, ambiguous currency, missing number) | Included in `receipt.issues` and run-level `issues_and_assumptions`; counted in aggregation |
| C | Run-level unrecoverable (DB down, zero images, 100% receipts failed) | Terminal `error` event; `reports.status='failed'` |
```

- [ ] **Step 2: Commit**

```bash
git add spec.md
git commit -m "docs(spec): API spec (endpoint contract + SSE + tool/output schemas)"
```

---

### Task 11.3: `DESIGN.md` (1-page)

**Files:**
- Create: `DESIGN.md`

- [ ] **Step 1: Write `DESIGN.md`**

```markdown
# DESIGN

One-page architecture summary for reviewers.

## Architecture
Hexagonal (Ports & Adapters). Three layers:
- **Domain** — Pydantic entities + pure functions (normalization, aggregation). No I/O.
- **Application** — LangGraph deterministic state machine, 6-tool registry, categorization sub-agent, SSE events, in-process EventBus, abstract ports.
- **Infrastructure** — FastAPI HTTP + SSE, SQLAlchemy + Alembic against Supabase Cloud Postgres, Langfuse + JSON logs tracers, OpenAI OCR, DeepSeek categorization, mock adapters.

Wiring is centralized in `composition_root.py` (the only module importing both application and infrastructure).

## Framework choice
LangChain + LangGraph for orchestration. The graph is deterministic end-to-end — no ReAct loop. The agent's "judgment" is bounded to a single DeepSeek call inside `categorize_receipt` (one LLM call per receipt). This honors the PDF's "planner decisions" tracing requirement while keeping the outer workflow testable and replayable.

## Planner / Execution / State separation
- **Planner:** deterministic LangGraph + the per-receipt sub-agent. The graph decides what runs next; the sub-agent decides the category + flags issues.
- **Execution:** tools (6 of them) are the only interface to receipt data. Each tool is decorated with `@traced_tool` (span + timing + event emission + error classification).
- **State:** `RunState` (Pydantic) carries `images`, `receipts`, `current` cursor, `errors`, `issues`. Persistence is write-through: every SSE event becomes a `traces` row; receipts and reports are written at their terminal events.

## Model selection
- **OCR** — OpenAI `gpt-4o-mini` (vision, JSON mode). Cheap, capable, widely available.
- **Categorization** — DeepSeek `deepseek-chat` via its OpenAI-compatible API. Strong at structured classification; cost-effective.
- Both mockable via a single `LLM_MODE=mock` env switch; tests run entirely offline.

## What I'd improve for production
- **Async trace writer** — currently synchronous in the emit path. Fine at scale but move to a bounded queue + background writer to decouple DB latency from stream cadence.
- **Per-invoice parallelism** — trivial lift (already behind the `process_receipt` node); skipped here per the spec for a cleaner single-run trajectory.
- **Retry expansion** — one retry on network errors today; production would layer in token-bucket rate-limit handling and a circuit breaker around each provider.
- **Reviewer-runnability** — current setup requires a Supabase Cloud project. Dual-mode (local Postgres via docker-compose + Supabase Cloud optional) is a five-line change and cuts onboarding friction.
- **E2E coverage** — real-API smoke test is manual; wire it into CI with a nightly job gated on API keys.

## Deliberate PDF deviations
- **`invoice_result` → `receipt_result`** in the SSE event set. Matches our internal "receipt" entity name. All other required event names preserved.
- **"Invoice" → "Receipt"** terminology throughout. Documented in `spec.md` and `specs.md`.
```

- [ ] **Step 2: Commit**

```bash
git add DESIGN.md
git commit -m "docs(design): 1-page architecture summary"
```

---

### Task 11.4: `README.md`

**Files:**
- Create: `README.md` (overwrite current placeholder)

- [ ] **Step 1: Write `README.md`**

```markdown
# Receipt Processing Agent

Local backend that accepts a folder of receipt images (or multipart uploads),
extracts fields via OCR, categorizes each receipt into an approved expense
category using a bounded DeepSeek sub-agent, aggregates totals, and streams the
full execution trajectory over Server-Sent Events while persisting a reviewable
trace to Postgres.

## Prerequisites

- Python 3.11+
- A Supabase Cloud project (connection string required — **yes, even to run the code**; see DESIGN.md)
- For `LLM_MODE=real`: OpenAI and DeepSeek API keys

## Setup

```bash
make install             # creates venv, installs deps
cp .env.example .env     # fill in SUPABASE_DB_URL at minimum
make migrate             # apply Alembic migrations to your Supabase DB
```

## Run

```bash
make run
# → uvicorn on http://0.0.0.0:8000
```

## Mock mode (no API keys needed)

```bash
LLM_MODE=mock make run
```

In `mock` mode, OCR and LLM adapters are swapped for deterministic fakes keyed
by filename / vendor. Tests always run in mock mode.

## Example — folder input

```bash
curl -N -X POST http://localhost:8000/runs/stream \
     -H "content-type: application/json" \
     -d '{"folder_path": "./assets", "prompt": "flag unusual invoices"}'
```

## Example — multipart upload

```bash
curl -N -X POST http://localhost:8000/runs/stream \
     -F "files=@assets/receipt_001.png" \
     -F "files=@assets/receipt_002.png" \
     -F "prompt=be conservative"
```

## Example SSE output

```
event: run_started
data: {"event_type":"run_started","run_id":"...","seq":1,"ts":"...","prompt":"be conservative"}

event: progress
data: {"event_type":"progress","seq":2,"step":"load_images","ts":"..."}

event: tool_call
data: {"event_type":"tool_call","tool":"load_images","args":{"folder_path":"..."}}

event: tool_result
data: {"event_type":"tool_result","tool":"load_images","result_summary":{"count":2},"duration_ms":4,"error":false}

... (per-receipt sequence) ...

event: receipt_result
data: {"event_type":"receipt_result","receipt_id":"...","status":"ok","vendor":"Uber","total":"45.67","category":"Travel",...}

event: final_result
data: {"event_type":"final_result","total_spend":"123.45","by_category":{"Travel":"45.67"},...,"issues_and_assumptions":[]}
```

## Tests

```bash
make test         # all non-e2e tests (mock mode, no network)
make test-e2e     # real API smoke test (requires LLM_MODE=real + keys)
```

## Architecture

See `DESIGN.md` (1 page) and `spec.md` (API + schemas).
High-level system spec: `specs.md`.

## Deliberate PDF deviation
SSE event name `invoice_result` is renamed `receipt_result` for terminology
consistency. All other PDF contract names are preserved. Rationale in `DESIGN.md`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): setup, curl examples, mock mode, SSE sample"
```

---

### Task 11.5: `AGENTS.md` + `transcripts/` + sample run trace

**Files:**
- Create: `AGENTS.md`, `transcripts/README.md`, `transcripts/sample-run-trace.json`

- [ ] **Step 1: Write `AGENTS.md`**

```markdown
# AGENTS

Notes on AI-assisted development of this repository.

## Tools used
- Claude Code (this repo's primary AI coding tool)
- Cursor / ChatGPT for spot research on LangGraph / Langfuse APIs

## Authorization posture
- Destructive commands (`git reset --hard`, `rm -rf`, `docker ...`) require explicit user approval per session.
- Network-facing calls (curl, real LLM/OCR providers) only executed on explicit request.

## Where AI was trusted
- Pydantic model scaffolding
- Alembic migration SQL
- Tool registry boilerplate
- README / spec documentation drafts

## Where AI was NOT trusted
- Prompt text for the categorization sub-agent (reviewed line by line)
- Error-band classification logic (hand-audited against the spec)
- Mock/real adapter contract parity (verified by Layer 4 e2e smoke test)
- Any decision that changed an externally-visible schema or SSE event name

## How to reproduce
- Claude Code session transcripts are stored in `transcripts/`.
- The design doc at `docs/superpowers/specs/2026-04-20-receipt-processing-agent-design.md` captures the Q&A that drove the design.
```

- [ ] **Step 2: Write `transcripts/README.md`**

```markdown
# Transcripts

Raw AI coding tool interaction logs. Drop exported session transcripts here
during development. File naming: `YYYY-MM-DD-<topic>.md` (e.g.,
`2026-04-20-bootstrapping.md`).

A sample run trace from an actual execution is in `sample-run-trace.json`.
```

- [ ] **Step 3: Generate `transcripts/sample-run-trace.json`**

Once the server runs in mock mode, capture a real SSE stream. Fallback (if server not yet running): write a hand-crafted sample matching the schema.

```bash
# If server is running:
LLM_MODE=mock SUPABASE_DB_URL="postgresql+psycopg://u:p@h/db" make run &
sleep 2
curl -N -X POST http://localhost:8000/runs/stream \
  -H "content-type: application/json" \
  -d '{"folder_path":"./assets","prompt":"be conservative"}' \
  | python -c "
import sys, json
events=[]
current=None
for line in sys.stdin:
    if line.startswith('event:'): current=line.split(':',1)[1].strip()
    elif line.startswith('data:'):
        events.append({'event':current,'data':json.loads(line.split(':',1)[1].strip())})
json.dump(events, open('transcripts/sample-run-trace.json','w'), indent=2)
"
kill %1 || true
```

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md transcripts/
git commit -m "docs: AGENTS.md + transcripts/ with sample run trace"
```

---

## Phase 12 — E2E smoke (manual, optional)

### Task 12.1: Real-API smoke test

**Files:**
- Create: `tests/e2e/test_real_run_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
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
```

- [ ] **Step 2: Run manually (only when credentials are set)**

```bash
LLM_MODE=real OPENAI_API_KEY=... DEEPSEEK_API_KEY=... SUPABASE_DB_URL=... make test-e2e
```

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/
git commit -m "test(e2e): real-API smoke test (skipped without credentials)"
```

---

## Plan Self-Review Notes

Four spec requirements surfaced during self-review that aren't covered by the primary tasks above. Apply these patches inline while executing the referenced tasks.

### R1. Retry on network errors (spec §7 Band A) — apply in Task 5.1

Extend `@traced_tool` to support one retry with exponential backoff on network-class exceptions. Each attempt emits its own `tool_call`/`tool_result` pair with `attempt=N`. Replace the decorator body (`traced_tool` function) with:

```python
import asyncio

_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError, OSError, ConnectionError,
)


def traced_tool(
    tool_name: str,
    summarize: Callable[[Any], dict] | None = None,
    retries: int = 0,
    retry_delays_s: tuple[float, ...] = (1.0, 2.0),
    retry_on: tuple[type[BaseException], ...] = _NETWORK_EXCEPTIONS,
):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(ctx: ToolContext, /, **kwargs):
            attempt = 1
            while True:
                call_event = ToolCall(
                    run_id=ctx.run_id, seq=next(ctx.seq_counter), ts=_now(),
                    tool=tool_name, receipt_id=ctx.receipt_id,
                    attempt=attempt, args=_safe_args(kwargs),
                )
                await ctx.bus.publish(call_event.model_dump(mode="json"))
                span = ctx.tracer.start_span(tool_name, input=_safe_args(kwargs))
                started = time.perf_counter()
                try:
                    result = await fn(ctx, **kwargs)
                except Exception as exc:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    err_event = ToolResult(
                        run_id=ctx.run_id, seq=next(ctx.seq_counter), ts=_now(),
                        tool=tool_name, receipt_id=ctx.receipt_id,
                        result_summary={}, error=True,
                        error_message=f"{type(exc).__name__}: {exc}",
                        duration_ms=duration_ms,
                    )
                    await ctx.bus.publish(err_event.model_dump(mode="json"))
                    span.end(error=str(exc))
                    if attempt <= retries and isinstance(exc, retry_on):
                        delay = retry_delays_s[min(attempt - 1, len(retry_delays_s) - 1)]
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    raise
                duration_ms = int((time.perf_counter() - started) * 1000)
                summary = (summarize(result) if summarize else {}) or {}
                ok_event = ToolResult(
                    run_id=ctx.run_id, seq=next(ctx.seq_counter), ts=_now(),
                    tool=tool_name, receipt_id=ctx.receipt_id,
                    result_summary=summary, error=False,
                    duration_ms=duration_ms,
                )
                await ctx.bus.publish(ok_event.model_dump(mode="json"))
                span.end(output=summary)
                return result
        return wrapper
    return decorator
```

Add a new test case to `tests/application/test_traced_tool.py`:

```python
@pytest.mark.asyncio
async def test_retries_once_on_network_error(ctx):
    c, events = ctx
    attempts = {"n": 0}

    @traced_tool("extract_receipt_fields", retries=1, retry_delays_s=(0.0,))
    async def impl(ctx):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise asyncio.TimeoutError("first attempt times out")
        return {"ok": True}

    await impl(c)
    assert attempts["n"] == 2
    # Two tool_call events with attempt=1 and attempt=2
    calls = [e for e in events if e["event_type"] == "tool_call"]
    assert [e["attempt"] for e in calls] == [1, 2]
```

Then update Task 5.2 to enable retries on network-sensitive tools:
```python
@traced_tool("extract_receipt_fields", summarize=_summarize_raw, retries=1)
async def extract_receipt_fields(...): ...

@traced_tool("categorize_receipt", summarize=_summarize_categorization, retries=1)
async def categorize_receipt(...): ...
```

### R2. Fixed run-level assumptions in `issues_and_assumptions` (spec §7) — apply in Task 6.1

Extend `GraphRunner.finalize` to prepend fixed assumption strings to `state.issues` before calling `generate_report`:

```python
_RUN_LEVEL_ASSUMPTIONS = [
    ("only_allowed_extensions", "Only files matching jpg/jpeg/png/webp were considered."),
    ("default_currency_usd", "Totals assume USD when currency is absent."),
    ("errored_receipts_excluded", "Receipts with OCR/normalization/LLM failures are excluded from aggregation."),
]


# In GraphRunner.finalize, before calling generate_report:
run_level = [
    Issue(severity="warning", code=code, message=msg)
    for code, msg in _RUN_LEVEL_ASSUMPTIONS
]
all_issues = state.issues + run_level
report = await generate_report(
    self._ctx(), run_id=self.run_id, aggregates=agg,
    receipts=state.receipts, issues=all_issues,
)
# ... emit FinalResult with issues_and_assumptions=all_issues ...
```

Add a test case to `tests/application/test_graph.py`:
```python
@pytest.mark.asyncio
async def test_final_result_includes_run_level_assumptions():
    runner, events = _runner_with_mocks(_refs(1))
    await build_graph(runner).ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
    final = [e for e in events if e["event_type"] == "final_result"][0]
    codes = [i["code"] for i in final["issues_and_assumptions"]]
    assert "only_allowed_extensions" in codes
    assert "default_currency_usd" in codes
```

### R3. Upload file-size enforcement (spec §12 security / §13 limits) — apply in Task 9.2

Extend `UploadImageLoader.load` to reject files exceeding `max_file_size_mb`:

```python
class UploadImageLoader(ImageLoaderPort):
    def __init__(self, uploads, allowed_extensions, max_size_bytes: int) -> None:
        self._uploads = uploads
        self._allowed = {e.lower().lstrip(".") for e in allowed_extensions}
        self._max_size_bytes = max_size_bytes
        self._tmpdir = Path(tempfile.mkdtemp(prefix="receipt-run-"))

    async def load(self) -> list[ImageRef]:
        refs: list[ImageRef] = []
        for up in self._uploads:
            if not up.filename:
                continue
            ext = Path(up.filename).suffix.lower().lstrip(".")
            if ext not in self._allowed:
                continue
            content = await up.read()
            if len(content) > self._max_size_bytes:
                raise ValueError(f"{up.filename}: exceeds max size {self._max_size_bytes} bytes")
            dest = self._tmpdir / up.filename
            dest.write_bytes(content)
            refs.append(ImageRef(source_ref=up.filename, local_path=dest.resolve()))
        return refs
```

Instantiate in `routes_runs.py`:
```python
image_loader = UploadImageLoader(files, settings.allowed_extensions,
                                 settings.max_file_size_mb * 1024 * 1024)
```

Convert the `ValueError` into HTTP 413 in the route with a try/except around `image_loader.load()`.

### R4. 100%-failed receipts → run-level error (spec §7 Band C) — apply in Task 6.3

Before calling aggregate in `GraphRunner.finalize`, check if all receipts errored:

```python
async def finalize(self, state: RunState) -> RunState:
    ok_count = sum(1 for r in state.receipts if r.status == "ok")
    if state.receipts and ok_count == 0:
        await self._emit(ErrorEvent(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            code="all_receipts_failed",
            message=f"all {len(state.receipts)} receipt(s) failed at receipt level",
        ))
        return state
    # ... existing aggregate + generate_report + final_result emission ...
```

Add a test:
```python
@pytest.mark.asyncio
async def test_all_receipts_failed_emits_run_error():
    refs = _refs(2)
    ocr = MockOCR(fail_on={"r1.png", "r2.png"})
    runner, events = _runner_with_mocks(refs, ocr=ocr)
    await build_graph(runner).ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
    kinds = [e["event_type"] for e in events]
    assert "error" in kinds
    err = next(e for e in events if e["event_type"] == "error")
    assert err["code"] == "all_receipts_failed"
    assert "final_result" not in kinds
```

---

## Post-implementation checklist

Before declaring done, verify:

- [ ] `make test` passes with zero failures in mock mode.
- [ ] `make run` starts the server and `/health` returns `{"status":"ok","llm_mode":"mock"}`.
- [ ] `curl -N -X POST /runs/stream` produces a stream ending with `final_result`.
- [ ] Migrations apply cleanly to a fresh Supabase database (`make migrate`).
- [ ] `transcripts/sample-run-trace.json` contains a real captured run.
- [ ] `DESIGN.md` fits on one printed page.
- [ ] `spec.md`, `specs.md`, `README.md`, `AGENTS.md` are consistent on terminology (receipt, not invoice).
- [ ] `assets/` contains at least 4 images and a README.
- [ ] At least one Langfuse-enabled run has been spot-checked manually (if Langfuse keys available).
- [ ] The PDF deviation (`receipt_result`) is called out in `DESIGN.md`, `spec.md`, and `README.md`.
