# `image_paths` Input Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third input mode to `POST /runs/stream` — clients send JSON `{"image_paths": [...], "prompt": "..."}` to process an explicit list of files under `./assets/`, alongside the existing `folder_path` and multipart-upload modes.

**Architecture:** Two Pydantic models in `routes_runs.py` (`FolderInput` unchanged, new `PathsInput`). One new `ImageLoaderPort` implementation (`ListPathImageLoader`) that validates every path up front and raises on any failure — so bad input is rejected before the graph starts. Handler branches on which key is present in the JSON body; requires exactly one.

**Tech Stack:** FastAPI, Pydantic v2, existing `ImageLoaderPort` / `ImageRef` contract, pytest.

**Reference spec:** `docs/superpowers/specs/2026-04-21-image-paths-input-design.md`.

---

## Prerequisites

This branch (`feat/image-paths-input`) has been rebased onto `origin/fix/runs-stream-content-type` so that the Content-Type-based JSON/multipart dispatch is already present in `routes_runs.py`. Do NOT branch off `main` — it lacks the `RuntimeError: Stream consumed` fix and the new JSON handling would regress on day one.

Before starting, verify:

```bash
cd /Users/juanmartincardenasortiz/AiProjects/CaseWareHomeTakeTest
git log --oneline main..HEAD | head -3
```

Should show (at least) these commits under your new spec commit:

```
<sha> docs(spec): image_paths input mode for /runs/stream
6fb2ab1 fix(alembic): keep +psycopg scheme so migrations use psycopg v3
1de524d fix(http): /runs/stream parses body based on Content-Type
```

Then verify the suite is green:

```bash
PYTHONPATH=src .venv/bin/pytest -v -m "not e2e"
```

Expected: `100 passed, 1 skipped`.

---

## File Structure

**Files to create (new):**
- `src/infrastructure/images/list_path_loader.py` — `ListPathImageLoader` class + `TooManyPathsError` exception. Implements `ImageLoaderPort`. Single responsibility: validate a list of paths against allowed extensions + `assets_dir` guardrail, build `ImageRef`s.
- `tests/infrastructure/test_list_path_loader.py` — 7 unit tests covering every validation branch.

**Files to modify:**
- `src/infrastructure/http/routes_runs.py` — add `PathsInput` Pydantic model; extend the JSON branch of `post_runs_stream` to dispatch on which field is present.
- `tests/infrastructure/test_runs_stream.py` — append 3 tests (happy path, bad-path 400, both-fields 422).

**Files NOT touched:** graph, tools, chat model, domain, other loaders, composition root. The new loader sits behind the existing `ImageLoaderPort`.

---

## Phase 1 — `ListPathImageLoader`

TDD-driven. Each task writes tests first, then the minimal implementation to pass.

### Task 1.1: Create the module stub + happy-path test

**Files:**
- Create: `src/infrastructure/images/list_path_loader.py`
- Create: `tests/infrastructure/test_list_path_loader.py`

- [ ] **Step 1: Write failing test at `tests/infrastructure/test_list_path_loader.py`**

```python
"""Unit tests for ListPathImageLoader — validates a list of paths against
the ASSETS_DIR guardrail and allowed extensions."""
from pathlib import Path
import os
import pytest

from infrastructure.images.list_path_loader import (
    ListPathImageLoader, TooManyPathsError,
)


ALLOWED = {"jpg", "jpeg", "png", "webp", "pdf"}


def _make_assets(tmp_path: Path) -> Path:
    """Create a tmp assets dir with a valid image and return the dir path."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "r1.png").write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
    (assets / "r2.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpg")
    return assets


@pytest.mark.asyncio
async def test_loads_valid_paths_under_assets(tmp_path):
    assets = _make_assets(tmp_path)
    paths = [str(assets / "r1.png"), str(assets / "r2.jpg")]
    loader = ListPathImageLoader(paths, ALLOWED, assets)
    refs = await loader.load()
    assert len(refs) == 2
    assert refs[0].source_ref == paths[0]
    assert refs[1].source_ref == paths[1]
    assert refs[0].local_path == (assets / "r1.png").resolve()
    assert refs[1].local_path == (assets / "r2.jpg").resolve()
```

- [ ] **Step 2: Run, expect failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/infrastructure/test_list_path_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'infrastructure.images.list_path_loader'`.

- [ ] **Step 3: Create `src/infrastructure/images/list_path_loader.py`**

```python
"""Explicit-paths image loader.

Validates a list of caller-supplied paths against the assets_dir
guardrail and allowed extensions. All validation happens in __init__
so bad input is rejected before the graph runs.
"""
from __future__ import annotations
import os
from pathlib import Path

from application.ports import ImageLoaderPort, ImageRef


class TooManyPathsError(ValueError):
    """Distinct subclass so the HTTP layer can map it to 413."""


class ListPathImageLoader(ImageLoaderPort):
    def __init__(
        self,
        paths: list[str],
        allowed_extensions: set[str],
        assets_dir: Path,
    ) -> None:
        if len(paths) == 0:
            raise ValueError("image_paths must not be empty")

        allowed = {e.lower().lstrip(".") for e in allowed_extensions}
        assets_resolved = assets_dir.resolve()

        bad: list[tuple[str, str]] = []
        refs: list[ImageRef] = []
        for original in paths:
            p = Path(original).resolve()
            if not str(p).startswith(str(assets_resolved)):
                bad.append((original, f"path must be under {assets_resolved}"))
                continue
            if not p.exists():
                bad.append((original, "path does not exist"))
                continue
            if not p.is_file():
                bad.append((original, "path is not a file"))
                continue
            ext = p.suffix.lower().lstrip(".")
            if ext not in allowed:
                bad.append((original, f"extension .{ext} not in allowed extensions"))
                continue
            if not os.access(p, os.R_OK):
                bad.append((original, "path is not readable"))
                continue
            refs.append(ImageRef(source_ref=original, local_path=p))

        if bad:
            lines = [f"  {orig}: {why}" for orig, why in bad]
            raise ValueError("invalid image_paths:\n" + "\n".join(lines))

        self._refs = refs

    async def load(self) -> list[ImageRef]:
        return list(self._refs)
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/infrastructure/test_list_path_loader.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/images/list_path_loader.py tests/infrastructure/test_list_path_loader.py
git commit -m "feat(images): ListPathImageLoader — happy path"
```

### Task 1.2: Error cases + failure aggregation

**Files:**
- Modify: `tests/infrastructure/test_list_path_loader.py` (append)

The loader already implements error aggregation. This task proves each branch is reachable and error-messages correctly.

- [ ] **Step 1: Append 6 failing tests**

Append to `tests/infrastructure/test_list_path_loader.py`:

```python
def test_rejects_empty_list(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        ListPathImageLoader([], ALLOWED, tmp_path)


def test_rejects_path_outside_assets(tmp_path):
    assets = _make_assets(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="must be under"):
        ListPathImageLoader([str(outside)], ALLOWED, assets)


def test_rejects_missing_file(tmp_path):
    assets = _make_assets(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        ListPathImageLoader([str(assets / "ghost.png")], ALLOWED, assets)


def test_rejects_directory(tmp_path):
    assets = _make_assets(tmp_path)
    subdir = assets / "sub"
    subdir.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        ListPathImageLoader([str(subdir)], ALLOWED, assets)


def test_rejects_disallowed_extension(tmp_path):
    assets = _make_assets(tmp_path)
    exe = assets / "script.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(ValueError, match=r"extension \.exe"):
        ListPathImageLoader([str(exe)], ALLOWED, assets)


def test_aggregates_all_failures(tmp_path):
    assets = _make_assets(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG")
    exe = assets / "script.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(ValueError) as exc_info:
        ListPathImageLoader(
            [str(outside), str(assets / "ghost.png"), str(exe)],
            ALLOWED, assets,
        )
    msg = str(exc_info.value)
    # All three bad paths reported in one message
    assert "outside.png" in msg
    assert "ghost.png" in msg
    assert "script.exe" in msg
    assert msg.count("\n") >= 3  # header line + 3 reason lines
```

- [ ] **Step 2: Run, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/infrastructure/test_list_path_loader.py -v
```

Expected: all 7 tests pass (the one from Task 1.1 + 6 new ones).

Note: these tests pass against the Task 1.1 implementation — the implementation was complete. The tests just prove each path is reachable and the error shape is correct.

- [ ] **Step 3: Commit**

```bash
git add tests/infrastructure/test_list_path_loader.py
git commit -m "test(images): cover all ListPathImageLoader error branches"
```

### Task 1.3: Max-count enforcement via `TooManyPathsError`

**Files:**
- Modify: `src/infrastructure/images/list_path_loader.py`
- Modify: `tests/infrastructure/test_list_path_loader.py` (append)

- [ ] **Step 1: Append failing test**

```python
def test_max_files_per_run_enforced(tmp_path):
    assets = _make_assets(tmp_path)
    # Create more valid images than the limit
    for i in range(5):
        (assets / f"extra_{i}.png").write_bytes(b"\x89PNG")
    paths = [str(p) for p in assets.iterdir() if p.suffix == ".png"]
    assert len(paths) >= 3, "precondition: need at least 3 png files"
    with pytest.raises(TooManyPathsError, match="too many paths"):
        ListPathImageLoader(paths, ALLOWED, assets, max_files_per_run=2)


def test_max_files_per_run_default_is_unlimited(tmp_path):
    """When max_files_per_run is None (default), any count is accepted."""
    assets = _make_assets(tmp_path)
    for i in range(5):
        (assets / f"extra_{i}.png").write_bytes(b"\x89PNG")
    paths = [str(p) for p in assets.iterdir() if p.suffix == ".png"]
    loader = ListPathImageLoader(paths, ALLOWED, assets)  # no max
    # Should not raise
    assert loader is not None
```

- [ ] **Step 2: Run, expect failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/infrastructure/test_list_path_loader.py -v -k max_files
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'max_files_per_run'`.

- [ ] **Step 3: Extend `ListPathImageLoader.__init__`**

Modify the signature and add the check AFTER the per-path loop (so the count is on valid paths; invalid ones already rejected). Replace the signature and the final block of `__init__`:

```python
    def __init__(
        self,
        paths: list[str],
        allowed_extensions: set[str],
        assets_dir: Path,
        max_files_per_run: int | None = None,
    ) -> None:
        if len(paths) == 0:
            raise ValueError("image_paths must not be empty")

        allowed = {e.lower().lstrip(".") for e in allowed_extensions}
        assets_resolved = assets_dir.resolve()

        bad: list[tuple[str, str]] = []
        refs: list[ImageRef] = []
        for original in paths:
            p = Path(original).resolve()
            if not str(p).startswith(str(assets_resolved)):
                bad.append((original, f"path must be under {assets_resolved}"))
                continue
            if not p.exists():
                bad.append((original, "path does not exist"))
                continue
            if not p.is_file():
                bad.append((original, "path is not a file"))
                continue
            ext = p.suffix.lower().lstrip(".")
            if ext not in allowed:
                bad.append((original, f"extension .{ext} not in allowed extensions"))
                continue
            if not os.access(p, os.R_OK):
                bad.append((original, "path is not readable"))
                continue
            refs.append(ImageRef(source_ref=original, local_path=p))

        if bad:
            lines = [f"  {orig}: {why}" for orig, why in bad]
            raise ValueError("invalid image_paths:\n" + "\n".join(lines))

        if max_files_per_run is not None and len(refs) > max_files_per_run:
            raise TooManyPathsError(
                f"too many paths (max {max_files_per_run}, got {len(refs)})"
            )

        self._refs = refs
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/infrastructure/test_list_path_loader.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/images/list_path_loader.py tests/infrastructure/test_list_path_loader.py
git commit -m "feat(images): enforce max_files_per_run via TooManyPathsError"
```

---

## Phase 2 — HTTP dispatch

### Task 2.1: `PathsInput` model + handler JSON-branch dispatch

**Files:**
- Modify: `src/infrastructure/http/routes_runs.py`

The current JSON branch (after PR #3's Content-Type fix) looks like:

```python
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
```

This task adds the `PathsInput` model and dispatches on body keys.

- [ ] **Step 1: Read the current `routes_runs.py` to find the exact JSON branch**

```bash
.venv/bin/python -c "from pathlib import Path; print(Path('src/infrastructure/http/routes_runs.py').read_text())"
```

Locate the `else:` block that starts with `body = await request.json()`.

- [ ] **Step 2: Add the `PathsInput` model**

Find the existing `class FolderInput(BaseModel):` (around line 36). Add `PathsInput` directly below it:

```python
class FolderInput(BaseModel):
    folder_path: str
    prompt: str | None = None


class PathsInput(BaseModel):
    image_paths: list[str]
    prompt: str | None = None
```

- [ ] **Step 3: Add the new import for the loader + exception**

Near the top of the file, find the existing image-loader imports (lines 18-19). Add:

```python
from infrastructure.images.list_path_loader import ListPathImageLoader, TooManyPathsError
```

- [ ] **Step 4: Replace the JSON branch with discriminated dispatch**

Replace the existing `else:` block (the JSON branch) with:

```python
    else:
        body = await request.json()
        has_folder = "folder_path" in body
        has_paths = "image_paths" in body
        if has_folder and has_paths:
            raise HTTPException(422, "provide exactly one of folder_path or image_paths")
        if not has_folder and not has_paths:
            raise HTTPException(422, "body must include folder_path or image_paths")

        if has_paths:
            try:
                pi = PathsInput(**body)
            except Exception as e:
                raise HTTPException(422, f"invalid body: {e}")
            try:
                image_loader = ListPathImageLoader(
                    pi.image_paths,
                    settings.allowed_extensions,
                    settings.assets_dir,
                    max_files_per_run=settings.max_files_per_run,
                )
            except TooManyPathsError as e:
                raise HTTPException(413, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            input_kind, input_ref = "paths", f"{len(pi.image_paths)} paths"
            prompt = pi.prompt
        else:
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
```

- [ ] **Step 5: Run the existing SSE contract tests to make sure folder mode + multipart mode are still green**

```bash
PYTHONPATH=src .venv/bin/pytest tests/infrastructure/test_runs_stream.py -v
```

Expected: all 3 existing tests still pass (folder, multipart, traces).

- [ ] **Step 6: Commit**

```bash
git add src/infrastructure/http/routes_runs.py
git commit -m "feat(http): PathsInput model + dispatch for image_paths body"
```

### Task 2.2: SSE contract tests for `image_paths`

**Files:**
- Modify: `tests/infrastructure/test_runs_stream.py` (append)

The existing conftest sets `ASSETS_DIR=./tests/fixtures/folder`. There's one fixture image at `tests/fixtures/folder/fixture_a.png`. The happy-path test uses that file.

- [ ] **Step 1: Append 3 failing tests**

```python
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
```

- [ ] **Step 2: Run, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/infrastructure/test_runs_stream.py -v
```

Expected: all 7 tests pass (3 existing + 4 new).

If the happy-path test fails because `tests/fixtures/folder/fixture_a.png` doesn't exist in your working tree, verify with `ls tests/fixtures/folder/` — it should be present (was shipped with Phase 8.2's test fixtures). If missing, create it as a tiny valid PNG and `git add` it.

- [ ] **Step 3: Commit**

```bash
git add tests/infrastructure/test_runs_stream.py
git commit -m "test(http): SSE contract tests for image_paths mode"
```

---

## Phase 3 — Final verification

### Task 3.1: Full suite green

- [ ] **Step 1: Run the full non-e2e suite**

```bash
PYTHONPATH=src .venv/bin/pytest -v -m "not e2e"
```

Expected: 109 passed (100 baseline + 9 new: 7 loader + 4 SSE, minus any that overlap = 109 total), 1 skipped.

If the count is off, check:
- Are all 9 new tests discovered? `pytest --collect-only tests/infrastructure/test_list_path_loader.py tests/infrastructure/test_runs_stream.py -q`
- Any import errors from the new module? `python -c "from infrastructure.images.list_path_loader import ListPathImageLoader, TooManyPathsError; print('ok')"`

### Task 3.2: Mock-mode live smoke (optional, if environment allows)

- [ ] **Step 1: Start the server**

```bash
lsof -tiTCP:8001 -sTCP:LISTEN 2>/dev/null | xargs -r kill
LLM_MODE=mock PYTHONPATH=src .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 &
sleep 2
curl -s http://localhost:8001/health
```

Expected: `{"status":"ok","llm_mode":"mock"}`.

- [ ] **Step 2: Hit with an `image_paths` body**

```bash
curl -N -X POST http://localhost:8001/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["./assets/receipt_001.jpg"], "prompt": null}'
```

Expected: SSE stream ends with `final_result`. (If the DB URL in `.env` still points at the paused Supabase host, you'll see a 500 with the same DNS failure as before — not introduced by this change.)

- [ ] **Step 3: Hit with a bad path to confirm 400 is returned**

```bash
curl -X POST http://localhost:8001/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"image_paths": ["/etc/passwd"]}'
```

Expected: HTTP 400, body mentions "must be under".

- [ ] **Step 4: Stop the server**

```bash
lsof -tiTCP:8001 -sTCP:LISTEN 2>/dev/null | xargs -r kill
```

No commit for this task.

---

## Post-implementation checklist

- [ ] `PYTHONPATH=src .venv/bin/pytest -v -m "not e2e"` fully green.
- [ ] `image_paths` with valid paths reaches `ingest_node` → `final_result`.
- [ ] `image_paths` with any bad path returns 400 before `run_started` is emitted (no DB row written).
- [ ] Both `folder_path` and `image_paths` in one body returns 422.
- [ ] Neither field returns 422.
- [ ] More than `max_files_per_run` paths returns 413.
- [ ] Folder mode + multipart mode still work unchanged.

---

## Self-Review Notes

**Spec coverage check:**

| Spec section | Implementing task |
|---|---|
| §3 field name `image_paths` | 2.1 (PathsInput) |
| §3 coexistence → 422 | 2.1 (has_folder + has_paths branch) |
| §3 `assets_dir` guardrail | 1.1 (per-path startswith check) |
| §3 fail-fast | 1.1 (constructor-time validation) |
| §4.3 accumulate all failures | 1.1 (loop builds `bad` list; raises once) |
| §4.3 max_files_per_run | 1.3 (TooManyPathsError) |
| §4.3 empty list 400 | 1.2 (test_rejects_empty_list) |
| §4.4 ImageRef with original source_ref | 1.1 (source_ref=original) |
| §5 dispatch, handler flow | 2.1 |
| §6 HTTP status map (422/400/413) | 2.1 (except blocks) |
| §7.1 unit tests | 1.1–1.3 |
| §7.2 contract tests | 2.2 |
| §8 files touched | matches §8 table |
| §9 symlink traversal via `.resolve()` | 1.1 (Path(s).resolve() before startswith) |

All spec sections covered.

**Placeholder scan:** none. All tests have concrete assertions; all implementation code is complete.

**Type consistency:**
- `ListPathImageLoader.__init__` signature stable across tasks (Task 1.3 adds `max_files_per_run=None` as a new kwarg; earlier tests don't pass it, which is fine).
- `TooManyPathsError` subclass of `ValueError` — handler's `except TooManyPathsError as e` comes before `except ValueError as e`, so ordering is correct.
- `source_ref` = original user-supplied string; `local_path` = resolved absolute `Path`. Consistent across loader, tests, and spec §4.4.
- Field name `image_paths` used identically in Pydantic model, handler dispatch, tests, error messages.

**No spec requirement without a task.**

Plan is ready for execution.
