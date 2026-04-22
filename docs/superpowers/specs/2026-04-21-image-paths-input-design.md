# `image_paths` Input Mode — Design

**Date:** 2026-04-21
**Status:** Approved by user; ready for implementation planning.
**Builds on:** `docs/superpowers/specs/2026-04-20-receipt-processing-agent-design.md` (endpoint contract) and `docs/superpowers/specs/2026-04-21-agentic-graph-design.md` (graph topology — unchanged by this spec).

## 1. Problem

`POST /runs/stream` currently accepts two input shapes: a single `folder_path` (JSON body) or a multipart file upload. There is no way to pass an **explicit list of image paths** that are already on the server's filesystem (e.g. under `./assets/`). The goal of this change is to add a third input mode without changing the endpoint URL, SSE event contract, or downstream graph code.

Target body shape:

```json
{
  "image_paths": ["./assets/receipt_001.jpg", "./assets/receipt_007.png"],
  "prompt": "focus on restaurants"
}
```

## 2. Non-goals

- No new endpoint URL.
- No changes to the graph, tools, agents, SSE event types, or DB schema.
- No change to the multipart-upload mode.
- No support for paths outside the configured `assets_dir`.
- No partial runs: a single bad path rejects the whole request.

## 3. Decisions locked during brainstorm

| Decision | Value | Rationale |
|---|---|---|
| Security guardrail | Every path must resolve under `settings.assets_dir` | Matches existing folder-mode posture; keeps attack surface identical |
| Bad path handling | Fail fast, all-or-nothing, 400 before the graph runs | Explicit list is a contract; silent drops are surprising |
| Field name | `image_paths` | Unambiguous; matches user's example |
| Body coexistence | Require exactly one of `folder_path` / `image_paths`; reject both/neither with 422 | Clear contract, easy to error-message |
| `prompt` | Optional on either mode | Consistent with folder-mode behavior |
| Structural approach | Two sibling Pydantic models + handler-side discrimination | Single-purpose models, transparent routing, no refactor of existing code |

## 4. Architecture

### 4.1 Components

**Existing (unchanged):**
- `FolderInput(folder_path: str, prompt: str | None)` in `src/infrastructure/http/routes_runs.py`
- `LocalFolderImageLoader` in `src/infrastructure/images/folder_loader.py`
- `UploadImageLoader` in `src/infrastructure/images/upload_loader.py`
- `ImageLoaderPort` abstract interface in `src/application/ports.py`

**New:**
- `PathsInput(image_paths: list[str], prompt: str | None)` — Pydantic model alongside `FolderInput`
- `ListPathImageLoader` — implements `ImageLoaderPort`; validates paths in `__init__`, returns pre-built `ImageRef`s in `load()`
- One new branch in `post_runs_stream` handler for the JSON-body path

### 4.2 Handler dispatch (JSON branch)

```python
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
    except ValidationError as e:
        raise HTTPException(422, f"invalid body: {e}")
    try:
        image_loader = ListPathImageLoader(
            pi.image_paths, settings.allowed_extensions, settings.assets_dir,
        )
    except TooManyPathsError as e:
        raise HTTPException(413, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    input_kind, input_ref = "paths", f"{len(pi.image_paths)} paths"
    prompt = pi.prompt
else:
    # existing folder flow — unchanged
```

### 4.3 `ListPathImageLoader` validation order

Per-path checks in one pass, accumulating all failures before raising (so clients see every bad path, not just the first):

1. Resolve the path (`Path(s).resolve()`).
2. Check it starts with `assets_dir.resolve()` → else "path must be under ASSETS_DIR".
3. Check existence → else "path does not exist".
4. Check it's a file (not a directory) → else "path is not a file".
5. Check extension is in `allowed_extensions` → else "extension .{ext} not in allowed extensions".
6. Check read permission via `os.access(p, os.R_OK)` → else "path is not readable".

After per-path validation, enforce `len(refs) <= settings.max_files_per_run`. If violated, raise `TooManyPathsError` (a subclass of `ValueError`) so the handler can distinguish and map to 413.

Empty list is a separate upfront check with its own message: `"image_paths must not be empty"` → 400.

### 4.4 `ImageRef` construction

```python
ImageRef(source_ref=original_path_as_given, local_path=resolved_path)
```

`source_ref` is the path string the client supplied (preserves what the user sees in traces). `local_path` is the fully resolved absolute path used by OCR.

## 5. Data flow

1. Client POSTs JSON body with `image_paths` and optional `prompt`.
2. Handler dispatches: detects `image_paths` key, validates body via Pydantic, constructs `ListPathImageLoader`.
3. Loader validates every path up front; on any failure, raises before the graph runs. No `run_started` event is emitted; no DB row is written.
4. On success, the handler proceeds identically to folder mode: inserts a `reports` row with `input_kind="paths"`, builds the `GraphRunner`, invokes `build_graph`, streams SSE.
5. `ingest_node` calls `load_images`, which invokes `ListPathImageLoader.load()`. The loader returns the pre-built `ImageRef` list — no further I/O.
6. The rest of the run (per-receipt processing, finalize) is unchanged.

## 6. Error handling

| Failure | HTTP status | Where | Message shape |
|---|---|---|---|
| Body parse (missing `image_paths`, wrong types) | 422 | `PathsInput(**body)` | `"invalid body: ..."` |
| Both `folder_path` and `image_paths` present | 422 | Handler | `"provide exactly one of folder_path or image_paths"` |
| Neither field present | 422 | Handler | `"body must include folder_path or image_paths"` |
| Empty `image_paths` list | 400 | Loader ctor | `"image_paths must not be empty"` |
| Any path outside `assets_dir` / missing / not a file / bad extension / unreadable | 400 | Loader ctor | `"invalid image_paths:\n  <path>: <reason>\n  ..."` |
| More than `max_files_per_run` paths | 413 | Loader ctor | `"too many paths (max N)"` |

All errors surface **before** any graph work — no SSE events, no DB writes, no OCR calls. Once the graph starts, the run follows the existing error bands (tool-attempt / receipt-level / run-level) and produces the same event types as before.

## 7. Testing

### 7.1 Unit tests for `ListPathImageLoader`

New file `tests/infrastructure/test_list_path_loader.py`:

- `test_loads_valid_paths_under_assets` — 2 valid paths → `len(refs) == 2`, `source_ref` equals the input string.
- `test_rejects_path_outside_assets` — `/etc/passwd` → `ValueError` mentioning "must be under".
- `test_rejects_missing_file` → `ValueError` mentioning "does not exist".
- `test_rejects_disallowed_extension` — `.exe` under tmp assets dir → `ValueError` mentioning "extension".
- `test_rejects_empty_list` → `ValueError("image_paths must not be empty")`.
- `test_aggregates_all_failures` — 3 bad paths mixed → error message contains all three.
- `test_max_files_per_run_enforced` — count > limit → `TooManyPathsError` with "too many paths".

### 7.2 SSE contract tests

Append to `tests/infrastructure/test_runs_stream.py`:

- `test_image_paths_happy_path(client1)` — JSON body `{"image_paths": ["./tests/fixtures/folder/r1.png"]}` → SSE stream has `run_started` → `receipt_result` → `final_result`.
- `test_image_paths_bad_path_returns_400` — non-existent path → HTTP 400 with the path in the error body.
- `test_image_paths_and_folder_path_both_returns_422` — both fields present → HTTP 422.

Reuses the existing `client1` fixture (1-receipt-sized mock script).

### 7.3 Tests not needing change

- `tests/application/test_graph.py` — graph is agnostic to the loader source.
- Tool registry tests — no new tools.
- Chat model / agent tests — unchanged.

## 8. Files touched

- **Modify:** `src/infrastructure/http/routes_runs.py` — add `PathsInput` model; add JSON-branch dispatch block for `image_paths`.
- **New:** `src/infrastructure/images/list_path_loader.py` — `ListPathImageLoader` + `TooManyPathsError`.
- **New:** `tests/infrastructure/test_list_path_loader.py` — unit tests above.
- **Modify:** `tests/infrastructure/test_runs_stream.py` — three new contract tests.

Nothing else touched.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| A path traversal via symlink points outside `assets_dir` | `.resolve()` follows symlinks to their target; the subsequent `startswith(assets_dir.resolve())` check runs on the resolved absolute path, so symlinked escapes are caught |
| A client submits 10K paths | `max_files_per_run` check catches this before validation explodes |
| A path is readable when validated but deleted before OCR runs | Out of scope here — would surface as a receipt-level OCR error via existing tool-retry / skip machinery. Validation is best-effort up-front. |
| Pydantic emits a confusing error when `image_paths` isn't a list of strings | 422 passes the raw `ValidationError` message through; acceptable for a local dev contract |

## 10. Open questions

None. All resolved during brainstorm (see §3).
