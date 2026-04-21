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
{"event_type":"progress","run_id":"...","seq":2,"ts":"...",
 "step":"ocr|normalize|categorize|load_images|aggregate|generate_report",
 "receipt_id":"optional","i":2,"n":5}
```

### `tool_call`
```json
{"event_type":"tool_call","run_id":"...","seq":3,"ts":"...",
 "tool":"extract_receipt_fields","receipt_id":"optional","attempt":1,
 "args":{"image":"..."}}
```

### `tool_result`
```json
{"event_type":"tool_result","run_id":"...","seq":4,"ts":"...",
 "tool":"extract_receipt_fields","receipt_id":"optional",
 "result_summary":{"vendor":"Uber","has_total":true,"ocr_confidence":0.92},
 "error":false,"error_message":null,"duration_ms":412}
```

### `receipt_result`
Terminal event for each receipt. Always emitted, even on error.
```json
{"event_type":"receipt_result","run_id":"...","seq":5,"ts":"...",
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
