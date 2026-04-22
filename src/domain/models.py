from datetime import date
from decimal import Decimal
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


class Anomaly(BaseModel):
    code: str
    message: str
    severity: Literal["warning", "notice"] = "warning"
