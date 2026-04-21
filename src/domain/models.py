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
