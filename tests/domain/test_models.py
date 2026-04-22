import pytest
from datetime import date as date_type
from decimal import Decimal
from uuid import uuid4
from domain.models import (
    AllowedCategory,
    Issue,
    Categorization,
    RawReceipt,
    NormalizedReceipt,
    Receipt,
    Aggregates,
    Report,
)


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


from domain.models import Anomaly


def test_anomaly_defaults_to_warning_severity():
    a = Anomaly(code="single_receipt_dominant", message="One receipt is 85% of total spend")
    assert a.code == "single_receipt_dominant"
    assert a.message.startswith("One receipt")
    assert a.severity == "warning"


def test_anomaly_accepts_notice_severity():
    a = Anomaly(code="currency_mix", message="Multiple currencies present", severity="notice")
    assert a.severity == "notice"


def test_receipt_accepts_filtered_status():
    from uuid import uuid4
    from domain.models import Receipt
    r = Receipt(id=uuid4(), source_ref="x.png", status="filtered")
    assert r.status == "filtered"
