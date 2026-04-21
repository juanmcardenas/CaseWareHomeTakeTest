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
