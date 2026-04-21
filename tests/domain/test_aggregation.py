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


def test_aggregate_sum_then_round_vs_per_item_round():
    """
    Three items at 0.005 each:
      per-item rounded (each 0.01) → sum = 0.03
      sum-then-round (0.015)       → 0.02
    We sum first, THEN round, so expect 0.02.
    """
    receipts = [
        _receipt(AllowedCategory.TRAVEL, Decimal("0.005")),
        _receipt(AllowedCategory.TRAVEL, Decimal("0.005")),
        _receipt(AllowedCategory.TRAVEL, Decimal("0.005")),
    ]
    a = aggregate(receipts)
    assert a.total_spend == Decimal("0.02")  # 0.015 → 0.02 with ROUND_HALF_UP
    assert a.by_category[AllowedCategory.TRAVEL.value] == Decimal("0.02")
