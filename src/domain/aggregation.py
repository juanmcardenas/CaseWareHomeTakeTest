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
