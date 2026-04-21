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
