"""
Pure normalization functions.

parse_date / parse_money tolerate None and raise ValueError on unparseable non-empty input.
`normalize(raw)` always returns a NormalizedReceipt; on parse failure it raises so the
tool wrapper can classify the error at the boundary.
"""
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from dateutil.parser import parse as _dateutil_parse, ParserError as _DateutilParserError
from domain.models import NormalizedReceipt, RawReceipt


_ISO_CURRENCIES: frozenset[str] = frozenset({
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF",
    "CNY", "HKD", "SEK", "NOK", "DKK", "NZD", "MXN",
})
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y")
_CURRENCY_SIGN = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_CURRENCY_RE = re.compile(r"\b([A-Z]{3})\b")
_MONEY_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def parse_date(raw: str | None) -> date | None:
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Fallback: dateutil handles 2-digit years, abbreviated months, dot
    # separators, trailing time/AM-PM, and locale quirks like "p. m.".
    # fuzzy=True tolerates extra characters; we still raise our own
    # ValueError on genuine noise.
    try:
        return _dateutil_parse(text, fuzzy=True).date()
    except (_DateutilParserError, ValueError, OverflowError, TypeError) as e:
        raise ValueError(f"unparseable date: {raw!r}") from e


def parse_money(raw: str | None) -> tuple[Decimal | None, str | None]:
    if raw is None or not raw.strip():
        return None, None
    text = raw.strip()

    # currency from sign
    currency: str | None = None
    for sign, iso in _CURRENCY_SIGN.items():
        if sign in text:
            currency = iso
            text = text.replace(sign, "")
            break

    # currency from ISO-3 code (must be in allowlist to avoid matching
    # arbitrary uppercase trigrams like "TAX" or "VAT")
    m = _CURRENCY_RE.search(text)
    if m and m.group(1) in _ISO_CURRENCIES:
        currency = currency or m.group(1)
        text = text.replace(m.group(1), "")

    # number
    nm = _MONEY_RE.search(text.replace(",", ""))
    if not nm:
        raise ValueError(f"unparseable money: {raw!r}")
    try:
        amount = Decimal(nm.group(0))
    except InvalidOperation as e:
        raise ValueError(f"unparseable money: {raw!r}") from e
    return amount, currency


def normalize(raw: RawReceipt) -> NormalizedReceipt:
    d = parse_date(raw.receipt_date)
    amount, currency_from_total = parse_money(raw.total_raw)
    currency = raw.currency_raw or currency_from_total or "USD"
    return NormalizedReceipt(
        source_ref=raw.source_ref,
        vendor=raw.vendor,
        receipt_date=d,
        receipt_number=raw.receipt_number,
        total=amount,
        currency=currency,
    )
