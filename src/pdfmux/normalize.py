"""Value normalization — parse dates, amounts, and currencies.

Converts messy PDF text values into clean, typed data.
Pure Python, zero external dependencies.

Examples:
    "28 Feb 2026"      → "2026-02-28"
    "AED 1,234.50 DR"  → {"amount": 1234.50, "direction": "debit", "currency": "AED"}
    "3.49% per month"  → {"rate": 3.49, "period": "monthly"}
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Common date patterns
_DATE_PATTERNS = [
    # DD MMM YYYY — "28 Feb 2026", "1 January 2026"
    re.compile(
        r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3,9})\s+(?P<year>\d{4})"
    ),
    # MMM DD, YYYY — "February 28, 2026"
    re.compile(
        r"(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
    ),
    # DD/MM/YYYY or DD-MM-YYYY
    re.compile(
        r"(?P<day>\d{1,2})[/\-](?P<month>\d{1,2})[/\-](?P<year>\d{4})"
    ),
    # YYYY-MM-DD (ISO)
    re.compile(
        r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
    ),
    # DD MMM (no year) — "01 Feb", "15 Mar"
    re.compile(
        r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3,9})$"
    ),
    # DD-MMM-YY — "01-Feb-26"
    re.compile(
        r"(?P<day>\d{1,2})-(?P<month>[A-Za-z]{3,9})-(?P<year>\d{2})"
    ),
]


def normalize_date(value: str, default_year: int | None = None) -> str | None:
    """Normalize a date string to ISO 8601 (YYYY-MM-DD).

    Args:
        value: Raw date string from PDF.
        default_year: Year to use if not present in value.

    Returns:
        ISO date string or None if parsing fails.
    """
    value = value.strip()
    if default_year is None:
        default_year = datetime.now().year

    for pattern in _DATE_PATTERNS:
        match = pattern.search(value)
        if not match:
            continue

        groups = match.groupdict()
        day = int(groups["day"])

        # Parse month
        month_raw = groups["month"]
        if month_raw.isdigit():
            month = int(month_raw)
        else:
            month = _MONTH_MAP.get(month_raw.lower())
            if month is None:
                continue

        # Parse year
        year_raw = groups.get("year")
        if year_raw is None:
            year = default_year
        elif len(year_raw) == 2:
            year = 2000 + int(year_raw)
        else:
            year = int(year_raw)

        # Validate
        if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100):
            continue

        try:
            datetime(year, month, day)
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            continue

    return None


# ---------------------------------------------------------------------------
# Amount normalization
# ---------------------------------------------------------------------------

# Currency symbols and codes
_CURRENCY_PATTERNS = re.compile(
    r"(?P<currency>AED|USD|EUR|GBP|INR|SAR|QAR|BHD|KWD|OMR|"
    r"Dhs|Rs|£|\$|€|¥|د\.إ)\s*",
    re.IGNORECASE,
)

# Direction indicators
_DEBIT_INDICATORS = {"dr", "dr.", "debit", "d", "-", "withdrawal"}
_CREDIT_INDICATORS = {"cr", "cr.", "credit", "c", "+", "deposit", "payment"}


def normalize_amount(value: str) -> dict[str, Any] | None:
    """Normalize a monetary amount string.

    Args:
        value: Raw amount string like "AED 1,234.50 DR" or "(1,234.50)".

    Returns:
        Dict with keys: amount (float), direction (debit/credit/unknown),
        currency (str or None), raw (str). Or None if parsing fails.
    """
    raw = value.strip()
    if not raw:
        return None

    # Extract currency
    currency = None
    currency_match = _CURRENCY_PATTERNS.search(raw)
    if currency_match:
        currency = currency_match.group("currency").upper()
        # Normalize common symbols
        symbol_map = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
                      "DHS": "AED", "RS": "INR", "د.إ": "AED"}
        currency = symbol_map.get(currency, currency)

    # Detect direction
    direction = "unknown"
    lower = raw.lower()

    # Check for parentheses (accounting negative)
    if "(" in raw and ")" in raw:
        direction = "debit"

    # Check for explicit direction words
    for word in lower.split():
        word_clean = word.strip(".,;:()")
        if word_clean in _DEBIT_INDICATORS:
            direction = "debit"
            break
        if word_clean in _CREDIT_INDICATORS:
            direction = "credit"
            break

    # Leading minus
    amount_str = raw
    if re.match(r"^\s*-", amount_str):
        direction = "debit"

    # Extract the numeric part
    # Remove currency, direction words, parentheses
    cleaned = _CURRENCY_PATTERNS.sub("", amount_str)
    cleaned = re.sub(r"(?i)\b(dr|cr|debit|credit|d|c)\b\.?", "", cleaned)
    cleaned = cleaned.replace("(", "").replace(")", "")
    cleaned = cleaned.strip(" \t-+")

    # Handle European format: 1.234,50 → 1234.50
    if re.match(r"^\d{1,3}(\.\d{3})+(,\d{2})?$", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        # Standard format: remove commas
        cleaned = cleaned.replace(",", "")

    # Parse float
    try:
        amount = float(cleaned)
    except (ValueError, TypeError):
        return None

    if amount < 0:
        amount = abs(amount)
        direction = "debit"

    return {
        "amount": round(amount, 2),
        "direction": direction,
        "currency": currency,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Rate normalization
# ---------------------------------------------------------------------------


def normalize_rate(value: str) -> dict[str, Any] | None:
    """Normalize a rate/percentage string.

    Args:
        value: Raw rate string like "3.49% per month" or "41.88% p.a.".

    Returns:
        Dict with keys: rate (float), period (monthly/annual/unknown).
        Or None if parsing fails.
    """
    raw = value.strip()
    if not raw:
        return None

    # Extract percentage
    rate_match = re.search(r"(\d+\.?\d*)\s*%", raw)
    if not rate_match:
        return None

    rate = float(rate_match.group(1))

    # Detect period
    lower = raw.lower()
    if any(w in lower for w in ("month", "p.m.", "per month", "monthly")):
        period = "monthly"
    elif any(w in lower for w in ("year", "annual", "p.a.", "per annum", "yearly")):
        period = "annual"
    else:
        period = "unknown"

    return {"rate": rate, "period": period, "raw": raw}


# ---------------------------------------------------------------------------
# Auto-normalize a key-value pair
# ---------------------------------------------------------------------------

# Keys that suggest date values
_DATE_KEYS = {
    "date", "statement date", "due date", "payment due date",
    "from", "to", "period", "issued", "expiry", "effective date",
    "invoice date", "order date", "report date",
}

# Keys that suggest amount values
_AMOUNT_KEYS = {
    "balance", "outstanding balance", "total outstanding", "amount",
    "credit limit", "available credit", "minimum payment",
    "minimum payment due", "minimum amount due", "total amount due",
    "previous balance", "new balance", "closing balance", "opening balance",
    "available balance", "total", "subtotal", "tax", "fee",
    "finance charge", "finance charges", "interest charge",
    "total spend", "total payments", "net amount",
}

# Keys that suggest rate values
_RATE_KEYS = {
    "interest rate", "rate", "apr", "annual percentage rate",
    "finance charges rate", "monthly percentage rate",
}


def auto_normalize(key: str, value: str) -> Any:
    """Auto-detect the value type from the key name and normalize.

    Returns the normalized value or the original string if no
    normalization applies.
    """
    key_lower = key.lower().strip()

    # Check for rate first (before amount — rates also have numbers)
    if key_lower in _RATE_KEYS or "rate" in key_lower or "%" in value:
        result = normalize_rate(value)
        if result:
            return result

    # Check for amount
    if key_lower in _AMOUNT_KEYS or any(k in key_lower for k in ("amount", "balance", "total", "fee", "charge", "payment", "limit")):
        result = normalize_amount(value)
        if result:
            return result

    # Check for date
    if key_lower in _DATE_KEYS or "date" in key_lower:
        result = normalize_date(value)
        if result:
            return result

    return value
