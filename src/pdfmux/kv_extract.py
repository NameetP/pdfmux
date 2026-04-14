"""Key-value pair extraction — detect label: value patterns in text.

Handles the scattered metadata common in bank statements, invoices,
government forms, and reports. Pure regex + heuristics, zero deps.

Patterns detected:
    - "Label: Value"           (colon-separated)
    - "Label    Value"         (tab/whitespace-aligned)
    - "Label............Value" (dot-leader)
"""

from __future__ import annotations

import re

from pdfmux.types import KeyValuePair

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Colon-separated: "Statement Date: 28 Feb 2026"
_COLON_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?P<key>[A-Z][A-Za-z0-9 /&\-'()]{2,40}?)"
    r"\s*:\s*"
    r"(?P<value>.+?)$",
    re.MULTILINE,
)

# Dot-leader: "Account Number.......1234567"
_DOT_LEADER_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?P<key>[A-Z][A-Za-z0-9 /&\-'()]{2,40}?)"
    r"\.{3,}\s*"
    r"(?P<value>.+?)$",
    re.MULTILINE,
)

# Whitespace-aligned: "Total Due          AED 5,000.00"
# Only match when there are 4+ spaces between key and value
_WHITESPACE_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?P<key>[A-Z][A-Za-z0-9 /&\-'()]{2,40}?)"
    r"[ \t]{4,}"
    r"(?P<value>\S.{1,60}?)$",
    re.MULTILINE,
)

# Known labels that are commonly key-value pairs
_KNOWN_LABELS = {
    "statement date",
    "card number",
    "account number",
    "credit limit",
    "outstanding balance",
    "total outstanding",
    "minimum payment",
    "minimum payment due",
    "minimum amount due",
    "payment due date",
    "due date",
    "previous balance",
    "interest rate",
    "annual percentage rate",
    "apr",
    "finance charge",
    "finance charges",
    "new balance",
    "closing balance",
    "opening balance",
    "available credit",
    "available balance",
    "total amount due",
    "date",
    "invoice number",
    "invoice date",
    "order number",
    "customer id",
    "patient name",
    "patient id",
    "report date",
    "page",
}


def extract_key_values(
    text: str,
    page_num: int = 0,
) -> list[KeyValuePair]:
    """Extract key-value pairs from page text.

    Applies multiple patterns and deduplicates by key.
    Known financial/document labels are prioritized.

    Args:
        text: The page text to extract from.
        page_num: 0-indexed page number for metadata.

    Returns:
        List of KeyValuePair objects.
    """
    candidates: dict[str, KeyValuePair] = {}

    for pattern in (_COLON_PATTERN, _DOT_LEADER_PATTERN, _WHITESPACE_PATTERN):
        for match in pattern.finditer(text):
            key = match.group("key").strip()
            value = match.group("value").strip()

            # Skip if key is too short or value is empty
            if len(key) < 2 or not value:
                continue

            # Skip if value looks like a table row (multiple | chars)
            if value.count("|") >= 2:
                continue

            # Skip if key is all caps and very long (likely a heading)
            if key.isupper() and len(key) > 30:
                continue

            # Normalize key for dedup
            norm_key = key.lower().strip()

            # If we already have this key, prefer known-label matches
            if norm_key in candidates:
                existing = candidates[norm_key]
                # Keep the one with a more complete value
                if len(value) > len(existing.value):
                    candidates[norm_key] = KeyValuePair(key=key, value=value, page_num=page_num)
            else:
                candidates[norm_key] = KeyValuePair(key=key, value=value, page_num=page_num)

    # Sort: known labels first, then by position in text
    result = sorted(
        candidates.values(),
        key=lambda kv: (
            0 if kv.key.lower() in _KNOWN_LABELS else 1,
            text.index(kv.key) if kv.key in text else 999,
        ),
    )

    return result
