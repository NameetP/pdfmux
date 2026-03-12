"""Schema-guided extraction — map extracted data to a JSON schema.

Pure rule-based mapping: no LLM, no cloud, no cost.
Uses fuzzy string matching to map extracted keys to schema fields,
then casts values to the schema's expected types.

Usage:
    schema = json.load(open("invoice.schema.json"))
    result = map_to_schema(tables, key_values, schema)
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from pdfmux.normalize import auto_normalize, normalize_amount, normalize_date
from pdfmux.types import ExtractedTable, KeyValuePair

# Minimum similarity for fuzzy key matching
_FUZZY_THRESHOLD = 0.6


def _similarity(a: str, b: str) -> float:
    """Fuzzy string similarity (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_match(key: str, candidates: list[str]) -> tuple[str, float] | None:
    """Find the best matching candidate for a key."""
    best = None
    best_score = 0.0

    for candidate in candidates:
        score = _similarity(key, candidate)
        if score > best_score:
            best_score = score
            best = candidate

    if best and best_score >= _FUZZY_THRESHOLD:
        return best, best_score
    return None


def _cast_value(value: Any, schema_type: str, schema_props: dict) -> Any:
    """Cast a value to the schema's expected type."""
    if value is None:
        return None

    if schema_type == "number" or schema_type == "integer":
        if isinstance(value, dict) and "amount" in value:
            return value["amount"]
        if isinstance(value, (int, float)):
            return value
        # Try to parse as number
        try:
            cleaned = str(value).replace(",", "").strip()
            cleaned = re.sub(r"[^\d.\-]", "", cleaned)
            if schema_type == "integer":
                return int(float(cleaned))
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    if schema_type == "string":
        fmt = schema_props.get("format")
        if fmt == "date":
            if isinstance(value, str) and re.match(r"\d{4}-\d{2}-\d{2}", value):
                return value
            result = normalize_date(str(value))
            return result if result else str(value)
        return str(value)

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        lower = str(value).lower()
        return lower in ("true", "yes", "1", "y")

    return value


def _extract_array_from_tables(
    tables: list[ExtractedTable],
    item_schema: dict,
) -> list[dict]:
    """Map table data to an array of objects per the schema."""
    if not tables:
        return []

    items_props = item_schema.get("properties", {})
    if not items_props:
        return []

    schema_fields = list(items_props.keys())
    results = []

    for table in tables:
        # Match table headers to schema fields
        header_mapping: dict[int, str] = {}  # col_index → schema_field
        for col_idx, header in enumerate(table.headers):
            match = _best_match(header, schema_fields)
            if match:
                header_mapping[col_idx] = match[0]

        if not header_mapping:
            continue

        # Map rows
        for row in table.rows:
            item = {}
            for col_idx, schema_field in header_mapping.items():
                if col_idx < len(row):
                    raw_value = row[col_idx]
                    field_schema = items_props[schema_field]
                    field_type = field_schema.get("type", "string")
                    item[schema_field] = _cast_value(
                        raw_value, field_type, field_schema
                    )

                    # Handle enum for direction/type fields
                    if "enum" in field_schema and item[schema_field] not in field_schema["enum"]:
                        # Try to infer from value
                        amount_info = normalize_amount(raw_value)
                        if amount_info and "direction" in amount_info:
                            direction = amount_info["direction"]
                            if direction in field_schema["enum"]:
                                item[schema_field] = direction

            if item:
                results.append(item)

    return results


def map_to_schema(
    tables: list[ExtractedTable],
    key_values: list[KeyValuePair],
    schema: dict,
) -> dict[str, Any]:
    """Map extracted data to a JSON schema using rule-based matching.

    Args:
        tables: Structured tables extracted from the document.
        key_values: Key-value pairs extracted from non-table regions.
        schema: JSON Schema defining the expected output structure.

    Returns:
        Dict conforming to the schema (best-effort, missing fields are None).
    """
    properties = schema.get("properties", {})
    result: dict[str, Any] = {}

    # Build a lookup from normalized key-value keys
    kv_by_key: dict[str, KeyValuePair] = {}
    for kv in key_values:
        kv_by_key[kv.key.lower()] = kv

    kv_keys = list(kv_by_key.keys())

    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "string")
        field_desc = field_schema.get("description", "")

        # Array fields → try to map from tables
        if field_type == "array":
            items_schema = field_schema.get("items", {})
            if items_schema.get("type") == "object":
                result[field_name] = _extract_array_from_tables(
                    tables, items_schema
                )
            continue

        # Object fields → recurse
        if field_type == "object":
            sub_props = field_schema.get("properties", {})
            sub_result = {}
            for sub_name, sub_schema in sub_props.items():
                sub_type = sub_schema.get("type", "string")
                # Try to find a KV match
                match = _best_match(sub_name, kv_keys)
                if not match:
                    match = _best_match(
                        sub_schema.get("description", sub_name), kv_keys
                    )
                if match:
                    kv = kv_by_key[match[0]]
                    normalized = auto_normalize(kv.key, kv.value)
                    sub_result[sub_name] = _cast_value(
                        normalized, sub_type, sub_schema
                    )
                else:
                    sub_result[sub_name] = None
            result[field_name] = sub_result
            continue

        # Scalar fields → match from key-values
        # Try field_name first, then description
        match = _best_match(field_name, kv_keys)
        if not match and field_desc:
            match = _best_match(field_desc, kv_keys)

        if match:
            kv = kv_by_key[match[0]]
            normalized = auto_normalize(kv.key, kv.value)
            result[field_name] = _cast_value(normalized, field_type, field_schema)
        else:
            result[field_name] = None

    return result


def load_schema(schema_path: str) -> dict:
    """Load a JSON schema from a file path or built-in preset name.

    Args:
        schema_path: Path to .json file or a preset name.

    Returns:
        Parsed JSON schema dict.
    """
    import os

    if os.path.isfile(schema_path):
        with open(schema_path) as f:
            return json.load(f)

    # Check for built-in presets
    presets_dir = os.path.join(os.path.dirname(__file__), "schemas")
    preset_path = os.path.join(presets_dir, f"{schema_path}.json")
    if os.path.isfile(preset_path):
        with open(preset_path) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Schema not found: {schema_path}. "
        "Provide a file path or a built-in preset name."
    )
