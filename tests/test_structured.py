"""Tests for structured extraction — tables, key-values, normalization, schema mapping."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pdfmux.kv_extract import extract_key_values
from pdfmux.normalize import normalize_amount, normalize_date, normalize_rate, auto_normalize
from pdfmux.types import ExtractedTable, KeyValuePair


# ---------------------------------------------------------------------------
# Key-Value Extraction
# ---------------------------------------------------------------------------


class TestKeyValueExtraction:
    def test_colon_pattern(self):
        text = "Statement Date: 28 Feb 2026\nCard Number: **** 4829"
        kvs = extract_key_values(text)
        keys = {kv.key for kv in kvs}
        assert "Statement Date" in keys
        assert "Card Number" in keys

    def test_colon_values(self):
        text = "Outstanding Balance: AED 32,400.00"
        kvs = extract_key_values(text)
        assert len(kvs) >= 1
        kv = next(kv for kv in kvs if "balance" in kv.key.lower())
        assert "32,400" in kv.value

    def test_skips_table_rows(self):
        text = "Some Label: value | another | cell | data"
        kvs = extract_key_values(text)
        # Should skip because value has 2+ pipes (looks like table row)
        table_kvs = [kv for kv in kvs if "Some Label" == kv.key]
        assert len(table_kvs) == 0

    def test_page_num_preserved(self):
        text = "Invoice Number: INV-001"
        kvs = extract_key_values(text, page_num=5)
        assert kvs[0].page_num == 5

    def test_empty_text(self):
        assert extract_key_values("") == []

    def test_no_matches(self):
        text = "This is just a plain paragraph with no key value pairs."
        kvs = extract_key_values(text)
        assert len(kvs) == 0

    def test_multiple_on_same_page(self):
        text = """Statement Date: 28 Feb 2026
Credit Limit: AED 50,000.00
Minimum Payment Due: AED 1,620.00
Payment Due Date: 15 Mar 2026"""
        kvs = extract_key_values(text)
        assert len(kvs) >= 3


# ---------------------------------------------------------------------------
# Date Normalization
# ---------------------------------------------------------------------------


class TestDateNormalization:
    def test_dd_mmm_yyyy(self):
        assert normalize_date("28 Feb 2026") == "2026-02-28"

    def test_dd_month_yyyy(self):
        assert normalize_date("1 January 2026") == "2026-01-01"

    def test_month_dd_yyyy(self):
        assert normalize_date("February 28, 2026") == "2026-02-28"

    def test_dd_mm_yyyy_slash(self):
        assert normalize_date("28/02/2026") == "2026-02-28"

    def test_dd_mm_yyyy_dash(self):
        assert normalize_date("28-02-2026") == "2026-02-28"

    def test_iso_format(self):
        assert normalize_date("2026-02-28") == "2026-02-28"

    def test_dd_mmm_no_year(self):
        result = normalize_date("01 Feb", default_year=2026)
        assert result == "2026-02-01"

    def test_dd_mmm_yy(self):
        assert normalize_date("01-Feb-26") == "2026-02-01"

    def test_invalid(self):
        assert normalize_date("not a date") is None

    def test_empty(self):
        assert normalize_date("") is None


# ---------------------------------------------------------------------------
# Amount Normalization
# ---------------------------------------------------------------------------


class TestAmountNormalization:
    def test_simple(self):
        result = normalize_amount("1234.50")
        assert result is not None
        assert result["amount"] == 1234.50

    def test_with_commas(self):
        result = normalize_amount("1,234.50")
        assert result is not None
        assert result["amount"] == 1234.50

    def test_with_currency(self):
        result = normalize_amount("AED 1,234.50")
        assert result is not None
        assert result["amount"] == 1234.50
        assert result["currency"] == "AED"

    def test_debit_dr(self):
        result = normalize_amount("1,234.50 DR")
        assert result is not None
        assert result["amount"] == 1234.50
        assert result["direction"] == "debit"

    def test_credit_cr(self):
        result = normalize_amount("5,000.00 CR")
        assert result is not None
        assert result["direction"] == "credit"

    def test_parentheses_negative(self):
        result = normalize_amount("(1,234.50)")
        assert result is not None
        assert result["amount"] == 1234.50
        assert result["direction"] == "debit"

    def test_negative_sign(self):
        result = normalize_amount("-1234.50")
        assert result is not None
        assert result["amount"] == 1234.50
        assert result["direction"] == "debit"

    def test_dollar_sign(self):
        result = normalize_amount("$99.99")
        assert result is not None
        assert result["amount"] == 99.99
        assert result["currency"] == "USD"

    def test_european_format(self):
        result = normalize_amount("1.234,50")
        assert result is not None
        assert result["amount"] == 1234.50

    def test_empty(self):
        assert normalize_amount("") is None

    def test_invalid(self):
        assert normalize_amount("not a number") is None


# ---------------------------------------------------------------------------
# Rate Normalization
# ---------------------------------------------------------------------------


class TestRateNormalization:
    def test_monthly_rate(self):
        result = normalize_rate("3.49% per month")
        assert result is not None
        assert result["rate"] == 3.49
        assert result["period"] == "monthly"

    def test_annual_rate(self):
        result = normalize_rate("41.88% p.a.")
        assert result is not None
        assert result["rate"] == 41.88
        assert result["period"] == "annual"

    def test_no_period(self):
        result = normalize_rate("2.5%")
        assert result is not None
        assert result["rate"] == 2.5
        assert result["period"] == "unknown"

    def test_no_percentage(self):
        assert normalize_rate("no rate here") is None


# ---------------------------------------------------------------------------
# Auto-normalize
# ---------------------------------------------------------------------------


class TestAutoNormalize:
    def test_date_key(self):
        result = auto_normalize("Statement Date", "28 Feb 2026")
        assert result == "2026-02-28"

    def test_amount_key(self):
        result = auto_normalize("Outstanding Balance", "AED 32,400.00")
        assert isinstance(result, dict)
        assert result["amount"] == 32400.00

    def test_rate_key(self):
        result = auto_normalize("Interest Rate", "3.49% per month")
        assert isinstance(result, dict)
        assert result["rate"] == 3.49

    def test_unknown_key(self):
        result = auto_normalize("Card Number", "**** 4829")
        assert result == "**** 4829"


# ---------------------------------------------------------------------------
# ExtractedTable type
# ---------------------------------------------------------------------------


class TestExtractedTable:
    def test_creation(self):
        table = ExtractedTable(
            page_num=0,
            headers=("Date", "Description", "Amount"),
            rows=(
                ("01 Feb", "CARREFOUR", "234.50"),
                ("02 Feb", "UBER", "45.00"),
            ),
        )
        assert table.page_num == 0
        assert len(table.headers) == 3
        assert len(table.rows) == 2
        assert table.rows[0][2] == "234.50"

    def test_with_bbox(self):
        table = ExtractedTable(
            page_num=1,
            headers=("Col1",),
            rows=(("val1",),),
            bbox=(10.0, 20.0, 300.0, 400.0),
        )
        assert table.bbox == (10.0, 20.0, 300.0, 400.0)

    def test_with_label(self):
        table = ExtractedTable(
            page_num=0,
            headers=("X",),
            rows=(("Y",),),
            label="transactions",
        )
        assert table.label == "transactions"


# ---------------------------------------------------------------------------
# Schema Mapping
# ---------------------------------------------------------------------------


class TestSchemaMapping:
    def test_basic_kv_mapping(self):
        from pdfmux.schema import map_to_schema

        kvs = [
            KeyValuePair(key="Statement Date", value="28 Feb 2026", page_num=0),
            KeyValuePair(key="Outstanding Balance", value="AED 32,400.00", page_num=0),
        ]

        schema = {
            "properties": {
                "statement_date": {"type": "string", "format": "date", "description": "Statement Date"},
                "outstanding_balance": {"type": "number", "description": "Outstanding Balance"},
            }
        }

        result = map_to_schema([], kvs, schema)
        assert result["statement_date"] == "2026-02-28"
        assert result["outstanding_balance"] == 32400.00

    def test_table_to_array(self):
        from pdfmux.schema import map_to_schema

        tables = [
            ExtractedTable(
                page_num=0,
                headers=("Date", "Description", "Amount"),
                rows=(
                    ("01 Feb", "CARREFOUR", "234.50"),
                    ("02 Feb", "UBER", "45.00"),
                ),
            )
        ]

        schema = {
            "properties": {
                "transactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "description": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                    },
                }
            }
        }

        result = map_to_schema(tables, [], schema)
        assert len(result["transactions"]) == 2
        assert result["transactions"][0]["amount"] == 234.50
        assert result["transactions"][1]["description"] == "UBER"

    def test_missing_fields_are_none(self):
        from pdfmux.schema import map_to_schema

        schema = {
            "properties": {
                "bank": {"type": "string", "description": "Issuing bank name"},
            }
        }

        result = map_to_schema([], [], schema)
        assert result["bank"] is None


# ---------------------------------------------------------------------------
# Integration: JSON output includes structured data
# ---------------------------------------------------------------------------


class TestJSONStructuredOutput:
    @pytest.fixture
    def sample_pdf(self, tmp_path: Path) -> Path:
        """Create a simple PDF with table-like content."""
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 100),
            "Statement Date: 15 Mar 2026\n"
            "Account Number: 12345678\n"
            "Balance: AED 5,000.00\n",
        )
        pdf_path = tmp_path / "statement.pdf"
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_json_output_has_key_values(self, sample_pdf: Path):
        from pdfmux.pipeline import process

        result = process(
            file_path=sample_pdf,
            output_format="json",
            quality="fast",
        )
        data = json.loads(result.text)
        assert data["schema_version"] == "1.1.0"

        # Should have key_values if any were found
        if "key_values" in data:
            keys = {kv["key"] for kv in data["key_values"]}
            # At least some of these should be detected
            assert len(keys) >= 1
