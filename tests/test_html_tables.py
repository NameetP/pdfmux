"""Tests for HTML table output in the fast extractor."""

from __future__ import annotations

from pdfmux.extractors.fast import _build_html_table, _escape_html


class TestEscapeHtml:
    """Tests for HTML escaping helper."""

    def test_ampersand(self):
        assert _escape_html("A & B") == "A &amp; B"

    def test_angle_brackets(self):
        assert _escape_html("<script>") == "&lt;script&gt;"

    def test_no_escaping_needed(self):
        assert _escape_html("hello world") == "hello world"

    def test_empty_string(self):
        assert _escape_html("") == ""


class TestBuildHtmlTable:
    """Tests for the HTML table builder."""

    def test_basic_table(self):
        headers = ("Name", "Age")
        rows = (("Alice", "25"), ("Bob", "30"))
        result = _build_html_table(headers, rows)
        assert "<table>" in result
        assert "</table>" in result
        assert "<th>Name</th>" in result
        assert "<th>Age</th>" in result
        assert "<td>Alice</td>" in result
        assert "<td>25</td>" in result
        assert "<td>Bob</td>" in result
        assert "<td>30</td>" in result

    def test_single_row(self):
        headers = ("Col1", "Col2")
        rows = (("A", "B"),)
        result = _build_html_table(headers, rows)
        assert result.count("<tr>") == 2  # 1 header + 1 data row

    def test_empty_rows(self):
        headers = ("Col1", "Col2")
        rows = ()
        result = _build_html_table(headers, rows)
        assert result.count("<tr>") == 1  # header row only

    def test_html_escaping_in_cells(self):
        headers = ("A & B",)
        rows = (("<value>",),)
        result = _build_html_table(headers, rows)
        assert "<th>A &amp; B</th>" in result
        assert "<td>&lt;value&gt;</td>" in result

    def test_no_markdown_pipes(self):
        """Output should be HTML, not markdown pipe tables."""
        headers = ("Name", "Score")
        rows = (("Alice", "92.5"),)
        result = _build_html_table(headers, rows)
        assert "|" not in result
        assert "---" not in result

    def test_accepts_lists(self):
        """Should accept both tuples and lists."""
        headers = ["Name", "Score"]
        rows = [["Alice", "92.5"], ["Bob", "88.0"]]
        result = _build_html_table(headers, rows)
        assert "<th>Name</th>" in result
        assert "<td>Alice</td>" in result
