"""CSV formatter — extract tables as CSV.

Best for PDFs that are primarily tabular data (invoices, spreadsheets,
data tables). Parses Markdown tables from extracted text into CSV format.
"""

from __future__ import annotations

import csv
import io
import re


def format_csv(text: str) -> str:
    """Extract tables from Markdown text and format as CSV.

    Finds Markdown tables (lines with | delimiters), parses them,
    and outputs CSV. Non-table text is ignored.

    Args:
        text: Post-processed extracted text containing Markdown tables.

    Returns:
        CSV string of all tables found. Tables are separated by blank lines.

    Raises:
        ValueError: If no tables are found in the text.
    """
    tables = _extract_markdown_tables(text)

    if not tables:
        raise ValueError(
            "No tables found in the extracted text. "
            "CSV format works best with table-heavy PDFs. "
            "Try 'markdown' format instead for general documents."
        )

    output = io.StringIO()
    writer = csv.writer(output)

    for i, table in enumerate(tables):
        if i > 0:
            writer.writerow([])  # Blank line between tables

        for row in table:
            writer.writerow(row)

    return output.getvalue()


def _extract_markdown_tables(text: str) -> list[list[list[str]]]:
    """Parse Markdown tables from text.

    Returns:
        List of tables, where each table is a list of rows,
        and each row is a list of cell strings.
    """
    lines = text.split("\n")
    tables: list[list[list[str]]] = []
    current_table: list[list[str]] = []

    for line in lines:
        line = line.strip()

        # Check if this line is a table row (contains | delimiters)
        if "|" in line and not re.match(r"^[\s|:-]+$", line):
            # Parse cells
            cells = [cell.strip() for cell in line.split("|")]
            # Remove empty first/last cells from leading/trailing |
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            if cells:
                current_table.append(cells)

        elif re.match(r"^[\s|:-]+$", line) and "|" in line:
            # This is a separator row (|---|---|), skip it
            continue

        else:
            # Not a table row — save current table if we have one
            if current_table:
                tables.append(current_table)
                current_table = []

    # Don't forget the last table
    if current_table:
        tables.append(current_table)

    return tables
