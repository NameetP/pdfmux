"""CLI entry point — the user-facing interface.

Usage:
    pdfmux invoice.pdf              → invoice.md
    pdfmux ./docs/ -o ./output/     → batch convert
    pdfmux report.pdf --confidence  → show confidence score
    pdfmux serve                    → start MCP server
    pdfmux doctor                   → check your setup
    pdfmux bench report.pdf         → benchmark extractors
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pdfmux import __version__
from pdfmux.pipeline import process

app = typer.Typer(
    name="pdfmux",
    help="The smart PDF-to-Markdown router. One command, zero config.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def convert(
    input_path: Path = typer.Argument(
        ...,
        help="PDF file or directory to convert.",
        exists=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file or directory. Defaults to same name with .md extension.",
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown (default), json, csv.",
    ),
    quality: str = typer.Option(
        "standard",
        "--quality",
        "-q",
        help="Quality preset: fast (rule-based), standard (auto), high (ML-based).",
    ),
    confidence: bool = typer.Option(
        False,
        "--confidence",
        help="Show confidence score in output.",
    ),
    stdout: bool = typer.Option(
        False,
        "--stdout",
        help="Print output to stdout instead of writing to file.",
    ),
) -> None:
    """Convert a PDF (or directory of PDFs) to Markdown."""
    if input_path.is_dir():
        _convert_directory(input_path, output, format, quality, confidence)
    else:
        _convert_file(input_path, output, format, quality, confidence, stdout)


@app.command()
def serve() -> None:
    """Start the MCP server for AI agent integration."""
    from pdfmux.mcp_server import run_server

    console.print("[bold]Starting Pdfmux MCP server...[/bold]")
    run_server()


@app.command()
def doctor() -> None:
    """Check your setup — installed extractors, versions, and readiness."""
    import importlib
    import sys

    console.print(f"\n[bold]pdfmux {__version__}[/bold]")
    console.print(f"Python {sys.version.split()[0]}\n")

    checks = [
        ("pymupdf", "fitz", "PyMuPDF", "Base (always available)"),
        ("pymupdf4llm", "pymupdf4llm", "pymupdf4llm", "Base (always available)"),
        ("docling", "docling.document_converter", "Docling", r"pip install pdfmux\[tables]"),
        ("surya-ocr", "surya.recognition", "Surya OCR", r"pip install pdfmux\[ocr]"),
        ("google-genai", "google.genai", "Gemini Flash", r"pip install pdfmux\[llm]"),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Extractor", min_width=14)
    table.add_column("Status", min_width=12)
    table.add_column("Version", min_width=10)
    table.add_column("Install", min_width=28)

    for pkg_name, import_name, display_name, install_hint in checks:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "—")
            table.add_row(display_name, "[green]✓ installed[/green]", ver, "")
        except ImportError:
            table.add_row(display_name, "[dim]✗ missing[/dim]", "—", f"[dim]{install_hint}[/dim]")

    console.print(table)

    # Check for API keys
    import os

    console.print()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        console.print("[green]✓[/green] GEMINI_API_KEY set")
    else:
        console.print(r"[dim]✗ GEMINI_API_KEY not set (only needed for pdfmux\[llm])[/dim]")

    console.print()


@app.command()
def bench(
    input_path: Path = typer.Argument(
        ...,
        help="PDF file to benchmark.",
        exists=True,
    ),
) -> None:
    """Benchmark all available extractors on a PDF. Shows speed and confidence side by side."""
    from pdfmux.detect import classify
    from pdfmux.extractors.fast import FastExtractor
    from pdfmux.postprocess import clean_and_score

    classification = classify(input_path)
    console.print(f"\n[bold]{input_path.name}[/bold] — {classification.page_count} pages")
    console.print(
        f"Detected: {'digital' if classification.is_digital else ''}"
        f"{'scanned' if classification.is_scanned else ''}"
        f"{'mixed' if classification.is_mixed else ''}"
        f"{', tables' if classification.has_tables else ''}\n"
    )

    extractors: list[tuple[str, str, type | None]] = [
        ("PyMuPDF", "pdfmux.extractors.fast", None),
        ("Docling", "pdfmux.extractors.tables", None),
        ("Surya OCR", "pdfmux.extractors.ocr", None),
        ("Gemini Flash", "pdfmux.extractors.llm", None),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Extractor", min_width=14)
    table.add_column("Time", min_width=10, justify="right")
    table.add_column("Confidence", min_width=12, justify="right")
    table.add_column("Output", min_width=10, justify="right")
    table.add_column("Status")

    for name, module_path, _ in extractors:
        try:
            if name == "PyMuPDF":
                ext = FastExtractor()
            elif name == "Docling":
                from pdfmux.extractors.tables import TableExtractor

                ext = TableExtractor()
            elif name == "Surya OCR":
                from pdfmux.extractors.ocr import OCRExtractor

                ext = OCRExtractor()
            elif name == "Gemini Flash":
                from pdfmux.extractors.llm import LLMExtractor

                ext = LLMExtractor()
            else:
                continue

            start = time.perf_counter()
            raw = ext.extract(input_path)
            elapsed = time.perf_counter() - start

            processed = clean_and_score(raw, classification.page_count)
            chars = len(raw)
            conf = processed.confidence

            table.add_row(
                name,
                f"{elapsed:.2f}s",
                f"{conf:.0%}",
                f"{chars:,} chars",
                "[green]✓[/green]",
            )
        except ImportError:
            table.add_row(name, "—", "—", "—", "[dim]not installed[/dim]")
        except Exception as e:
            msg = str(e)[:40]
            table.add_row(name, "—", "—", "—", f"[red]✗ {msg}[/red]")

    console.print(table)
    console.print()


@app.command()
def version() -> None:
    """Show the version."""
    console.print(f"pdfmux {__version__}")


def _convert_file(
    input_path: Path,
    output: Path | None,
    fmt: str,
    quality: str,
    confidence: bool,
    to_stdout: bool,
) -> None:
    """Convert a single PDF file."""
    if output is None:
        ext = {"markdown": ".md", "json": ".json", "csv": ".csv"}.get(fmt, ".md")
        output = input_path.with_suffix(ext)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Converting {input_path.name}...", total=None)
        result = process(
            file_path=input_path,
            output_format=fmt,
            quality=quality,
            show_confidence=confidence,
        )

    if to_stdout:
        console.print(result.text)
    else:
        output.write_text(result.text, encoding="utf-8")
        console.print(
            f"[green]✓[/green] {input_path.name} → {output.name} "
            f"({result.page_count} pages, {result.confidence:.0%} confidence, "
            f"via {result.extractor_used})"
        )

    if result.warnings:
        for warning in result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning}")


def _convert_directory(
    input_dir: Path,
    output_dir: Path | None,
    fmt: str,
    quality: str,
    confidence: bool,
) -> None:
    """Convert all PDFs in a directory."""
    if output_dir is None:
        output_dir = input_dir

    pdfs = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    if not pdfs:
        console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Converting {len(pdfs)} PDFs from {input_dir}...")

    success = 0
    failed = 0

    for pdf in pdfs:
        ext = {"markdown": ".md", "json": ".json", "csv": ".csv"}.get(fmt, ".md")
        out_file = output_dir / pdf.with_suffix(ext).name

        try:
            result = process(
                file_path=pdf,
                output_format=fmt,
                quality=quality,
                show_confidence=confidence,
            )
            out_file.write_text(result.text, encoding="utf-8")
            console.print(
                f"  [green]✓[/green] {pdf.name} → {out_file.name} ({result.confidence:.0%})"
            )
            success += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] {pdf.name}: {e}")
            failed += 1

    console.print(f"\nDone: {success} converted, {failed} failed")


if __name__ == "__main__":
    app()
