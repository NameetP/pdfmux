"""CLI entry point — the user-facing interface.

Usage:
    pdfmux invoice.pdf              → invoice.md
    pdfmux ./docs/ -o ./output/     → batch convert
    pdfmux report.pdf --confidence  → show confidence score
    pdfmux report.pdf -f llm        → LLM-ready chunked JSON
    pdfmux analyze report.pdf       → per-page extraction breakdown
    pdfmux serve                    → start MCP server
    pdfmux doctor                   → check your setup
    pdfmux bench report.pdf         → benchmark extractors

Exit codes:
    0 — success
    1 — extraction or runtime error
    2 — usage error (bad arguments, file not found)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pdfmux import __version__
from pdfmux.pipeline import process, process_batch

app = typer.Typer(
    name="pdfmux",
    help="PDF extraction that checks its own work.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _configure_logging(verbose: bool = False, debug: bool = False, quiet: bool = False) -> None:
    """Configure pdfmux logging based on CLI flags."""
    logger = logging.getLogger("pdfmux")
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING

    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)


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
        help="Output format: markdown (default), json, csv, llm.",
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
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="JSON schema file or preset for structured extraction. "
        "Extracts tables as JSON, key-value pairs, and maps to schema fields.",
    ),
    stdout: bool = typer.Option(
        False,
        "--stdout",
        help="Print output to stdout instead of writing to file.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO-level logs."),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress all logs except errors."),
) -> None:
    """Convert a PDF (or directory of PDFs) to Markdown."""
    _configure_logging(verbose=verbose, debug=debug, quiet=quiet)

    # Auto-switch to JSON format when schema is provided
    effective_format = format
    if schema and format == "markdown":
        effective_format = "json"

    if input_path.is_dir():
        _convert_directory(input_path, output, effective_format, quality, confidence)
    else:
        _convert_file(input_path, output, effective_format, quality, confidence, stdout, schema=schema)


def _version_callback(value: bool) -> None:
    if value:
        import sys

        sys.stderr.write(f"pdfmux {__version__}\n")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show pdfmux version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """PDF extraction that checks its own work."""


@app.command()
def serve() -> None:
    """Start the MCP server for AI agent integration."""
    import sys

    try:
        from pdfmux.mcp_server import run_server
    except ImportError:
        console.print(
            '[red]MCP server requires the "mcp" package.[/red]\n'
            'Install with: [bold]pip install "pdfmux[serve]"[/bold]'
        )
        raise typer.Exit(1)

    # Print to stderr — stdout is reserved for MCP JSON-RPC protocol
    sys.stderr.write("Starting pdfmux MCP server...\n")
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
        ("opendataloader-pdf", "opendataloader_pdf", "OpenDataLoader", r"pip install pdfmux\[opendataloader]"),
        ("docling", "docling.document_converter", "Docling", r"pip install pdfmux\[tables]"),
        ("rapidocr", "rapidocr", "RapidOCR", r"pip install pdfmux\[ocr]"),
        ("surya-ocr", "surya.recognition", "Surya OCR", r"pip install pdfmux\[ocr-heavy]"),
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
    """Benchmark all available extractors on a PDF."""
    from pdfmux.detect import classify
    from pdfmux.extractors.fast import FastExtractor
    from pdfmux.postprocess import clean_and_score

    classification = classify(input_path)
    console.print(f"\n[bold]{input_path.name}[/bold] — {classification.page_count} pages")
    detected_types = []
    if classification.is_digital:
        detected_types.append("digital")
    if classification.is_scanned:
        detected_types.append("scanned")
    if classification.is_mixed:
        detected_types.append("mixed")
    if classification.is_graphical:
        n = len(classification.graphical_pages)
        detected_types.append(f"[yellow]graphical ({n} image-heavy pages)[/yellow]")
    if classification.has_tables:
        detected_types.append("tables")
    console.print(f"Detected: {', '.join(detected_types)}\n")

    extractors_list = [
        "PyMuPDF",
        "Multi-pass",
        "OpenDataLoader",
        "Docling",
        "RapidOCR",
        "Surya OCR",
        "Gemini Flash",
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Extractor", min_width=16)
    table.add_column("Time", min_width=10, justify="right")
    table.add_column("Confidence", min_width=12, justify="right")
    table.add_column("Output", min_width=10, justify="right")
    table.add_column("Status")

    for name in extractors_list:
        try:
            start = time.perf_counter()

            if name == "PyMuPDF":
                ext = FastExtractor()
                raw = ext.extract_text(input_path)
                elapsed = time.perf_counter() - start
                processed = clean_and_score(
                    raw,
                    classification.page_count,
                    extraction_limited=classification.is_graphical,
                    graphical_page_count=(
                        len(classification.graphical_pages) if classification.is_graphical else 0
                    ),
                )
                chars = len(raw)
                conf = processed.confidence
                status = "[green]✓[/green]"

            elif name == "Multi-pass":
                result = process(
                    file_path=input_path,
                    output_format="markdown",
                    quality="standard",
                )
                elapsed = time.perf_counter() - start
                chars = len(result.text)
                conf = result.confidence
                n_ocr = len(result.ocr_pages)
                status = (
                    f"[green]✓[/green] {n_ocr} pages OCR'd"
                    if n_ocr > 0
                    else "[green]✓[/green] all pages good"
                )

            elif name == "OpenDataLoader":
                from pdfmux.extractors.opendataloader import OpenDataLoaderExtractor

                ext_o = OpenDataLoaderExtractor()
                if not ext_o.available():
                    raise ImportError("Not installed")
                raw = "\n\n---\n\n".join(p.text for p in ext_o.extract(input_path))
                elapsed = time.perf_counter() - start
                processed = clean_and_score(raw, classification.page_count)
                chars = len(raw)
                conf = processed.confidence
                status = "[green]✓[/green]"

            elif name == "Docling":
                from pdfmux.extractors.tables import TableExtractor

                ext_t = TableExtractor()
                if not ext_t.available():
                    raise ImportError("Not installed")
                raw = "\n\n---\n\n".join(p.text for p in ext_t.extract(input_path))
                elapsed = time.perf_counter() - start
                processed = clean_and_score(raw, classification.page_count)
                chars = len(raw)
                conf = processed.confidence
                status = "[green]✓[/green]"

            elif name == "RapidOCR":
                from pdfmux.extractors.rapid_ocr import RapidOCRExtractor

                ext_r = RapidOCRExtractor()
                if not ext_r.available():
                    raise ImportError("Not installed")
                raw = "\n\n---\n\n".join(p.text for p in ext_r.extract(input_path))
                elapsed = time.perf_counter() - start
                processed = clean_and_score(raw, classification.page_count)
                chars = len(raw)
                conf = processed.confidence
                status = "[green]✓[/green]"

            elif name == "Surya OCR":
                from pdfmux.extractors.ocr import OCRExtractor

                ext_s = OCRExtractor()
                if not ext_s.available():
                    raise ImportError("Not installed")
                raw = "\n\n---\n\n".join(p.text for p in ext_s.extract(input_path))
                elapsed = time.perf_counter() - start
                processed = clean_and_score(raw, classification.page_count)
                chars = len(raw)
                conf = processed.confidence
                status = "[green]✓[/green]"

            elif name == "Gemini Flash":
                from pdfmux.extractors.llm import LLMExtractor

                ext_l = LLMExtractor()
                if not ext_l.available():
                    raise ImportError("Not installed")
                raw = "\n\n---\n\n".join(p.text for p in ext_l.extract(input_path))
                elapsed = time.perf_counter() - start
                processed = clean_and_score(raw, classification.page_count)
                chars = len(raw)
                conf = processed.confidence
                status = "[green]✓[/green]"

            else:
                continue

            table.add_row(
                name,
                f"{elapsed:.2f}s",
                f"{conf:.0%}",
                f"{chars:,} chars",
                status,
            )
        except ImportError:
            table.add_row(name, "—", "—", "—", "[dim]not installed[/dim]")
        except Exception as e:
            msg = str(e)[:40]
            table.add_row(name, "—", "—", "—", f"[red]✗ {msg}[/red]")

    console.print(table)
    console.print()


@app.command()
def analyze(
    input_path: Path = typer.Argument(
        ...,
        help="PDF file to analyze.",
        exists=True,
    ),
) -> None:
    """Analyze a PDF — per-page extraction breakdown with confidence scores."""
    from pdfmux.audit import audit_document
    from pdfmux.detect import classify

    classification = classify(input_path)
    audit = audit_document(input_path)

    console.print(f"\n[bold]{input_path.name}[/bold] — {classification.page_count} pages\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Page", min_width=6, justify="right")
    table.add_column("Type", min_width=12)
    table.add_column("Quality", min_width=14)
    table.add_column("Chars", min_width=10, justify="right")

    for page_audit in audit.pages:
        page_num = page_audit.page_num + 1

        if page_audit.page_num in getattr(classification, "graphical_pages", []):
            page_type = "[yellow]graphical[/yellow]"
        elif page_audit.page_num in getattr(classification, "scanned_pages", []):
            page_type = "[cyan]scanned[/cyan]"
        else:
            page_type = "[green]digital[/green]"

        if page_audit.quality == "good":
            quality_str = "[green]good[/green] → fast extraction"
        elif page_audit.quality == "bad":
            quality_str = "[yellow]bad[/yellow] → needs OCR"
        else:
            quality_str = "[red]empty[/red] → needs OCR"

        table.add_row(
            str(page_num),
            page_type,
            quality_str,
            f"{page_audit.text_len:,}",
        )

    console.print(table)

    result = process(
        file_path=input_path,
        output_format="markdown",
        quality="standard",
    )

    console.print()

    conf = result.confidence
    if conf >= 0.8:
        conf_str = f"[green]{conf:.0%}[/green]"
    elif conf >= 0.5:
        conf_str = f"[yellow]{conf:.0%}[/yellow]"
    else:
        conf_str = f"[red]{conf:.0%}[/red]"

    console.print(f"  Confidence: {conf_str}")

    if result.ocr_pages:
        ocr_display = ", ".join(str(p + 1) for p in result.ocr_pages)
        console.print(f"  OCR pages:  {ocr_display}")

    console.print(f"  Extractor:  {result.extractor_used}")

    if result.warnings:
        console.print()
        for warning in result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {rich_escape(warning)}")

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
    *,
    schema: str | None = None,
) -> None:
    """Convert a single PDF file."""
    if output is None:
        ext = {"markdown": ".md", "json": ".json", "csv": ".csv", "llm": ".json"}.get(fmt, ".md")
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
            schema=schema,
        )

    if to_stdout:
        import sys

        sys.stdout.write(result.text)
        sys.stdout.write("\n")
    else:
        output.write_text(result.text, encoding="utf-8")

        conf = result.confidence
        if conf >= 0.8:
            conf_str = f"[green]{conf:.0%} confidence[/green]"
        elif conf >= 0.5:
            conf_str = f"[yellow]{conf:.0%} confidence[/yellow]"
        else:
            conf_str = f"[red]{conf:.0%} confidence[/red]"

        ocr_info = ""
        if result.ocr_pages:
            ocr_info = f", {len(result.ocr_pages)} pages OCR'd"

        console.print(
            f"[green]✓[/green] {input_path.name} → {output.name} "
            f"({result.page_count} pages, {conf_str}{ocr_info}, "
            f"via {result.extractor_used})"
        )

    if result.warnings:
        for warning in result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {rich_escape(warning)}")


def _convert_directory(
    input_dir: Path,
    output_dir: Path | None,
    fmt: str,
    quality: str,
    confidence: bool,
) -> None:
    """Convert all PDFs in a directory using process_batch()."""
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

    for path, result_or_error in process_batch(pdfs, output_format=fmt, quality=quality):
        ext = {"markdown": ".md", "json": ".json", "csv": ".csv", "llm": ".json"}.get(fmt, ".md")
        out_file = output_dir / path.with_suffix(ext).name

        if isinstance(result_or_error, Exception):
            console.print(f"  [red]✗[/red] {path.name}: {result_or_error}")
            failed += 1
        else:
            out_file.write_text(result_or_error.text, encoding="utf-8")
            console.print(
                f"  [green]✓[/green] {path.name} → {out_file.name} "
                f"({result_or_error.confidence:.0%})"
            )
            success += 1

    console.print(f"\nDone: {success} converted, {failed} failed")


if __name__ == "__main__":
    app()
