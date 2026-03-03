"""CLI entry point — the user-facing interface.

Usage:
    pdfmux invoice.pdf              → invoice.md
    pdfmux ./docs/ -o ./output/     → batch convert
    pdfmux report.pdf --confidence  → show confidence score
    pdfmux serve                    → start MCP server
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
