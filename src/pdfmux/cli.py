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
import os
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
profiles_app = typer.Typer(
    name="profiles",
    help="Manage saved configuration profiles (invoices, receipts, papers, ...).",
    no_args_is_help=True,
)
app.add_typer(profiles_app, name="profiles")
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
    chunk: bool = typer.Option(
        False,
        "--chunk",
        help="Output RAG-ready chunks instead of a single document. Layout-aware splitting.",
    ),
    max_tokens: int = typer.Option(
        500,
        "--max-tokens",
        help="Maximum tokens per chunk (only with --chunk).",
    ),
    overlap: int = typer.Option(
        50,
        "--overlap",
        help="Token overlap between adjacent chunks (only with --chunk).",
    ),
    stdout: bool = typer.Option(
        False,
        "--stdout",
        help="Print output to stdout instead of writing to file.",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        "-m",
        help="Routing strategy: economy (cheapest), balanced (default), premium (best quality).",
    ),
    budget: float | None = typer.Option(
        None,
        "--budget",
        help="Maximum cost in USD for LLM extraction (e.g. 0.50).",
    ),
    llm_provider: str | None = typer.Option(
        None,
        "--llm-provider",
        help="LLM provider override: gemini, claude, openai, ollama.",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="LLM model override (e.g. gpt-4o-mini, claude-sonnet-4-6-20250514).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable the smart result cache for this run.",
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Wipe the smart result cache before extracting.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO-level logs."),
    debug: bool = typer.Option(False, "--debug", help="Show DEBUG-level logs."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress all logs except errors."),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Apply a saved profile (e.g. invoices, receipts, papers, contracts, bulk-rag). "
        "Explicit flags override profile values.",
    ),
) -> None:
    """Convert a PDF (or directory of PDFs) to Markdown."""
    _configure_logging(verbose=verbose, debug=debug, quiet=quiet)

    # Apply profile defaults — explicit flags win over profile values.
    if profile:
        from pdfmux.profiles import apply_profile_defaults

        explicit: dict[str, object | None] = {
            "quality": quality if quality != "standard" else None,
            "format": format if format != "markdown" else None,
            "mode": mode,
            "schema": schema,
            "chunk": True if chunk else None,
            "max_tokens": max_tokens if max_tokens != 500 else None,
            "overlap": overlap if overlap != 50 else None,
            "confidence": True if confidence else None,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "budget": budget,
        }
        try:
            merged = apply_profile_defaults(profile, explicit)
        except KeyError:
            console.print(
                f"[red]Profile '{profile}' not found.[/red] "
                "Run [bold]pdfmux profiles list[/bold] to see available profiles."
            )
            raise typer.Exit(2)

        quality = str(merged.get("quality") or "standard")
        format = str(merged.get("format") or "markdown")
        if merged.get("mode") is not None:
            mode = str(merged["mode"])
        if merged.get("schema") is not None:
            schema = str(merged["schema"])
        chunk = bool(merged.get("chunk")) or chunk
        max_tokens = int(merged.get("max_tokens") or max_tokens)
        overlap = int(merged.get("overlap") or overlap)
        confidence = bool(merged.get("confidence")) or confidence
        if not llm_provider and merged.get("llm_provider"):
            llm_provider = str(merged["llm_provider"])
        if not llm_model and merged.get("llm_model"):
            llm_model = str(merged["llm_model"])
        if budget is None and merged.get("budget") is not None:
            budget = float(merged["budget"])  # type: ignore[arg-type]

    # Set LLM provider/model via env vars so extractors pick them up
    if llm_provider:
        os.environ["PDFMUX_LLM_PROVIDER"] = llm_provider
    if llm_model:
        os.environ["PDFMUX_LLM_MODEL"] = llm_model
    if mode:
        os.environ["PDFMUX_MODE"] = mode
    if budget is not None:
        os.environ["PDFMUX_BUDGET"] = str(budget)

    # Smart cache controls
    if clear_cache:
        from pdfmux.result_cache import get_default_cache

        removed = get_default_cache().clear()
        console.print(f"[dim]Cleared {removed} cache entries[/dim]")

    # Auto-switch to JSON format when schema is provided
    effective_format = format
    if schema and format == "markdown":
        effective_format = "json"

    if input_path.is_dir():
        _convert_directory(
            input_path,
            output,
            effective_format,
            quality,
            confidence,
            use_cache=not no_cache,
        )
    else:
        _convert_file(
            input_path,
            output,
            effective_format,
            quality,
            confidence,
            stdout,
            schema=schema,
            chunk_mode=chunk,
            max_tokens=max_tokens,
            overlap_tokens=overlap,
            use_cache=not no_cache,
        )


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
def serve(
    http: bool = typer.Option(
        False,
        "--http",
        help="Use Streamable HTTP transport instead of stdio. Required for Smithery deployment.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port for HTTP transport (default: 8000). Only used with --http.",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind HTTP server to (default: 0.0.0.0). Only used with --http.",
    ),
) -> None:
    """Start the MCP server for AI agent integration.

    Default: stdio transport (for Claude Desktop, Cursor, etc.)
    With --http: Streamable HTTP transport (for Smithery, remote deployment)
    """
    import sys

    # Also support TRANSPORT env var for Docker/Smithery
    use_http = http or os.environ.get("TRANSPORT", "").lower() == "http"

    try:
        from pdfmux.mcp_server import run_http_server, run_server
    except ImportError:
        console.print(
            '[red]MCP server requires the "mcp" package.[/red]\n'
            'Install with: [bold]pip install "pdfmux[serve]"[/bold]'
        )
        raise typer.Exit(1)

    if use_http:
        sys.stderr.write(f"Starting pdfmux MCP server (HTTP) on {host}:{port}...\n")
        run_http_server(host=host, port=port)
    else:
        # Print to stderr — stdout is reserved for MCP JSON-RPC protocol
        sys.stderr.write("Starting pdfmux MCP server (stdio)...\n")
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
        (
            "opendataloader-pdf",
            "opendataloader_pdf",
            "OpenDataLoader",
            r"pip install pdfmux\[opendataloader]",
        ),
        ("docling", "docling.document_converter", "Docling", r"pip install pdfmux\[tables]"),
        ("rapidocr", "rapidocr", "RapidOCR", r"pip install pdfmux\[ocr]"),
        ("surya-ocr", "surya.recognition", "Surya OCR", r"pip install pdfmux\[ocr-heavy]"),
        ("marker-pdf", "marker.converters.pdf", "Marker", r"pip install pdfmux\[marker]"),
        ("mistralai", "mistralai", "Mistral OCR SDK", r"pip install pdfmux\[llm-mistral]"),
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

    # LLM providers
    console.print()
    console.print("[bold]LLM Providers[/bold]")

    llm_table = Table(show_header=True, header_style="bold")
    llm_table.add_column("Provider", min_width=10)
    llm_table.add_column("SDK", min_width=12)
    llm_table.add_column("API Key", min_width=12)
    llm_table.add_column("Default Model", min_width=20)
    llm_table.add_column("Install", min_width=28)

    install_hints = {
        "gemini": r"pip install pdfmux\[llm]",
        "claude": r"pip install pdfmux\[llm-claude]",
        "openai": r"pip install pdfmux\[llm-openai]",
        "ollama": r"pip install pdfmux\[llm-ollama]",
    }

    try:
        from pdfmux.providers import all_provider_status

        for p in all_provider_status():
            sdk_status = "[green]✓[/green]" if p["sdk_installed"] else "[dim]✗[/dim]"
            key_status = "[green]✓[/green]" if p["has_credentials"] else "[dim]✗[/dim]"
            hint = "" if p["available"] else f"[dim]{install_hints.get(p['name'], '')}[/dim]"
            display_name = f"{p['name']} [dim](custom)[/dim]" if p.get("custom") else p["name"]
            llm_table.add_row(display_name, sdk_status, key_status, str(p["default_model"]), hint)
    except Exception:
        llm_table.add_row("(error loading providers)", "", "", "", "")

    console.print(llm_table)

    # --- Coverage by document type ---
    console.print()
    console.print("[bold]Coverage by Document Type[/bold]")

    from pdfmux.router.engine import QUALITY_ESTIMATES, RouterEngine

    engine = RouterEngine()
    available = engine._get_available_extractors()

    page_types = [
        ("digital", "Digital text"),
        ("scanned", "Scanned docs"),
        ("tables", "Tables"),
        ("graphical", "Image-heavy"),
        ("mixed", "Mixed content"),
        ("handwritten", "Handwriting"),
        ("forms", "Forms"),
    ]

    # Best possible score per type (across ALL extractors including unavailable)
    best_possible: dict[str, tuple[float, str]] = {}
    best_available: dict[str, tuple[float, str]] = {}
    for pt, _ in page_types:
        best_p = (0.0, "none")
        best_a = (0.0, "none")
        for (ext, ptype), quality in QUALITY_ESTIMATES.items():
            if ptype == pt:
                if quality > best_p[0]:
                    best_p = (quality, ext)
                if ext in available and quality > best_a[0]:
                    best_a = (quality, ext)
        best_possible[pt] = best_p
        best_available[pt] = best_a

    for pt, display_name in page_types:
        avail_score, avail_ext = best_available.get(pt, (0.0, "none"))
        best_score, best_ext = best_possible.get(pt, (0.0, "none"))

        # Build bar
        bar_width = 20
        filled = int(avail_score * bar_width)
        bar = (
            "[green]" + "█" * filled + "[/green]" + "[dim]" + "░" * (bar_width - filled) + "[/dim]"
        )

        pct = f"{avail_score:.0%}"
        info = f"({avail_ext})"

        if best_score > avail_score + 0.05 and best_ext != avail_ext:
            gap = f" [yellow]→ add {best_ext} for {best_score:.0%}[/yellow]"
        else:
            gap = ""

        console.print(f"  {display_name:<15} {bar} {pct:>4}  [dim]{info}[/dim]{gap}")

    # --- Recommendations ---
    recommendations = []
    for pt, display_name in page_types:
        avail_score, avail_ext = best_available.get(pt, (0.0, "none"))
        best_score, best_ext = best_possible.get(pt, (0.0, "none"))
        improvement = best_score - avail_score
        if improvement > 0.05 and best_ext not in available:
            recommendations.append((improvement, pt, display_name, best_ext, best_score))

    if recommendations:
        recommendations.sort(reverse=True)
        console.print()
        console.print("[bold]Recommendations[/bold]")
        for improvement, pt, display_name, ext, score in recommendations[:3]:
            hint = install_hints.get(ext, f"pip install pdfmux[{ext}]")
            console.print(
                f"  [yellow]⚡[/yellow] Add [bold]{ext}[/bold] "
                f"for +{improvement:.0%} better {display_name.lower()} extraction"
            )
            console.print(f"     [dim]{hint}[/dim]")

    # --- Readiness score ---
    if best_available:
        avg_coverage = sum(s for s, _ in best_available.values()) / len(best_available)
        readiness = int(avg_coverage * 100)
        color = "green" if readiness >= 80 else "yellow" if readiness >= 60 else "red"
        console.print()
        console.print(f"[bold]Overall readiness: [{color}]{readiness}%[/{color}][/bold]")
        if readiness < 80:
            console.print("[dim]Add 1 more provider for better coverage[/dim]")

    console.print()


@app.command()
def benchmark(
    dataset: Path | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Directory containing benchmark PDFs + ground truth (.gt.md). Default: built-in dataset.",
    ),
    extractor: str | None = typer.Option(
        None,
        "--extractor",
        "-e",
        help="Benchmark a specific extractor only (e.g. pymupdf, docling, llm).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results JSON to this path. Default: ~/.config/pdfmux/eval_results.json",
    ),
) -> None:
    """Benchmark all available extractors against ground truth PDFs."""
    from pdfmux.eval.runner import BenchmarkRunner

    runner = BenchmarkRunner(dataset_dir=dataset)

    datasets = runner.discover_datasets()
    if not datasets:
        console.print("[red]No benchmark datasets found.[/red]")
        console.print(
            "Place PDFs with .gt.md ground truth files in eval/datasets/ or use --dataset."
        )
        raise typer.Exit(1)

    console.print(f"\n[bold]pdfmux benchmark[/bold] — {len(datasets)} documents\n")

    extractors_list = [extractor] if extractor else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=None)
        results = runner.run_all(extractors=extractors_list)
        progress.update(task, completed=True)

    # Display summary
    summary = results.summary_by_type()
    for page_type, extractors_data in summary.items():
        console.print(f"\n[bold]{page_type}[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Extractor", min_width=14)
        table.add_column("Overall", min_width=10, justify="right")
        table.add_column("Text Acc", min_width=10, justify="right")
        table.add_column("Structure", min_width=10, justify="right")
        table.add_column("Table F1", min_width=10, justify="right")
        table.add_column("Halluc", min_width=10, justify="right")
        table.add_column("Latency", min_width=10, justify="right")
        table.add_column("N", min_width=5, justify="right")

        # Sort by overall score
        sorted_ext = sorted(extractors_data.items(), key=lambda x: x[1]["overall"], reverse=True)
        for ext_name, metrics in sorted_ext:
            table.add_row(
                ext_name,
                f"{metrics['overall']:.2%}",
                f"{metrics['text_accuracy']:.2%}",
                f"{metrics['structure_preservation']:.2%}",
                f"{metrics['table_f1']:.2%}" if metrics["table_f1"] > 0 else "—",
                f"{metrics['hallucination_rate']:.2%}",
                f"{metrics['avg_latency_ms']:.0f}ms",
                str(int(metrics["n_documents"])),
            )
        console.print(table)

    # Save results
    saved_path = runner.save_results(results, output)
    console.print(f"\n[dim]Results saved to {saved_path}[/dim]")
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
        "LLM Vision",
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

            elif name == "LLM Vision":
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
def stream(
    input_path: Path = typer.Argument(
        ...,
        help="PDF file to stream-extract.",
        exists=True,
    ),
    quality: str = typer.Option(
        "standard",
        "--quality",
        "-q",
        help="Quality preset: fast (no OCR), standard (default), high.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write NDJSON to this file instead of stdout.",
    ),
) -> None:
    """Stream-extract a PDF, page-by-page, as NDJSON.

    Each line is a JSON object with ``type`` and ``data``. The first event
    is always ``classified``; the last is always ``complete``. Pipe into
    ``jq`` for live processing of large documents::

        pdfmux stream report.pdf | jq 'select(.type=="page") | .data.text'
    """
    import json as _json
    import sys

    from pdfmux.streaming import process_streaming

    out = output.open("w", encoding="utf-8") if output else None
    try:
        for ev in process_streaming(input_path, quality=quality):
            line = _json.dumps(ev.to_dict(), ensure_ascii=False)
            if out is not None:
                out.write(line)
                out.write("\n")
                out.flush()
            else:
                sys.stdout.write(line)
                sys.stdout.write("\n")
                sys.stdout.flush()
    finally:
        if out is not None:
            out.close()


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
    chunk_mode: bool = False,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    use_cache: bool = True,
) -> None:
    """Convert a single PDF file."""
    if output is None:
        if chunk_mode:
            ext = ".chunks.json"
        else:
            ext = {"markdown": ".md", "json": ".json", "csv": ".csv", "llm": ".json"}.get(
                fmt, ".md"
            )
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
            use_cache=use_cache,
        )

    # RAG chunking mode
    if chunk_mode:
        import json

        from pdfmux.chunking import chunk_for_rag

        chunks = chunk_for_rag(
            result.text,
            confidence=result.confidence,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            extractor=result.extractor_used,
        )

        chunk_output = {
            "source": str(input_path),
            "total_chunks": len(chunks),
            "total_tokens": sum(c.tokens for c in chunks),
            "confidence": result.confidence,
            "chunks": [
                {
                    "index": i,
                    "title": c.title,
                    "text": c.text,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "tokens": c.tokens,
                    "confidence": c.confidence,
                }
                for i, c in enumerate(chunks)
            ],
        }

        output_text = json.dumps(chunk_output, indent=2, ensure_ascii=False)

        if to_stdout:
            import sys

            sys.stdout.write(output_text)
            sys.stdout.write("\n")
        else:
            output.write_text(output_text, encoding="utf-8")
            console.print(
                f"[green]✓[/green] {input_path.name} → {output.name} "
                f"({len(chunks)} chunks, ~{sum(c.tokens for c in chunks)} tokens, "
                f"max {max_tokens} tok/chunk, {overlap_tokens} overlap)"
            )
        return

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
    *,
    use_cache: bool = True,
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

    for path, result_or_error in process_batch(
        pdfs,
        output_format=fmt,
        quality=quality,
        use_cache=use_cache,
    ):
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


# ---------------------------------------------------------------------------
# Profiles subcommands
# ---------------------------------------------------------------------------


@profiles_app.command("list")
def profiles_list_cmd() -> None:
    """List all available profiles (built-in + user)."""
    from pdfmux.profiles import list_profiles, load_profile

    rows = list_profiles()
    if not rows:
        console.print("[dim]No profiles found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Profile", min_width=12)
    table.add_column("Source", min_width=8)
    table.add_column("Highlights", min_width=40)

    for name, source in rows:
        try:
            settings = load_profile(name)
        except KeyError:
            settings = {}
        highlights = ", ".join(
            f"{k}={v}" for k, v in sorted(settings.items()) if v not in (None, False)
        )
        if not highlights:
            highlights = "[dim](empty)[/dim]"
        src_color = "green" if source == "user" else "cyan"
        table.add_row(name, f"[{src_color}]{source}[/{src_color}]", highlights)

    console.print(table)


@profiles_app.command("show")
def profiles_show_cmd(
    name: str = typer.Argument(..., help="Profile name to show."),
) -> None:
    """Show the settings for a single profile."""
    from pdfmux.profiles import load_profile

    try:
        settings = load_profile(name)
    except KeyError:
        console.print(f"[red]Profile '{name}' not found.[/red]")
        raise typer.Exit(2)

    console.print(f"\n[bold]{name}[/bold]")
    if not settings:
        console.print("[dim](no settings)[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting", min_width=14)
    table.add_column("Value", min_width=20)
    for k, v in sorted(settings.items()):
        table.add_row(k, str(v))
    console.print(table)


@profiles_app.command("save")
def profiles_save_cmd(
    name: str = typer.Argument(..., help="Profile name to save."),
    quality: str | None = typer.Option(None, "--quality", "-q"),
    format: str | None = typer.Option(None, "--format", "-f"),
    mode: str | None = typer.Option(None, "--mode", "-m"),
    schema: str | None = typer.Option(None, "--schema", "-s"),
    chunk: bool = typer.Option(False, "--chunk"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    overlap: int | None = typer.Option(None, "--overlap"),
    confidence: bool = typer.Option(False, "--confidence"),
    llm_provider: str | None = typer.Option(None, "--llm-provider"),
    llm_model: str | None = typer.Option(None, "--llm-model"),
    budget: float | None = typer.Option(None, "--budget"),
) -> None:
    """Save a new profile with the supplied flags."""
    from pdfmux.profiles import save_profile

    settings: dict[str, object] = {
        "quality": quality,
        "format": format,
        "mode": mode,
        "schema": schema,
        "chunk": True if chunk else None,
        "max_tokens": max_tokens,
        "overlap": overlap,
        "confidence": True if confidence else None,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "budget": budget,
    }
    cleaned = {k: v for k, v in settings.items() if v is not None}
    if not cleaned:
        console.print("[yellow]No settings supplied — refusing to save an empty profile.[/yellow]")
        raise typer.Exit(2)

    try:
        path = save_profile(name, cleaned)
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]Failed to save profile:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Saved profile '[bold]{name}[/bold]' → {path}")


@profiles_app.command("delete")
def profiles_delete_cmd(
    name: str = typer.Argument(..., help="Profile name to delete."),
) -> None:
    """Delete a user profile (built-ins cannot be deleted)."""
    from pdfmux.profiles import delete_profile

    try:
        deleted = delete_profile(name)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)

    if deleted:
        console.print(f"[green]✓[/green] Deleted profile '{name}'")
    else:
        console.print(f"[yellow]Profile '{name}' not found.[/yellow]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


@app.command()
def watch(
    directory: Path = typer.Argument(
        ...,
        help="Directory to watch for new PDFs.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profile to apply when processing each file.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to write outputs to. Defaults to the watched directory.",
    ),
    quality: str = typer.Option("standard", "--quality", "-q"),
    format: str = typer.Option("markdown", "--format", "-f"),
    mode: str | None = typer.Option(None, "--mode", "-m"),
    schema: str | None = typer.Option(None, "--schema", "-s"),
    chunk: bool = typer.Option(False, "--chunk"),
    max_tokens: int = typer.Option(500, "--max-tokens"),
    overlap: int = typer.Option(50, "--overlap"),
    process_existing: bool = typer.Option(
        False,
        "--process-existing",
        help="Also process PDFs that already exist in the directory at startup.",
    ),
) -> None:
    """Watch a directory and auto-process new PDFs as they appear.

    Press Ctrl+C to stop. A summary of processed/failed files is printed on exit.
    """
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        console.print(
            "[red]Watch mode requires the 'watchdog' package.[/red]\n"
            'Install with: [bold]pip install "pdfmux[watch]"[/bold]'
        )
        raise typer.Exit(1)

    if profile:
        from pdfmux.profiles import apply_profile_defaults

        explicit = {
            "quality": quality if quality != "standard" else None,
            "format": format if format != "markdown" else None,
            "mode": mode,
            "schema": schema,
            "chunk": True if chunk else None,
            "max_tokens": max_tokens if max_tokens != 500 else None,
            "overlap": overlap if overlap != 50 else None,
        }
        try:
            merged = apply_profile_defaults(profile, explicit)
        except KeyError:
            console.print(f"[red]Profile '{profile}' not found.[/red]")
            raise typer.Exit(2)
        quality = str(merged.get("quality") or quality)
        format = str(merged.get("format") or format)
        if merged.get("mode") is not None:
            mode = str(merged["mode"])
        if merged.get("schema") is not None:
            schema = str(merged["schema"])
        chunk = bool(merged.get("chunk")) or chunk
        max_tokens = int(merged.get("max_tokens") or max_tokens)
        overlap = int(merged.get("overlap") or overlap)

    if mode:
        os.environ["PDFMUX_MODE"] = mode

    out_dir = output_dir or directory
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"processed": 0, "failed": 0}

    def _process_pdf(pdf_path: Path) -> None:
        try:
            console.print(f"[dim]→ processing {pdf_path.name}...[/dim]")
            effective_format = format
            if schema and format == "markdown":
                effective_format = "json"

            from pdfmux.pipeline import process

            result = process(
                file_path=pdf_path,
                output_format=effective_format,
                quality=quality,
                schema=schema,
            )

            if chunk:
                import json

                from pdfmux.chunking import chunk_for_rag

                chunks = chunk_for_rag(
                    result.text,
                    confidence=result.confidence,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap,
                    extractor=result.extractor_used,
                )
                payload = {
                    "source": str(pdf_path),
                    "total_chunks": len(chunks),
                    "confidence": result.confidence,
                    "chunks": [
                        {
                            "index": i,
                            "title": c.title,
                            "text": c.text,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "tokens": c.tokens,
                            "confidence": c.confidence,
                        }
                        for i, c in enumerate(chunks)
                    ],
                }
                out_file = out_dir / f"{pdf_path.stem}.chunks.json"
                out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            else:
                ext = {"markdown": ".md", "json": ".json", "csv": ".csv", "llm": ".json"}.get(
                    effective_format, ".md"
                )
                out_file = out_dir / f"{pdf_path.stem}{ext}"
                out_file.write_text(result.text, encoding="utf-8")

            console.print(
                f"[green]✓[/green] {pdf_path.name} → {out_file.name} "
                f"({result.confidence:.0%})"
            )
            counts["processed"] += 1
        except Exception as e:
            console.print(f"[red]✗[/red] {pdf_path.name}: {e}")
            counts["failed"] += 1

    class PdfEventHandler(FileSystemEventHandler):
        def on_created(self, event: object) -> None:
            src_path = getattr(event, "src_path", None)
            is_dir = getattr(event, "is_directory", False)
            if is_dir or not src_path:
                return
            p = Path(str(src_path))
            if p.suffix.lower() != ".pdf":
                return
            time.sleep(0.5)
            if p.exists():
                _process_pdf(p)

    if process_existing:
        existing = sorted(directory.glob("*.pdf")) + sorted(directory.glob("*.PDF"))
        if existing:
            console.print(f"[dim]Processing {len(existing)} existing PDFs...[/dim]")
            for p in existing:
                _process_pdf(p)

    observer = Observer()
    handler = PdfEventHandler()
    observer.schedule(handler, str(directory), recursive=False)
    observer.start()

    console.print(f"[bold]Watching[/bold] {directory} — press [bold]Ctrl+C[/bold] to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping...[/dim]")
    finally:
        observer.stop()
        observer.join()

    console.print(
        f"\nDone: [green]{counts['processed']} processed[/green], "
        f"[red]{counts['failed']} failed[/red]"
    )


# ---------------------------------------------------------------------------
# Estimate command
# ---------------------------------------------------------------------------


@app.command()
def estimate(
    input_path: Path = typer.Argument(
        ...,
        help="PDF file to estimate cost for.",
        exists=True,
    ),
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Routing strategy: economy / balanced / premium.",
    ),
) -> None:
    """Estimate extraction cost across all available LLM providers.

    Classifies the PDF (cheap), then for each provider/model shows the
    expected cost given the chosen routing strategy.
    """
    from pdfmux.detect import classify
    from pdfmux.providers import discover_all_providers
    from pdfmux.router.engine import RouterEngine
    from pdfmux.router.strategies import Strategy

    classification = classify(input_path)
    n_pages = classification.page_count

    page_types: list[str] = []
    for i in range(n_pages):
        if i in classification.scanned_pages:
            page_types.append("scanned")
        elif i in classification.graphical_pages:
            page_types.append("graphical")
        elif classification.has_tables:
            page_types.append("tables")
        else:
            page_types.append("digital")

    try:
        strategy = Strategy(mode)
    except ValueError:
        console.print(f"[red]Invalid mode '{mode}'. Use economy/balanced/premium.[/red]")
        raise typer.Exit(2)

    engine = RouterEngine()
    base_estimate = engine.estimate_document_cost(page_types, strategy)

    console.print(
        f"\n[bold]{input_path.name}[/bold] — {n_pages} pages, mode=[bold]{mode}[/bold]\n"
    )

    llm_pages = 0
    for pt in page_types:
        decision = engine.select(pt, strategy)
        if decision.extractor == "llm":
            llm_pages += 1

    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider", min_width=10)
    table.add_column("Model", min_width=24)
    table.add_column("Per page", min_width=10, justify="right")
    table.add_column(f"Total ({llm_pages} LLM pages)", min_width=18, justify="right")
    table.add_column("Available", min_width=10)

    providers = discover_all_providers()
    if not providers:
        console.print("[yellow]No LLM providers discovered.[/yellow]")
    else:
        for name, provider in sorted(providers.items()):
            available = provider.available()
            avail_str = "[green]✓[/green]" if available else "[dim]—[/dim]"
            for model_info in provider.supported_models() or []:
                cost = provider.estimate_cost(image_bytes_count=200_000)
                per_page = cost.cost_usd
                total = per_page * llm_pages
                table.add_row(
                    name,
                    model_info.id,
                    f"${per_page:.4f}",
                    f"${total:.4f}",
                    avail_str,
                )

    console.print(table)
    console.print(
        f"\n[dim]Router-estimated total (current mode): ${base_estimate:.4f}[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Diff command
# ---------------------------------------------------------------------------


@app.command()
def diff(
    file1: Path = typer.Argument(..., help="First PDF.", exists=True),
    file2: Path = typer.Argument(..., help="Second PDF.", exists=True),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table (default) or unified.",
    ),
) -> None:
    """Extract two PDFs and report a structural diff.

    Compares page count, total chars, headings, table count, and confidence
    so you can spot regressions or differences between document versions.
    """
    from pdfmux.pipeline import process

    def _summary(path: Path) -> dict[str, object]:
        result = process(file_path=path, output_format="markdown", quality="standard")
        text = result.text
        lines = text.splitlines()
        headings = [
            line.strip()
            for line in lines
            if line.strip().startswith("#") and not line.strip().startswith("#!")
        ]
        tables = sum(
            1
            for line in lines
            if line.strip().startswith("|---") or line.strip().startswith("| ---")
        )
        return {
            "path": path,
            "page_count": result.page_count,
            "total_chars": len(text),
            "headings": headings,
            "tables": tables,
            "text": text,
            "confidence": result.confidence,
            "extractor": result.extractor_used,
        }

    s1 = _summary(file1)
    s2 = _summary(file2)

    if output_format == "unified":
        import difflib

        diff_lines = list(
            difflib.unified_diff(
                str(s1["text"]).splitlines(keepends=True),
                str(s2["text"]).splitlines(keepends=True),
                fromfile=str(file1),
                tofile=str(file2),
                n=3,
            )
        )
        if not diff_lines:
            console.print("[green]Files are identical at the text level.[/green]")
        else:
            console.print("".join(diff_lines))
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", min_width=18)
    table.add_column(file1.name, min_width=20)
    table.add_column(file2.name, min_width=20)
    table.add_column("Δ", min_width=10)

    def _delta(a: int, b: int) -> str:
        d = b - a
        if d == 0:
            return "[dim]0[/dim]"
        sign = "+" if d > 0 else ""
        return f"[yellow]{sign}{d}[/yellow]"

    h1_list = s1["headings"]
    h2_list = s2["headings"]
    assert isinstance(h1_list, list) and isinstance(h2_list, list)

    table.add_row(
        "Page count",
        str(s1["page_count"]),
        str(s2["page_count"]),
        _delta(int(s1["page_count"]), int(s2["page_count"])),  # type: ignore[arg-type]
    )
    table.add_row(
        "Total chars",
        f"{int(s1['total_chars']):,}",
        f"{int(s2['total_chars']):,}",
        _delta(int(s1["total_chars"]), int(s2["total_chars"])),  # type: ignore[arg-type]
    )
    table.add_row(
        "Headings",
        str(len(h1_list)),
        str(len(h2_list)),
        _delta(len(h1_list), len(h2_list)),
    )
    table.add_row(
        "Table rows",
        str(s1["tables"]),
        str(s2["tables"]),
        _delta(int(s1["tables"]), int(s2["tables"])),  # type: ignore[arg-type]
    )
    table.add_row(
        "Confidence",
        f"{float(s1['confidence']):.0%}",
        f"{float(s2['confidence']):.0%}",
        "",
    )
    table.add_row("Extractor", str(s1["extractor"]), str(s2["extractor"]), "")

    console.print(f"\n[bold]Diff:[/bold] {file1.name} vs {file2.name}\n")
    console.print(table)

    h1 = set(h1_list)
    h2 = set(h2_list)
    only1 = sorted(h1 - h2)
    only2 = sorted(h2 - h1)
    if only1 or only2:
        console.print()
        console.print("[bold]Heading differences[/bold]")
        for h in only1:
            console.print(f"  [red]- {h}[/red]")
        for h in only2:
            console.print(f"  [green]+ {h}[/green]")
    else:
        console.print("\n[dim]Same headings on both files.[/dim]")
    console.print()


if __name__ == "__main__":
    app()
