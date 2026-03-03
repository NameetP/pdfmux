# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-03

### Added
- Docling table extractor (97.9% table accuracy) — `pip install readable[tables]`
- Surya OCR extractor for scanned PDFs — `pip install readable[ocr]`
- Gemini 2.5 Flash LLM fallback for hardest cases — `pip install readable[llm]`
- JSON output format (`-f json`) with structured metadata and per-page chunks
- CSV output format (`-f csv`) for table-heavy documents
- Smart routing: pipeline now auto-routes to Docling/OCR/LLM when available
- `--quality high` mode uses LLM vision for maximum accuracy
- Mixed PDF handling: digital pages via PyMuPDF + scanned pages via OCR
- Graceful fallback: missing optional deps fall back to next-best extractor

### Fixed
- Security: added *.pem, *.key, credentials, secrets to .gitignore

## [0.1.0] - 2026-03-03

### Added
- PDF type detection (digital vs scanned, table detection)
- PyMuPDF fast extractor for digital PDFs (~0.01s/page)
- Markdown output formatter optimized for LLM consumption
- CLI with `readable convert` command (single file and batch)
- MCP server via `readable serve` for AI agent integration
- Confidence scoring with `--confidence` flag
- Post-processing: whitespace normalization, broken-word fixing, encoding cleanup
- Quality presets: fast, standard, high
- GitHub Actions CI (lint → test → build)
- Docker multi-stage build
