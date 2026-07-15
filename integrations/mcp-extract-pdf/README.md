# pdfmux — `extract_pdf` MCP server

One tool. Point an AI agent at a PDF, get clean Markdown back — with a
confidence score so the agent can tell a good extraction from a degraded one
instead of silently trusting garbage text.

This is the **single-tool** pdfmux MCP server, built for hosts and directories
that reward a small, obvious tool surface: Claude Desktop, Cursor,
[open-notebook](https://github.com/lfnovo/open-notebook), and the Claude MCP
directory. If you also want triage, batch conversion, table/key-value
extraction, and page-by-page streaming, run the full multi-tool server instead
(`pdfmux serve` — see the main pdfmux README).

## Why it exists

Most PDF-to-text tools fail silently. A scanned page comes back as three
characters of OCR noise, a rotated table collapses into a wall of numbers, and
the agent downstream has no way to know the text it just ingested is wrong.

`extract_pdf` returns a **per-document confidence score** alongside the text.
When extraction is uncertain — an image-only scan, a low-confidence page, a
partial result — it prepends a short header (confidence, extractor used, which
pages needed OCR, any warnings) so your agent can decide whether to trust the
output, re-run at higher quality, or ask a human. That's the whole point:
**prevent silent extraction failure.**

## The tool

### `extract_pdf(file_path, quality="standard")`

Convert a PDF to AI-readable Markdown.

| Arg | Default | Meaning |
|-----|---------|---------|
| `file_path` | — | Absolute path to the PDF. |
| `quality` | `"standard"` | `"fast"` (native text only, cheapest), `"standard"` (native text + OCR fallback on image pages), `"high"` (OCR/vision on every page that needs it, most thorough). |

**Returns:** the document as Markdown. If confidence is below 80% or any
warnings were raised, a metadata header precedes the content:

```
**Extraction confidence: 63%**
Extractor: ocr-fallback
Pages: 4
OCR pages: 2, 3

**Warnings:**
- Page 2 extracted below the confidence threshold — text may be unreliable.

---

<the extracted markdown...>
```

Clean digital PDFs come back as Markdown with no header — high confidence, no
warnings, nothing to flag.

## Install

```bash
pip install "pdfmux[serve]"        # core + MCP server
pip install "pdfmux[serve,ocr]"    # add OCR for scanned documents (recommended)
```

## Run

```bash
pdfmux-extract          # stdio transport (Claude Desktop, Cursor, open-notebook)
python -m pdfmux.mcp_extract   # equivalent
```

## Configure your MCP host

**Claude Desktop / Cursor** (`claude_desktop_config.json` or the Cursor MCP
config):

```json
{
  "mcpServers": {
    "pdfmux": {
      "command": "pdfmux-extract"
    }
  }
}
```

**open-notebook** — add the same server block to your MCP config; the single
`extract_pdf` tool shows up as a source-ingestion action.

## File access

`extract_pdf` resolves and validates every path before touching the
filesystem: paths are canonicalized and confined to an allowlist (working
directory by default; extend it with `PDFMUX_ALLOWED_DIRS`), and traversal
attempts are rejected. It reads the PDF you point it at and nothing else.

## Optional LLM extraction

`quality="high"` uses a vision model to read image-only pages. pdfmux
auto-detects a configured provider from your environment:

| Provider | Env var | Install |
|----------|---------|---------|
| Gemini / Gemma | `GEMINI_API_KEY` | `pip install "pdfmux[llm]"` |
| Claude | `ANTHROPIC_API_KEY` | `pip install "pdfmux[llm-claude]"` |
| OpenAI | `OPENAI_API_KEY` | `pip install "pdfmux[llm-openai]"` |
| Ollama (local) | — | `pip install "pdfmux[llm-ollama]"` |

`"fast"` and `"standard"` work with no LLM key at all.

---

> ## 🔒 Maintainer note — HOLD public distribution until the provisional is filed
>
> The code in this directory is **ready to ship but must not be published yet.**
> Two distribution moves are **on hold** until the pdfmux US provisional patent
> is filed (target **before 2026-07-20**):
>
> 1. **Claude MCP directory submission** — do not submit.
> 2. **open-notebook PR** (adding pdfmux as an MCP source connector) — do not
>    open.
>
> This README and the server itself are outcome-only ("what it does, never how
> it works") and are safe to have in the repo. The hold is on the *public
> distribution surfaces*, which draw eyes to the extraction architecture before
> the priority date is locked. **The instant the provisional is filed, both
> moves are green** — submit the directory listing and open the open-notebook PR
> the same day.
>
> See the pdfmux RUNBOOK patent-embargo banner and
> `products/pdfmux/decisions/2026-06-26-patent-disclosure-embargo.md`.
