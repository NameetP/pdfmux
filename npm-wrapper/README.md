# pdfmux-mcp

MCP server for [pdfmux](https://pdfmux.com) — give AI agents reliable PDF reading.

**#2 on [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)** (200 real-world PDFs). The only PDF extractor that audits every page.

## Quick setup

### Claude Desktop / Cursor

```json
{
  "mcpServers": {
    "pdfmux": {
      "command": "npx",
      "args": ["-y", "pdfmux-mcp"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add pdfmux -- npx -y pdfmux-mcp
```

That's it. Python 3.11+ and pdfmux are installed automatically on first run.

## Tools

| Tool | What it does |
|------|-------------|
| `get_pdf_metadata` | Page count, file size, type detection (instant, no extraction) |
| `convert_pdf` | Full extraction to Markdown, JSON, or LLM chunks |
| `analyze_pdf` | Per-page quality audit without full extraction |
| `batch_convert` | Convert all PDFs in a directory |
| `extract_structured` | Tables, key-values, and schema mapping |
| `extract_streaming` | NDJSON page-by-page events (best for long documents) |

## Requirements

- Node.js 18+ (for npx)
- Python 3.11+ (auto-detected)
- pdfmux 1.6.0+ (auto-installed)

## Remote / Docker

For remote deployment (Smithery, Railway, etc.):

```bash
docker pull pdfmux/mcp-server
docker run -p 8000:8000 pdfmux/mcp-server
```

Or self-host:

```bash
pip install "pdfmux[serve]"
pdfmux serve --http --port 8000
```

## License

MIT
