# LlamaIndex Readers Integration: pdfmux

[pdfmux](https://github.com/NameetP/pdfmux) extracts PDFs into section-aware, LLM-ready chunks. It scores each page's extraction confidence and routes low-confidence or scanned pages to an OCR fallback automatically, so mixed digital/scanned documents come back as clean text without manual pre-processing. Each chunk is returned as a LlamaIndex `Document` with title, page range, token estimate, and confidence metadata.

## Installation

```bash
pip install llama-index-readers-pdfmux
```

## Usage

```python
from llama_index.readers.pdfmux import PDFMuxReader

reader = PDFMuxReader(quality="standard")  # "fast" | "standard" | "high"
documents = reader.load_data("report.pdf")

for doc in documents:
    print(doc.metadata["title"], "—", doc.metadata["tokens"], "tokens")
```

Each returned `Document` carries metadata: `source`, `title`, `page_start`, `page_end`, `tokens`, `confidence`. Pass `extra_info={...}` to `load_data` to merge additional metadata into every document.

This loader is designed for RAG/LLM pipelines that ingest a mix of scanned and digital PDFs and need per-page confidence to decide what to trust.
