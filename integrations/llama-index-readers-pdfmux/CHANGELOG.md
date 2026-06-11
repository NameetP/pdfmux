# CHANGELOG

## [0.1.1]

- **Fix packaging namespace.** 0.1.0 shipped the reader as a flat top-level module (`llama_index_readers_pdfmux`), so the conventional `from llama_index.readers.pdfmux import PDFMuxReader` import failed. 0.1.1 ships the correct `llama_index/readers/pdfmux/` namespace package that the LlamaIndex ecosystem and LlamaHub expect.
- `load_data` now accepts `extra_info` and merges it into every Document's metadata (LlamaIndex `BaseReader` convention).
- Tightened dependency pins (`llama-index-core>=0.13.0,<0.15`).

## [0.1.0]

- Initial release. `PDFMuxReader` — a LlamaIndex document reader backed by pdfmux, with automatic OCR fallback for low-confidence/scanned pages and per-chunk confidence metadata.
