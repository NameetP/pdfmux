import sys
import types
from unittest.mock import patch

from llama_index.core.readers.base import BaseReader
from llama_index.readers.pdfmux import PDFMuxReader

# pdfmux ships native extraction deps that need not be installed to test the
# reader's mapping logic. Ensure an importable `pdfmux` module exists so the
# lazy `import pdfmux` in load_data resolves; the actual call is mocked.
if "pdfmux" not in sys.modules:
    try:
        import pdfmux  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("pdfmux")
        _stub.load_llm_context = lambda *args, **kwargs: []  # patched per-test
        sys.modules["pdfmux"] = _stub


def test_class():
    names = [b.__name__ for b in PDFMuxReader.__mro__]
    assert BaseReader.__name__ in names


def test_load_data_maps_chunks_to_documents():
    fake_chunks = [
        {
            "title": "Introduction",
            "text": "hello world",
            "page_start": 1,
            "page_end": 2,
            "tokens": 12,
            "confidence": 0.98,
        },
        {
            "title": "Methods",
            "text": "scanned page text",
            "page_start": 3,
            "page_end": 3,
            "tokens": 40,
            "confidence": 0.71,
        },
    ]
    with patch("pdfmux.load_llm_context", return_value=fake_chunks):
        reader = PDFMuxReader(quality="standard")
        docs = reader.load_data("report.pdf")

    assert len(docs) == 2
    assert docs[0].text == "hello world"
    assert docs[0].metadata["title"] == "Introduction"
    assert docs[0].metadata["page_start"] == 1
    assert docs[0].metadata["page_end"] == 2
    assert docs[0].metadata["tokens"] == 12
    assert docs[0].metadata["confidence"] == 0.98
    assert docs[0].metadata["source"] == "report.pdf"
    assert docs[1].metadata["confidence"] == 0.71


def test_extra_info_is_merged_into_metadata():
    fake_chunks = [
        {
            "title": "T",
            "text": "x",
            "page_start": 1,
            "page_end": 1,
            "tokens": 3,
            "confidence": 0.9,
        }
    ]
    with patch("pdfmux.load_llm_context", return_value=fake_chunks):
        docs = PDFMuxReader().load_data("a.pdf", extra_info={"team": "rag", "tag": "v1"})

    assert docs[0].metadata["team"] == "rag"
    assert docs[0].metadata["tag"] == "v1"
    # original chunk metadata still present
    assert docs[0].metadata["title"] == "T"


def test_quality_passed_through():
    fake_chunks = []
    with patch("pdfmux.load_llm_context", return_value=fake_chunks) as m:
        PDFMuxReader(quality="high").load_data("b.pdf")
    # quality kwarg forwarded to pdfmux.load_llm_context
    assert m.call_args.kwargs.get("quality") == "high"
