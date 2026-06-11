"""pdfmux reader."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PDFMuxReader(BaseReader):
    """
    pdfmux PDF reader.

    Extracts a PDF into section-aware, LLM-ready chunks using pdfmux. pdfmux
    scores each page's extraction confidence and routes low-confidence or
    scanned pages to an OCR fallback automatically, so mixed digital/scanned
    documents come back as clean text without manual pre-processing. Each
    chunk becomes a LlamaIndex Document carrying title, page range, token
    estimate, and confidence metadata.

    Args:
        quality (str): Extraction quality preset — "fast", "standard"
            (default), or "high".

    """

    def __init__(self, *args: Any, quality: str = "standard", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quality = quality

    def load_data(
        self, file: str, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load a PDF and return a list of LlamaIndex Documents.

        Args:
            file (str): Path to the PDF file.
            extra_info (Optional[Dict]): Extra metadata merged into the
                metadata of every returned Document.

        Returns:
            List[Document]: One Document per pdfmux chunk, each with
                ``source``, ``title``, ``page_start``, ``page_end``,
                ``tokens``, and ``confidence`` metadata.

        """
        import pdfmux

        path = Path(file)
        chunks = pdfmux.load_llm_context(path, quality=self.quality)

        documents: List[Document] = []
        for chunk in chunks:
            metadata = {
                "source": str(path),
                "title": chunk["title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "tokens": chunk["tokens"],
                "confidence": chunk["confidence"],
            }
            if extra_info:
                metadata.update(extra_info)
            documents.append(Document(text=chunk["text"], metadata=metadata))

        return documents
