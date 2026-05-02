"""Mistral OCR extractor — cheap, high-volume cloud OCR.

Uses Mistral's hosted OCR API ($0.002/page, 96.6% table accuracy).
Best for cost-sensitive batch extraction of scanned PDFs and tables.

Install: pip install pdfmux[llm-mistral]

Env vars:
  MISTRAL_API_KEY  — required
  PDFMUX_MISTRAL_MODEL  — override the default `mistral-ocr-latest` model
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

from pdfmux.extractors import register
from pdfmux.types import PageQuality, PageResult

logger = logging.getLogger(__name__)


# $0.002 per page — flat rate at time of writing.
MISTRAL_COST_PER_PAGE = 0.002
DEFAULT_MISTRAL_MODEL = "mistral-ocr-latest"


def _check_mistral_sdk() -> bool:
    """Check if the mistralai SDK is installed."""
    try:
        import mistralai  # noqa: F401

        return True
    except ImportError:
        return False


def _has_api_key() -> bool:
    """Check if MISTRAL_API_KEY is set."""
    return bool(os.environ.get("MISTRAL_API_KEY"))


@register(name="mistral_ocr", priority=25)
class MistralOCRExtractor:
    """Extract text from PDFs using Mistral's OCR API.

    Uploads the PDF to Mistral's file API, gets a signed URL, then
    invokes the OCR endpoint. Returns one PageResult per page with
    confidence derived from response metadata.
    """

    def __init__(self) -> None:
        self._client = None

    def _resolve_client(self):
        """Lazily build the Mistral client on first use."""
        if self._client is not None:
            return self._client

        if not _check_mistral_sdk():
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "Mistral SDK is not installed. Install with: pip install pdfmux[llm-mistral]"
            )

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            from pdfmux.errors import ExtractorNotAvailable

            raise ExtractorNotAvailable(
                "MISTRAL_API_KEY is not set. "
                "Get a key at https://console.mistral.ai/ and export it."
            )

        from mistralai import Mistral

        self._client = Mistral(api_key=api_key)
        return self._client

    @property
    def name(self) -> str:
        return "mistral_ocr"

    def available(self) -> bool:
        """True if SDK is installed AND API key is configured."""
        return _check_mistral_sdk() and _has_api_key()

    def extract(
        self,
        file_path: str | Path,
        pages: list[int] | None = None,
    ) -> Iterator[PageResult]:
        """Yield one PageResult per page via Mistral OCR.

        Local PDFs are uploaded to Mistral's file API first, then a
        signed URL is generated and passed to the OCR endpoint.
        """
        if not self.available():
            from pdfmux.errors import ExtractorNotAvailable

            if not _check_mistral_sdk():
                raise ExtractorNotAvailable(
                    "Mistral SDK is not installed. Install with: pip install pdfmux[llm-mistral]"
                )
            raise ExtractorNotAvailable(
                "MISTRAL_API_KEY is not set. "
                "Get a key at https://console.mistral.ai/ and export it."
            )

        client = self._resolve_client()
        file_path = Path(file_path)
        model = os.environ.get("PDFMUX_MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL)

        # Step 1: upload the PDF
        try:
            with open(file_path, "rb") as f:
                uploaded = client.files.upload(
                    file={
                        "file_name": file_path.name,
                        "content": f.read(),
                    },
                    purpose="ocr",
                )
        except Exception as e:
            from pdfmux.errors import ExtractionError

            logger.warning("Mistral upload failed for %s: %s", file_path.name, e)
            raise ExtractionError(f"Mistral upload failed: {e}") from e

        # Step 2: get a signed URL
        try:
            signed = client.files.get_signed_url(file_id=uploaded.id)
            document_url = getattr(signed, "url", None) or signed["url"]  # type: ignore[index]
        except Exception as e:
            from pdfmux.errors import ExtractionError

            logger.warning("Mistral signed URL failed for %s: %s", file_path.name, e)
            raise ExtractionError(f"Mistral signed URL failed: {e}") from e

        # Step 3: run OCR
        try:
            response = client.ocr.process(
                model=model,
                document={"type": "document_url", "document_url": document_url},
            )
        except Exception as e:
            from pdfmux.errors import ExtractionError

            logger.warning("Mistral OCR failed on %s: %s", file_path.name, e)
            raise ExtractionError(f"Mistral OCR failed: {e}") from e

        # Step 4: parse response — `response.pages` is a list of OCRPageObject
        ocr_pages = getattr(response, "pages", None) or []

        for i, page_obj in enumerate(ocr_pages):
            if pages is not None and i not in pages:
                continue

            # OCRPageObject.markdown carries page text; index reflects 0-based page
            page_index = getattr(page_obj, "index", i)
            text = getattr(page_obj, "markdown", "") or ""

            has_text = len(text.strip()) > 10

            # Confidence: derive from response metadata if available.
            # Mistral OCR exposes per-page `confidence` on some response shapes;
            # fall back to a conservative default when missing.
            page_confidence = getattr(page_obj, "confidence", None)
            if page_confidence is None:
                # `usage_info` sometimes carries an aggregate score
                usage = getattr(response, "usage_info", None)
                page_confidence = getattr(usage, "confidence", None) if usage else None

            if has_text:
                if isinstance(page_confidence, (int, float)) and 0.0 <= page_confidence <= 1.0:
                    confidence = float(page_confidence)
                else:
                    # 96.6% table accuracy → high default
                    confidence = 0.93
            else:
                confidence = 0.0

            yield PageResult(
                page_num=page_index,
                text=text.strip(),
                confidence=confidence,
                quality=PageQuality.GOOD if has_text else PageQuality.EMPTY,
                extractor=self.name,
                ocr_applied=True,
                cost_usd=MISTRAL_COST_PER_PAGE,
            )
