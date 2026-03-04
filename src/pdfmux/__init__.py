"""pdfmux — The smart PDF-to-Markdown router."""

# Suppress pymupdf4llm "Consider using pymupdf_layout" noise on import
import io as _io
import sys as _sys

_orig = _sys.stdout
_sys.stdout = _io.StringIO()
try:
    import pymupdf4llm as _pmll  # noqa: F401
except ImportError:
    pass
finally:
    _sys.stdout = _orig
del _orig, _io

__version__ = "0.3.0"
