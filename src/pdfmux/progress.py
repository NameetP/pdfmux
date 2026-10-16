"""Page-progress reporting for long extractions (``NameetP/pdfmux#15``).

Progress goes through the standard ``pdfmux`` logger at INFO level, so
``pdfmux convert -v`` shows it and library callers get it through their own
logging config. Logging rather than a callback is deliberate: on Linux the
extraction runs in a forkserver child under a hard timeout (see
``_timeout.py``), and a closure handed across that boundary would never update
the parent. A log record written to the inherited stderr does.

The cadence is throttled so a 5000-page document produces on the order of 50
lines per stage, not 5000, while small documents report every page.
"""

from __future__ import annotations

# At or below this many pages, every page is reported.
SMALL_DOC_PAGES = 20
# On larger documents, report roughly every 2% but never more often than this.
MIN_INTERVAL = 5


def log_interval(total: int, *, every_page_below: int = SMALL_DOC_PAGES) -> int:
    """Pages between progress lines: 1 for small documents, ~2% (min 5) otherwise."""
    if total <= every_page_below:
        return 1
    return max(MIN_INTERVAL, total // 50)


def should_log(done: int, total: int, *, every_page_below: int = SMALL_DOC_PAGES) -> bool:
    """True when ``done`` of ``total`` pages is a reporting point.

    The final page is always a reporting point, so every stage ends on an
    unambiguous ``N/N`` line. Zero or negative counts never report.
    """
    if total <= 0 or done <= 0:
        return False
    return done >= total or done % log_interval(total, every_page_below=every_page_below) == 0
