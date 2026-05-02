"""Retry helper — exponential backoff with smart transient-error detection.

LLM API calls fail transiently all the time: rate limits, network blips,
upstream 502/503/504 from CDNs. This helper wraps a callable with a small,
honest retry loop.

Design goals:
    - Retry transient errors (rate limit, timeout, 5xx).
    - Never retry permanent errors (auth, bad request, validation, 404).
    - Honor `Retry-After` HTTP header when present.
    - Cap total wait time so a stuck endpoint can't hang the pipeline.
    - Log every retry with attempt number so debugging is easy.

Usage::

    @with_retry(max_attempts=3, backoff_base=2.0)
    def extract_page(self, image_bytes, prompt, model=None):
        ...

You can also use it imperatively::

    result = retry_call(fn, *args, max_attempts=3, backoff_base=2.0)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("pdfmux.retry")

F = TypeVar("F", bound=Callable[..., Any])


# Substrings that strongly indicate a transient failure. Conservative —
# we'd rather under-retry than retry a permanent error 5 times.
_TRANSIENT_HINTS: tuple[str, ...] = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection error",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "502",
    "503",
    "504",
    "529",  # Anthropic overloaded
    "overloaded",
)

# Substrings that indicate a permanent failure — never retry.
_PERMANENT_HINTS: tuple[str, ...] = (
    "invalid api key",
    "api key not valid",
    "unauthorized",
    "forbidden",
    "permission denied",
    "authentication",
    "not found",
    "404",
    "400 bad request",
    "invalid request",
    "validation error",
)


def _err_text(exc: BaseException) -> str:
    """Best-effort string representation of an exception for matching."""
    parts: list[str] = [type(exc).__name__, str(exc)]
    # Some SDKs put status code on .status_code or .code
    for attr in ("status_code", "code", "http_status"):
        v = getattr(exc, attr, None)
        if v is not None:
            parts.append(str(v))
    return " ".join(parts).lower()


def is_transient(exc: BaseException) -> bool:
    """Return True if `exc` looks like a transient/retryable failure."""
    text = _err_text(exc)

    # Permanent first — wins ties (e.g. a 401 that mentions "rate limit").
    for hint in _PERMANENT_HINTS:
        if hint in text:
            return False

    # Numeric status code on the exception object (some SDKs attach it).
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int):
        if status in (408, 425, 429) or 500 <= status < 600:
            return True
        if 400 <= status < 500:
            return False

    for hint in _TRANSIENT_HINTS:
        if hint in text:
            return True

    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a Retry-After delay from the exception, in seconds, or None."""
    # SDK exceptions often expose response/headers
    for attr_chain in (
        ("response", "headers"),
        ("response", "headers"),
        ("headers",),
    ):
        obj: Any = exc
        try:
            for a in attr_chain:
                obj = getattr(obj, a, None)
                if obj is None:
                    break
            if obj is None:
                continue
            # obj should be a Mapping[str, str]
            value = None
            if hasattr(obj, "get"):
                value = obj.get("Retry-After") or obj.get("retry-after")
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        except Exception:
            continue
    return None


def _sleep_for(attempt: int, backoff_base: float, max_sleep: float, jitter: bool) -> float:
    """Compute and execute the backoff sleep. Returns seconds slept."""
    delay = min(max_sleep, backoff_base**attempt)
    if jitter:
        delay = delay * (0.5 + random.random() / 2.0)  # 50–100% of computed delay
    time.sleep(delay)
    return delay


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    *,
    max_sleep: float = 30.0,
    jitter: bool = True,
    transient: Callable[[BaseException], bool] | None = None,
) -> Callable[[F], F]:
    """Decorator: wrap a function with exponential-backoff retry.

    Args:
        max_attempts: Total tries including the first call. 1 disables retry.
        backoff_base: Base for exponential delay. delay = base ** attempt.
        max_sleep:    Cap on individual sleep between attempts (seconds).
        jitter:       If True, randomize each delay to 50–100% of computed.
        transient:    Override the default transient-detection predicate.

    Returns:
        A decorator that retries the wrapped function on transient errors.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    is_transient_fn = transient or is_transient

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:
                    last_exc = exc
                    if not is_transient_fn(exc):
                        raise
                    if attempt >= max_attempts - 1:
                        raise

                    # Honor Retry-After if present, else exponential backoff.
                    retry_after = _retry_after_seconds(exc)
                    if retry_after is not None:
                        delay = min(max_sleep, retry_after)
                        time.sleep(delay)
                    else:
                        delay = _sleep_for(attempt + 1, backoff_base, max_sleep, jitter)

                    logger.warning(
                        "retry %d/%d for %s after %.2fs: %s",
                        attempt + 1,
                        max_attempts - 1,
                        getattr(fn, "__name__", "fn"),
                        delay,
                        exc,
                    )
            # Should never reach here — final attempt either returns or raises.
            assert last_exc is not None
            raise last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_call(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    max_sleep: float = 30.0,
    jitter: bool = True,
    transient: Callable[[BaseException], bool] | None = None,
    **kwargs: Any,
) -> Any:
    """Imperative form of `with_retry` — run a callable with retries."""
    wrapped = with_retry(
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        max_sleep=max_sleep,
        jitter=jitter,
        transient=transient,
    )(fn)
    return wrapped(*args, **kwargs)
