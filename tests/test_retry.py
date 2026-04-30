"""Tests for the retry/backoff helper."""

from __future__ import annotations

import pytest

from pdfmux.retry import is_transient, retry_call, with_retry


class _FakeResp:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _RateLimitErr(Exception):
    def __init__(self, msg: str = "Rate limit hit", status_code: int = 429) -> None:
        super().__init__(msg)
        self.status_code = status_code


class _AuthErr(Exception):
    def __init__(self) -> None:
        super().__init__("Invalid API key")
        self.status_code = 401


class _RetryAfterErr(Exception):
    def __init__(self) -> None:
        super().__init__("Too many requests")
        self.response = _FakeResp({"Retry-After": "0.01"})
        self.status_code = 429


class TestIsTransient:
    def test_rate_limit_text_is_transient(self) -> None:
        assert is_transient(Exception("Rate limit exceeded"))

    def test_timeout_is_transient(self) -> None:
        assert is_transient(Exception("Request timed out"))

    def test_503_is_transient(self) -> None:
        assert is_transient(Exception("503 Service Unavailable"))

    def test_429_status_is_transient(self) -> None:
        assert is_transient(_RateLimitErr())

    def test_5xx_status_is_transient(self) -> None:
        e = Exception("server error")
        e.status_code = 503  # type: ignore[attr-defined]
        assert is_transient(e)

    def test_auth_is_not_transient(self) -> None:
        assert not is_transient(_AuthErr())

    def test_400_is_not_transient(self) -> None:
        e = Exception("400 Bad Request validation error")
        e.status_code = 400  # type: ignore[attr-defined]
        assert not is_transient(e)

    def test_404_is_not_transient(self) -> None:
        assert not is_transient(Exception("404 Not Found"))


class TestRetryDecorator:
    def test_succeeds_first_try(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=3, backoff_base=1.01, jitter=False, max_sleep=0)
        def fn() -> int:
            calls["n"] += 1
            return 42

        assert fn() == 42
        assert calls["n"] == 1

    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=3, backoff_base=1.01, jitter=False, max_sleep=0)
        def fn() -> int:
            calls["n"] += 1
            if calls["n"] < 3:
                raise _RateLimitErr()
            return 7

        assert fn() == 7
        assert calls["n"] == 3

    def test_gives_up_after_max_attempts(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=2, backoff_base=1.01, jitter=False, max_sleep=0)
        def fn() -> None:
            calls["n"] += 1
            raise _RateLimitErr()

        with pytest.raises(_RateLimitErr):
            fn()
        assert calls["n"] == 2

    def test_does_not_retry_permanent(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=5, backoff_base=1.01, jitter=False, max_sleep=0)
        def fn() -> None:
            calls["n"] += 1
            raise _AuthErr()

        with pytest.raises(_AuthErr):
            fn()
        assert calls["n"] == 1

    def test_retry_after_header_honored(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=2, backoff_base=1.01, jitter=False, max_sleep=0.5)
        def fn() -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RetryAfterErr()
            return 1

        assert fn() == 1
        assert calls["n"] == 2

    def test_retry_call_imperative(self) -> None:
        calls = {"n": 0}

        def fn() -> int:
            calls["n"] += 1
            if calls["n"] < 2:
                raise _RateLimitErr()
            return 99

        out = retry_call(fn, max_attempts=3, backoff_base=1.01, jitter=False, max_sleep=0)
        assert out == 99

    def test_max_attempts_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            with_retry(max_attempts=0)

    def test_preserves_function_metadata(self) -> None:
        @with_retry(max_attempts=1)
        def my_fn() -> None:
            """Docstring."""

        assert my_fn.__name__ == "my_fn"
        assert my_fn.__doc__ == "Docstring."
