"""chatgpt-plugin-kit, Python port (canonical TypeScript: Drumworks/chatgpt-plugin-kit src/kit.ts).

Same rules as every other Drumworks ChatGPT plugin:

* TRUST is the secret path token. ``_meta["openai/subject"]`` arrives in the JSON-RPC body and is
  spoofable, so it is honoured only on a request that reached the token path.
* Client IP is the LAST X-Forwarded-For entry (the one our nginx appended).
* Limits FAIL CLOSED: any error counting is a refusal.
* Kill switch is a flag file, read per request: no restart needed.
* Model-visible copy is neutral: no pricing, upgrade, subscription or checkout language.
* Third-party text is wrapped in <<<untrusted>>> markers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KIT_VERSION = "1.0.0-py"

BANNED_COPY = re.compile(
    r"\$\s?\d|\bpricing\b|\bupgrade\b|\bsubscri(be|ption)\b|\bcheckout\b|\bapi key\b|\bfree tier\b|"
    r"\bpro key\b|/mo\b|per month",
    re.IGNORECASE,
)

NEUTRAL = {
    "limited": "This tool has reached its usage limit for now. Please try again later.",
    "disabled": "This tool is temporarily unavailable. Please try again later.",
    "error": "Something went wrong running this tool. Please try again.",
    "bad_input": "The request could not be understood. Check the arguments and try again.",
    "busy": "This tool is busy right now. Please try again in a minute.",
}


def find_banned_copy(text: str) -> str | None:
    m = BANNED_COPY.search(text)
    return m.group(0) if m else None


def token_matches(presented: str | None, expected: str | None) -> bool:
    if not presented or not expected or len(expected) < 32:
        return False
    a = hashlib.sha256(presented.encode()).digest()
    b = hashlib.sha256(expected.encode()).digest()
    return hmac.compare_digest(a, b)


def hash_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def client_ip(headers: dict[str, str] | Any) -> str:
    xff = headers.get("x-forwarded-for") if headers is not None else None
    if xff:
        parts = [p.strip() for p in str(xff).split(",") if p.strip()]
        if parts:
            return parts[-1][:64]
    return "unknown"


@dataclass(frozen=True)
class OpenAIMeta:
    subject: str | None
    session: str | None
    locale: str | None


def read_openai_meta(params: dict[str, Any] | None) -> OpenAIMeta:
    meta = params.get("_meta") if isinstance(params, dict) else None
    meta = meta if isinstance(meta, dict) else {}

    def s(k: str) -> str | None:
        v = meta.get(k)
        return v[:200] if isinstance(v, str) and v else None

    return OpenAIMeta(s("openai/subject"), s("openai/session"), s("openai/locale"))


@dataclass(frozen=True)
class Caller:
    key: str
    kind: str  # "subject" | "ip"
    ip: str


def identify_caller(meta: OpenAIMeta, ip: str) -> Caller:
    if meta.subject:
        return Caller(f"sub:{hash_id(meta.subject)}", "subject", ip)
    return Caller(f"ip:{ip}", "ip", ip)


class Limits:
    """In-process sliding windows. pdfmux's plugin runs as one process, and a call costs CPU, not
    money, so a restart resetting counters is an acceptable trade against a datastore."""

    def __init__(self, per_caller_daily: int, burst_per_min: int, global_pages_daily: int) -> None:
        self.per_caller_daily = per_caller_daily
        self.burst_per_min = burst_per_min
        self.global_pages_daily = global_pages_daily
        self._calls: dict[str, deque[float]] = {}
        self._pages: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, q: deque, now: float, window: float) -> None:
        while q and (q[0][0] if isinstance(q[0], tuple) else q[0]) <= now - window:
            q.popleft()

    def check(self, caller: Caller, now: float | None = None) -> str | None:
        """Return None if allowed, else a reason. Any internal error is a refusal."""
        try:
            now = now or time.time()
            with self._lock:
                q = self._calls.setdefault(caller.key, deque())
                self._prune(q, now, 86_400)
                if sum(1 for t in q if t > now - 60) >= self.burst_per_min:
                    return "burst"
                if len(q) >= self.per_caller_daily:
                    return "daily"
                self._prune(self._pages, now, 86_400)
                if sum(p for _, p in self._pages) >= self.global_pages_daily:
                    return "global"
                return None
        except Exception:  # noqa: BLE001
            return "store_error"

    def record(self, caller: Caller, pages: int, now: float | None = None) -> None:
        now = now or time.time()
        with self._lock:
            self._calls.setdefault(caller.key, deque()).append(now)
            self._pages.append((now, max(pages, 0)))
            if len(self._calls) > 50_000:
                for k in [k for k, q in self._calls.items() if not q or q[-1] <= now - 86_400]:
                    del self._calls[k]

    def pages_today(self, now: float | None = None) -> int:
        now = now or time.time()
        with self._lock:
            self._prune(self._pages, now, 86_400)
            return sum(p for _, p in self._pages)


def is_disabled(flag: Path) -> bool:
    try:
        return flag.exists()
    except OSError:
        return True


def untrusted(text: str) -> str:
    clean = re.sub(r"<<</?(end-)?untrusted>>>", "", text)
    return f"<<<untrusted>>>\n{clean}\n<<<end-untrusted>>>"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        pass  # logging must never fail a call
