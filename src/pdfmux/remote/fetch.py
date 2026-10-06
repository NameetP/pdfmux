"""SSRF-safe PDF download for the remote extract_tables tool.

The front end is the only part of the plugin with network access, and it fetches URLs chosen
by whoever calls the tool. Everything here exists so that URL can't point inward:

* https only, default port only (443).
* Every address the hostname resolves to must be globally routable. ``ip.is_global`` rejects
  loopback, RFC 1918, link-local (incl. 169.254.169.254, the cloud metadata service), multicast,
  reserved, and 100.64.0.0/10, which is Tailscale's range. Tailnet-only services (gbrain, internal
  admin) are reachable from this box, and a generic "is_private" check misses that range.
* The address actually connected to is re-checked after connecting, so DNS rebinding between
  resolve and connect (a TOCTOU) can't land on an internal address.
* Redirects are followed manually (at most 3) and every hop is re-validated.
* The body is streamed with a hard byte cap and must start with ``%PDF-``.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

MAX_BYTES = 25 * 1024 * 1024
MAX_REDIRECTS = 3
TIMEOUT = httpx.Timeout(20.0, connect=8.0)


class FetchError(RuntimeError):
    """Safe-to-show message about why the file couldn't be fetched."""


def _public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def validate_url(url: str, resolver=None) -> str:
    """Return the hostname if the URL is fetchable; raise FetchError otherwise."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchError("The file link must start with https://.")
    if parts.username or parts.password:
        raise FetchError("The file link can't contain credentials.")
    if parts.port not in (None, 443):
        raise FetchError("The file link must use the standard https port.")
    host = parts.hostname or ""
    if not host or host.endswith((".local", ".internal", ".localhost")) or host == "localhost":
        raise FetchError("That file link isn't reachable.")
    try:
        # Resolved at call time (not bound as a default) so tests and callers can swap it.
        infos = (resolver or socket.getaddrinfo)(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise FetchError("That file link isn't reachable.") from None
    addrs = {i[4][0] for i in infos}
    if not addrs or not all(_public(a) for a in addrs):
        raise FetchError("That file link isn't reachable.")
    allowed = [
        h.strip().lower()
        for h in os.environ.get("PDFMUX_REMOTE_FETCH_HOSTS", "").split(",")
        if h.strip()
    ]
    if allowed and not any(host == h or host.endswith("." + h) for h in allowed):
        raise FetchError("That file link isn't from an allowed source.")
    return host


def _peer_ip(response: httpx.Response) -> str | None:
    stream = response.extensions.get("network_stream")
    if stream is None:
        return None
    addr = stream.get_extra_info("server_addr")
    return addr[0] if addr else None


def fetch_pdf(url: str, dest_dir: Path, client: httpx.Client | None = None) -> Path:
    """Download a PDF to ``dest_dir`` and return its path. Raises FetchError with a safe message."""
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=False, trust_env=False)
    try:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            validate_url(current)
            with client.stream("GET", current, headers={"User-Agent": "pdfmux-remote/1.0"}) as resp:
                peer = _peer_ip(resp)
                if peer is not None and not _public(peer):
                    raise FetchError("That file link isn't reachable.")
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError("That file link couldn't be downloaded.")
                    current = urljoin(current, location)
                    continue
                if resp.status_code != 200:
                    raise FetchError("That file link couldn't be downloaded.")
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                    raise FetchError("That PDF is larger than 25 MB.")
                dest_dir.mkdir(parents=True, exist_ok=True)
                path = dest_dir / f"{uuid.uuid4().hex}.pdf"
                size = 0
                first = b""
                with path.open("wb") as fh:
                    for chunk in resp.iter_bytes(64 * 1024):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            fh.close()
                            path.unlink(missing_ok=True)
                            raise FetchError("That PDF is larger than 25 MB.")
                        if len(first) < 5:
                            first += chunk[: 5 - len(first)]
                        fh.write(chunk)
                if not first.startswith(b"%PDF-"):
                    path.unlink(missing_ok=True)
                    raise FetchError("That file isn't a PDF.")
                return path
        raise FetchError("That file link redirected too many times.")
    except httpx.HTTPError:
        raise FetchError("That file link couldn't be downloaded.") from None
    finally:
        if own:
            client.close()
