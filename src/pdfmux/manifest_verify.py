"""Offline verification of pdfmux Cloud signed manifests — the free half of the wedge.

Every **paid** (Verified / Verified Pro) extraction from pdfmux Cloud ships an
Ed25519 signature over its extraction manifest. Ed25519 is *asymmetric*: pdfmux
signs with a private key it never discloses, and anyone verifies with the
**published public key**. So a customer — or their auditor — can prove a manifest
is authentic and unaltered **offline**: no API call, no account, no trust in
pdfmux required.

That asymmetry *is* the product. Generating a signed manifest is the one thing a
local ``pip install pdfmux`` cannot do (only the key-holder can). *Verifying* one
is free and open, and lives here in the MIT core on purpose — the more people who
can independently check a pdfmux certification, the more the certification is
worth.

The signature is computed over the canonical bytes
``json.dumps(manifest, sort_keys=True).encode("utf-8")`` — byte-for-byte the same
canonicalization the Cloud signer uses. Reproduce those exact bytes and check the
signature against the published key. This module keeps that canonicalization in
lock-step with the Cloud worker (``worker/manifest_signing.py``) and the recipe
in ``docs/verifying-a-manifest.md``; the three must never drift.

Only dependency beyond the stdlib: ``cryptography``.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

SIGNING_ALGORITHM = "ed25519"

# Public key published by pdfmux Cloud. Fetch once, then verify forever offline.
DEFAULT_PUBKEY_URL = "https://api.pdfmux.com/.well-known/pdfmux-manifest-pubkey"


class ManifestVerificationError(Exception):
    """Raised for malformed input (not for a merely-invalid signature — that is a
    clean ``False`` / ``VerifyResult(valid=False)``)."""


def canonical_bytes(manifest: dict[str, Any]) -> bytes:
    """The exact byte string that was signed.

    MUST stay byte-for-byte identical to the Cloud worker's
    ``manifest_signing.canonical_bytes``: ``json.dumps(manifest, sort_keys=True)``
    with Python's default separators (``", "`` / ``": "``), UTF-8 encoded.
    """
    return json.dumps(manifest, sort_keys=True).encode("utf-8")


def load_public_key(material: str) -> Any:
    """Load an Ed25519 public key from a PEM (SubjectPublicKeyInfo) block or a
    base64-encoded 32-byte raw public key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    material = material.strip()
    if "BEGIN" in material:
        key = serialization.load_pem_public_key(material.encode("utf-8"))
        if not isinstance(key, Ed25519PublicKey):
            raise ManifestVerificationError("public key is not an Ed25519 key")
        return key
    try:
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(material))
    except Exception as exc:  # noqa: BLE001
        raise ManifestVerificationError(f"could not parse public key: {exc}") from exc


def verify_manifest(
    manifest: dict[str, Any],
    signature: str,
    public_key: str,
) -> bool:
    """True iff ``signature`` (base64) is a valid pdfmux Ed25519 signature over
    ``manifest`` for ``public_key`` (PEM or base64).

    Returns ``False`` for any tampering, wrong key, or malformed signature. Raises
    :class:`ManifestVerificationError` only if the *public key* can't be parsed —
    a key problem is the caller's to fix, a bad signature is a real verdict.
    """
    from cryptography.exceptions import InvalidSignature

    key = load_public_key(public_key)  # raises on a bad key (caller error)
    try:
        key.verify(base64.b64decode(signature), canonical_bytes(manifest))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying a downloaded manifest payload."""

    valid: bool
    signed: bool  # was there a signature at all (free-tier manifests are unsigned)
    tier: str | None
    verified_flag: bool | None  # the manifest's own self-declared `verified`
    reason: str

    @property
    def ok(self) -> bool:
        return self.valid


def _extract_manifest_and_signature(
    payload: dict[str, Any],
    signature_override: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Pull ``(manifest, signature)`` from a downloaded payload.

    Accepts both shapes documented for pdfmux Cloud:
      * a ``GET /v1/jobs/{id}`` response — ``{"manifest": {...}, "signature": "..."}``
      * a bare manifest object (with the signature supplied via ``--signature``).
    """
    if signature_override is not None and "manifest" in payload:
        return payload["manifest"], signature_override
    if "manifest" in payload and isinstance(payload["manifest"], dict):
        sig = signature_override if signature_override is not None else payload.get("signature")
        return payload["manifest"], sig
    # Treat the whole object as the manifest; signature must come from the flag.
    return payload, signature_override


def verify_payload(
    payload: dict[str, Any],
    public_key: str,
    signature_override: str | None = None,
) -> VerifyResult:
    """Verify a downloaded manifest payload against a public key.

    Distinguishes the three real outcomes a customer cares about: a valid signed
    manifest, an *unsigned* free-tier manifest (nothing to verify — that is the
    difference the paid tier buys), and a signed manifest that FAILS (tampered or
    wrong key).
    """
    manifest, signature = _extract_manifest_and_signature(payload, signature_override)
    tier = manifest.get("tier") if isinstance(manifest, dict) else None
    verified_flag = manifest.get("verified") if isinstance(manifest, dict) else None

    if not signature:
        return VerifyResult(
            valid=False,
            signed=False,
            tier=tier,
            verified_flag=verified_flag,
            reason=(
                "manifest carries no signature — this is a free-tier (unsigned) "
                "extraction; there is nothing to verify. A paid Verified extraction "
                "ships a signature."
            ),
        )

    valid = verify_manifest(manifest, signature, public_key)
    return VerifyResult(
        valid=valid,
        signed=True,
        tier=tier,
        verified_flag=verified_flag,
        reason=(
            "signature is valid — the manifest is byte-for-byte what pdfmux signed"
            if valid
            else "signature is INVALID — the manifest was altered or signed by a different key"
        ),
    )


def fetch_public_key(url: str = DEFAULT_PUBKEY_URL, timeout: float = 10.0) -> str:
    """Fetch the published public key (PEM). Network is opt-in — offline
    verification with a locally-cached key is the default and the whole point."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — https only, caller-chosen
        if resp.status != 200:
            raise ManifestVerificationError(f"fetching public key returned HTTP {resp.status}")
        return resp.read().decode("utf-8")
