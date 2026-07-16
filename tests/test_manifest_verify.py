"""Tests for offline Cloud-manifest verification (``pdfmux verify-manifest``).

The critical property is **byte-for-byte agreement with the Cloud signer**: the
worker signs ``json.dumps(manifest, sort_keys=True)`` with an Ed25519 private key,
and this MIT-core verifier must accept exactly those signatures and reject any
tampering. These tests reproduce the Cloud signing locally with an ephemeral
keypair, so a drift in canonicalization on either side fails here.
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from pdfmux.cli import app
from pdfmux.manifest_verify import canonical_bytes, verify_manifest, verify_payload

runner = CliRunner()


# --- helpers: reproduce the Cloud signer exactly ---------------------------


def _keypair() -> tuple[Ed25519PrivateKey, str]:
    priv = Ed25519PrivateKey.generate()
    pub_pem = (
        priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return priv, pub_pem


def _sign(manifest: dict, priv: Ed25519PrivateKey) -> str:
    # Byte-identical to worker/manifest_signing.sign_manifest.
    return base64.b64encode(priv.sign(canonical_bytes(manifest))).decode("ascii")


def _paid_manifest() -> dict:
    return {
        "source": "invoice.pdf",
        "pages": 3,
        "confidence": 0.97,
        "tier": "verified",
        "verified": True,
        "signature_algorithm": "ed25519",
    }


# --- module-level API ------------------------------------------------------


def test_valid_signature_verifies():
    priv, pub = _keypair()
    m = _paid_manifest()
    assert verify_manifest(m, _sign(m, priv), pub) is True


def test_tampered_field_fails():
    priv, pub = _keypair()
    m = _paid_manifest()
    sig = _sign(m, priv)
    m["pages"] = 4  # alter after signing
    assert verify_manifest(m, sig, pub) is False


def test_relabel_free_to_verified_fails():
    """The signature covers `tier`/`verified` — a free manifest can't be
    re-labelled verified without breaking it."""
    priv, pub = _keypair()
    m = {"source": "x.pdf", "tier": "free", "verified": False}
    sig = _sign(m, priv)
    m["tier"] = "verified"
    m["verified"] = True
    assert verify_manifest(m, sig, pub) is False


def test_wrong_key_fails():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    m = _paid_manifest()
    assert verify_manifest(m, _sign(m, priv), other_pub) is False


def test_key_order_independence():
    """A verifier that rebuilt the dict in a different key order must still pass —
    canonicalization sorts keys."""
    priv, pub = _keypair()
    m = _paid_manifest()
    sig = _sign(m, priv)
    reordered = dict(reversed(list(m.items())))
    assert verify_manifest(reordered, sig, pub) is True


# --- payload-level (the /v1/jobs/{id} shape) -------------------------------


def test_verify_payload_valid_signed():
    priv, pub = _keypair()
    m = _paid_manifest()
    payload = {"status": "succeeded", "manifest": m, "signature": _sign(m, priv)}
    res = verify_payload(payload, pub)
    assert res.valid and res.signed and res.tier == "verified"


def test_verify_payload_unsigned_free_tier():
    _, pub = _keypair()
    payload = {"manifest": {"source": "x.pdf", "tier": "free", "verified": False}}
    res = verify_payload(payload, pub)
    assert res.valid is False and res.signed is False
    assert "free-tier" in res.reason


def test_verify_payload_tampered_invalid():
    priv, pub = _keypair()
    m = _paid_manifest()
    sig = _sign(m, priv)
    m["source"] = "different.pdf"
    res = verify_payload({"manifest": m, "signature": sig}, pub)
    assert res.valid is False and res.signed is True


# --- CLI -------------------------------------------------------------------


def test_cli_verified(tmp_path):
    priv, pub = _keypair()
    m = _paid_manifest()
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"manifest": m, "signature": _sign(m, priv)}))
    key = tmp_path / "pub.pem"
    key.write_text(pub)

    result = runner.invoke(app, ["verify-manifest", str(job), "--pubkey", str(key)])
    assert result.exit_code == 0, result.output
    assert "VERIFIED" in result.output


def test_cli_tampered_exits_4(tmp_path):
    priv, pub = _keypair()
    m = _paid_manifest()
    sig = _sign(m, priv)
    m["pages"] = 999
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"manifest": m, "signature": sig}))
    key = tmp_path / "pub.pem"
    key.write_text(pub)

    result = runner.invoke(app, ["verify-manifest", str(job), "--pubkey", str(key)])
    assert result.exit_code == 4, result.output
    assert "INVALID" in result.output


def test_cli_unsigned_exits_4(tmp_path):
    _, pub = _keypair()
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"manifest": {"tier": "free", "verified": False}}))
    key = tmp_path / "pub.pem"
    key.write_text(pub)

    result = runner.invoke(app, ["verify-manifest", str(job), "--pubkey", str(key)])
    assert result.exit_code == 4, result.output
    assert "UNSIGNED" in result.output


def test_cli_json_format(tmp_path):
    priv, pub = _keypair()
    m = _paid_manifest()
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"manifest": m, "signature": _sign(m, priv)}))
    key = tmp_path / "pub.pem"
    key.write_text(pub)

    result = runner.invoke(
        app, ["verify-manifest", str(job), "--pubkey", str(key), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["valid"] is True and parsed["tier"] == "verified"


def test_cli_no_key_is_usage_error(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"manifest": _paid_manifest(), "signature": "x"}))
    result = runner.invoke(app, ["verify-manifest", str(job)])
    assert result.exit_code == 2, result.output
    assert "No public key" in result.output
