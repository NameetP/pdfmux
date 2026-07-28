#!/usr/bin/env python3
"""Verify whether pdfmux's Gemma provider can be upgraded from Gemma 3 to Gemma 4.

Background
----------
`providers/gemma.py` sends `gemma-3-27b-it`. Google publishes two Gemma 4 IDs on
the Gemini API — `gemma-4-31b-it` and `gemma-4-26b-a4b-it`:

    https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api

The upgrade was deliberately not made on 2026-07-28 because three things could
not be checked without a live key, and each silently degrades extraction rather
than failing loudly:

  1. Whether the OpenAI-compatible shim pdfmux uses (`/v1beta/openai/`) serves
     those IDs at all. Google documents them only on native `generateContent`.
  2. The pricing constants, which feed the budget cap. Wrong values mis-bill
     every routed page without any error.
  3. `max_input_tokens` — Gemma 4's larger sizes document 256K vs the 128K
     currently declared for Gemma 3.

This script answers 1 and 3 empirically and tells you exactly what to check for 2.
It is read-only: it sends two tiny requests and changes nothing.

Usage
-----
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
    python scripts/verify-gemma4.py

Exit codes: 0 = all candidates usable, 1 = none usable, 2 = partial, 3 = setup problem.
"""

from __future__ import annotations

import base64
import os
import sys

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CANDIDATES = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
CURRENT = "gemma-3-27b-it"

# 1x1 PNG — the point is to prove the vision path accepts an image at all,
# not to test OCR quality. Keep it trivial so a failure is unambiguous.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def check(client, model: str) -> tuple[bool, str]:
    """Send one minimal vision request. Returns (usable, detail)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with the single word: ok"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(_PNG_1X1).decode()
                            },
                        },
                    ],
                }
            ],
            max_tokens=16,
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a "not usable" answer
        return False, f"{type(exc).__name__}: {str(exc)[:180]}"

    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    detail = f"replied {text[:40]!r}"
    if usage is not None:
        detail += (
            f"; prompt_tokens={getattr(usage, 'prompt_tokens', '?')}"
            f" completion_tokens={getattr(usage, 'completion_tokens', '?')}"
        )
    return True, detail


def main() -> int:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("No GEMINI_API_KEY or GOOGLE_API_KEY in the environment.", file=sys.stderr)
        return 3
    try:
        import openai
    except ImportError:
        print("openai SDK missing. pip install 'pdfmux[llm-openai]'", file=sys.stderr)
        return 3

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    client = openai.OpenAI(base_url=BASE_URL, api_key=key)

    print(f"endpoint: {BASE_URL}")
    print(f"baseline: {CURRENT}\n")

    ok_baseline, detail = check(client, CURRENT)
    print(f"  [{'OK ' if ok_baseline else 'FAIL'}] {CURRENT}  {detail}")
    if not ok_baseline:
        print(
            "\nThe CURRENT model failed, so a Gemma 4 failure below would be "
            "inconclusive — fix credentials/network first.",
            file=sys.stderr,
        )
        return 3

    print()
    usable = []
    for model in CANDIDATES:
        ok, detail = check(client, model)
        print(f"  [{'OK ' if ok else 'FAIL'}] {model}  {detail}")
        if ok:
            usable.append(model)

    print("\n" + "=" * 72)
    if not usable:
        print("VERDICT: the OpenAI-compat endpoint does NOT serve Gemma 4.")
        print("Keep Gemma 3. Unknown (1) is now answered: no.")
        return 1

    print(f"VERDICT: usable over the OpenAI-compat endpoint: {', '.join(usable)}")
    print("\nUnknown (1) answered. Before changing `default_model`, still resolve:")
    print("  (2) PRICING — confirm the per-1M-token input/output rates for these")
    print("      models at https://ai.google.dev/pricing and update")
    print("      _INPUT_COST_PER_MTOK / _OUTPUT_COST_PER_MTOK in providers/gemma.py.")
    print("      These feed the budget cap; a wrong value mis-bills silently.")
    print("  (3) CONTEXT — Gemma 4's larger sizes document 256K. Update")
    print("      max_input_tokens in supported_models() (currently 128_000).")
    print("\nThen update supported_models() ids + default_model. release-gate.sh")
    print("check 6 compares advertised-vs-sent and will pass once docs follow.")
    return 0 if len(usable) == len(CANDIDATES) else 2


if __name__ == "__main__":
    raise SystemExit(main())
