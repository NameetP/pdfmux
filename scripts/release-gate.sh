#!/usr/bin/env bash
# release-gate.sh — run against the BUILT artifact before any PyPI upload.
#
# Why this exists, concretely:
#   * 1.8.1 shipped with a "#2 on opendataloader-bench" claim that was not then
#     verified. A source grep would not have caught it; the claim was in the
#     built METADATA.
#   * 1.8.3 shipped reporting its own version as 1.8.2 (a hardcoded literal that
#     drifted from pyproject.toml) — and that string is stamped into every
#     certification manifest as the `tool` field.
#   * 1.8.4 shipped with a superseded benchmark score (0.905) in the PyPI
#     Summary while its own long description said 0.903. The ad-hoc gate used at
#     the time matched banned *phrases*, not superseded *numbers*, so it passed.
#
# The pattern in all three: the artifact said something the repo no longer
# believed. This gate reads the artifact, not the repo.
#
# Usage: bash scripts/release-gate.sh dist/pdfmux-<version>-py3-none-any.whl
# Exit 0 = safe to upload. Exit 1 = do not upload.

set -uo pipefail

WHL="${1:-}"
[ -n "$WHL" ] && [ -f "$WHL" ] || { echo "usage: release-gate.sh <wheel>" >&2; exit 2; }

# --- Canonical facts. Update here, deliberately, when the truth changes. -----
# The reproduced opendataloader-bench result for the CURRENT engine. Any other
# 0.9xx score presented as pdfmux's overall is superseded and must not ship.
CANON_BENCH="0.903"
SUPERSEDED_BENCH="0\.905|0\.900|0\.918|0\.887|0\.852"
# Claims that were once published and are now known false, or links that 404.
#
# "#1 free" / "#1 among free" is banned on evidence: the engine ahead of pdfmux
# on this benchmark is `opendataloader-hybrid` (0.909), which runs
# opendataloader-pdf (Apache-2.0, ~28k stars) with hybrid="docling-fast" — no API
# key, no token, no network call in the adapter at all. It is free and open
# source, so "#1 free" is not imprecise, it is false. pdfmux is #2 of the 8
# engines measured; say that instead.
BANNED='verifiedextraction\.org|precision 1\.000|runtime calibration loop|ships in pdfmux Cloud|1,000 free pages|#1 free|#1 among free|paid hybrid engine'

FAILS=0
fail() { echo "  ✗ $*"; FAILS=$((FAILS+1)); }
pass() { echo "  ✓ $*"; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
unzip -qo "$WHL" -d "$TMP" || { echo "could not unpack $WHL" >&2; exit 2; }

META=$(find "$TMP" -name METADATA -path "*.dist-info/*" | head -1)
[ -n "$META" ] || { echo "no METADATA in wheel" >&2; exit 2; }

VERSION=$(grep -m1 '^Version:' "$META" | awk '{print $2}')
echo "== release gate: pdfmux ${VERSION} =="

# 1. Banned claims / dead links, anywhere in the shipped metadata.
if grep -qiE "$BANNED" "$META"; then
  fail "banned claim or dead link in METADATA:"
  grep -oiE "$BANNED" "$META" | sort -u | sed 's/^/      /'
else
  pass "no banned claims or dead links in METADATA"
fi

# 2. Same, in the shipped code (the CLI has twice carried a dead URL).
if grep -rqiE "$BANNED" "$TMP/pdfmux" 2>/dev/null; then
  fail "banned claim or dead link in shipped code:"
  grep -rloiE "$BANNED" "$TMP/pdfmux" 2>/dev/null | sed 's/^/      /'
else
  pass "no banned claims or dead links in shipped code"
fi

# 3. Superseded benchmark numbers claimed AS PDFMUX'S. This is the check 1.8.4
#    needed and lacked. It must be context-aware: the same numerals are also the
#    legitimate scores of competitors in the comparison table (Docling's 0.887
#    tables, 0.900 reading order). A gate that fires on those gets ignored —
#    which is the failure mode of the prose gates this replaces.
STALE_AS_OURS="(pdfmux[^.]{0,60}(${SUPERSEDED_BENCH})|(${SUPERSEDED_BENCH})[^.]{0,40}(overall|#1 free|#2 of all))"
if grep -qiE "$STALE_AS_OURS" "$META"; then
  fail "superseded score presented as pdfmux's (canonical is ${CANON_BENCH}):"
  grep -oiE "$STALE_AS_OURS" "$META" | sort -u | head -5 | sed 's/^/      /'
else
  pass "no superseded score claimed as pdfmux's"
fi

# 4. Summary and long description must not disagree. 1.8.4's Summary said 0.905
#    while its body said 0.903 — self-contradiction on the page most developers
#    read first.
SUMMARY=$(grep -m1 '^Summary:' "$META" || true)
if echo "$SUMMARY" | grep -qE '0\.9[0-9]{2}'; then
  if echo "$SUMMARY" | grep -qF "$CANON_BENCH"; then
    pass "Summary carries the canonical score (${CANON_BENCH})"
  else
    fail "Summary cites a benchmark score that is not ${CANON_BENCH}: ${SUMMARY}"
  fi
fi

# 5. The package must report its own version correctly. A literal that drifts
#    from pyproject.toml misstates the engine inside signed manifests.
if grep -qE '^__version__ = "[0-9]' "$TMP/pdfmux/__init__.py" 2>/dev/null; then
  fail "__init__.py hardcodes __version__ — it must derive from importlib.metadata"
else
  pass "__version__ is derived, not hardcoded"
fi

# 6. The Gemma model generation we ADVERTISE must match the one we SEND.
#    Every doc surface said "Gemma 4" for months while the provider requested
#    gemma-3-27b-it. The README's own table gave it away — "Gemma 4 | 27B IT",
#    and Gemma 4 has no 27B size. Prose drifted; the model ID never moved.
#
#    This compares the two rather than banning either string, so a genuine
#    Gemma 4 upgrade passes the moment the model IDs actually change.
#
#    It must NOT fire on prose that discusses the other generation in order to
#    rule it out — providers/gemma.py documents at length why it is Gemma 3 and
#    what adopting Gemma 4 would require. A gate that flags that gets ignored,
#    which is the failure mode of the prose gates check 3 replaces. So the code
#    scan parses the AST and looks only at string literals that are NOT
#    docstrings (i.e. text a user can actually be shown, like the CLI's
#    recommendation line), and comments are excluded for free.
GEMMA_SRC="$TMP/pdfmux/providers/gemma.py"
if [ -f "$GEMMA_SRC" ] && command -v python3 >/dev/null 2>&1; then
  GEMMA_CHECK=$(python3 - "$TMP/pdfmux" "$META" <<'PYEOF'
import ast, pathlib, re, sys

pkg, meta = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

# The generation we SEND comes from the model IDs the provider actually
# requests — `default_model` and each ModelInfo(id=...). Deliberately not a
# regex over the file: the module docstring names Gemma 4 IDs while explaining
# why they are not used, and a text scan reads those as configuration.
gemma_tree = ast.parse((pkg / "providers/gemma.py").read_text())
sent_ids = set()
for node in ast.walk(gemma_tree):
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "default_model" for t in node.targets
    ):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            sent_ids.add(node.value.value)
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ModelInfo":
        for kw in node.keywords:
            if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                sent_ids.add(kw.value.value)

sent = sorted({m.group(1) for i in sent_ids if (m := re.match(r"gemma-(\d+)-", i))})
if len(sent) != 1:
    print(f"BAD:?:provider sends mixed or unparseable Gemma generations: {sorted(sent_ids)}")
    raise SystemExit
gen = sent[0]
claims = set()

# METADATA is the PyPI page — every word of it is a user-facing claim.
claims |= set(re.findall(r"Gemma (\d+)", meta.read_text()))

# Shipped code: only non-docstring string literals can reach a user.
for py in pkg.rglob("*.py"):
    try:
        tree = ast.parse(py.read_text())
    except SyntaxError:
        continue
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = node.body[0] if node.body else None
            if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) \
               and isinstance(doc.value.value, str):
                docstrings.add(id(doc.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
           and id(node) not in docstrings:
            claims |= set(re.findall(r"Gemma (\d+)", node.value))

bad = sorted(claims - {gen})
print(f"BAD:{gen}:{','.join(bad)}" if bad else f"OK:{gen}")
PYEOF
)
  case "$GEMMA_CHECK" in
    OK:*)   pass "Gemma generation claimed matches the model IDs sent (gemma-${GEMMA_CHECK#OK:}-*)" ;;
    SKIP:*) pass "Gemma generation check skipped (${GEMMA_CHECK#SKIP:})" ;;
    BAD:*)  rest="${GEMMA_CHECK#BAD:}"
            fail "advertises Gemma ${rest#*:} but the provider sends gemma-${rest%%:*}-*" ;;
    *)      fail "Gemma generation check errored: ${GEMMA_CHECK}" ;;
  esac
fi

echo
if [ "$FAILS" -gt 0 ]; then
  echo "GATE FAILED (${FAILS}) — do not upload."
  exit 1
fi
echo "GATE PASSED — safe to upload."
