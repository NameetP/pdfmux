# VERIFIER-VALIDATION — pdfmux `verify_extraction` FP/FN measurement (GT-0)

**Engine measured:** `pdfmux.verifier.verify_extraction` (shipped v1.8.2, byte-identical to `origin/main` — this harness never modifies it).  
**Corpus:** GT-0, 24 documents across 8 categories (manifest SHA-256 `12f3a360d0b02c84cfd8cf62fcb6eabbb12821e3f516777defbea5950b4d6823`).  
**Regenerate:** `python pdfmux-bench/validate_verifier.py --corpus pdfmux-bench/corpus --out pdfmux-bench/VERIFIER-VALIDATION.md` (deterministic; two runs are byte-identical).

Of the 24 documents, pdfmux's fast audit re-derives a usable source text layer (≥ 200 chars) for **15 ("verifiable")**. The rest split into **6 pure image-only scans** (source < 20 chars → verifier verdict `unverifiable`, an honest refusal) and **3 text PDFs the audit under-reads** (a valid text layer, but pdfmux's cheap re-derivation extracted only a handful of chars — see §6). FP and FN are measured only on the verifiable documents.

## 1. False positives — did the verifier condemn a *correct* extraction?

A **false positive** = `FAIL` verdict or a `silent_drop` cry on an extraction we know is correct. `clean-gt` feeds the human image-transcribed ground truth (independent, known-good). `clean-pdfmux` feeds pdfmux's own extraction (production self-audit).

| Clean case | Verifiable docs | False positives | FP rate | Soft REVIEWs (non-fatal) | Unverifiable (declined) |
|---|---:|---:|---:|---:|---:|
| `clean-gt` | 15 | 1 | 6.7% | 3 | 6 |
| `clean-pdfmux` | 15 | 1 | 6.7% | 4 | 6 |

- `clean-gt` false positives: `irs-f1040`
- `clean-gt` soft REVIEWs (correct extraction, non-fatal flag): `irs-fw9`, `irs-fw4`, `rfc1951-deflate`
- `clean-pdfmux` false positives: `irish-census-1926`
- `clean-pdfmux` soft REVIEWs (correct extraction, non-fatal flag): `irs-fw9`, `irs-fw4`, `arxiv-1706.03762-attention`, `rfc1951-deflate`

## 2. False negatives — did the verifier miss a *known* defect?

Each defect is seeded deterministically (seeded by doc id; no unseeded randomness). A **false negative** = a clean `PASS` (no flag at all) on the defective extraction, on a verifiable source. Defects seeded on image-only scans are `declined` (source `unverifiable`) — the verifier refuses rather than mis-certifying; those are not scored as detections or misses.

| Defect class | Expected | Applicable (verifiable) | Detected | False negatives | FN rate | Declined (unverifiable src) |
|---|---|---:|---:|---:|---:|---:|
| `drop_page` | fail | 15 | 15 | 0 | 0.0% | 9 |
| `truncate_table` | flag | 5 | 1 | 4 | 80.0% | 0 |
| `inject_offsource` | flag | 15 | 15 | 0 | 0.0% | 9 |
| `severe_loss` | fail | 15 | 15 | 0 | 0.0% | 9 |

- `truncate_table` false negatives (verifier PASSed defective content): `arxiv-2203.02155-instructgpt`, `bls-empsit`, `gao-24-106214`, `irish-census-1926`

## 3. Confusion table (verdict × condition)

Rows = actual condition; columns = verifier verdict (`unverifiable` = at least one page declined and not otherwise FAILed).

| Condition | PASS | REVIEW | FAIL | unverifiable |
|---|---:|---:|---:|---:|
| clean (correct) | 22 | 12 | 2 | 12 |
| seeded defect | 10 | 20 | 35 | 12 |

## 4. Per-category breakdown

| Category | Docs | Verifiable | Non-verifiable src | clean-gt FP | clean-gt REVIEW | Defect FN | Defects applied |
|---|---:|---:|---:|---:|---:|---:|---:|
| forms | 3 | 3 | 0 | 1 | 2 | 0 | 10 |
| academic | 3 | 3 | 0 | 0 | 0 | 0 | 9 |
| complex-tables | 3 | 3 | 0 | 0 | 0 | 3 | 12 |
| digital-native | 3 | 2 | 1 | 0 | 1 | 0 | 6 |
| rtl | 3 | 1 | 2 | 0 | 0 | 0 | 3 |
| scanned | 3 | 0 | 3 | 0 | 0 | 0 | 0 |
| handwriting | 3 | 0 | 3 | 0 | 0 | 0 | 0 |
| degraded | 3 | 3 | 0 | 0 | 0 | 1 | 10 |

## 5. `unverifiable` rate and source re-derivation failures

- **Pure image-only scans (source < 20 chars → verifier verdict `unverifiable`):** 6 / 24 (25.0%).
- **Clean verification cases that hit at least one `unverifiable` page:** 12 / 48 (25.0%).
- Image-only documents: `telegram-garfield-1881`, `ndl-treaty-report`, `ndl-hainan-memo`, `letter-peabody-1863`, `letter-beloved-friend-garrison`, `letter-dear-friend-garrison`.
- **Audit under-read (valid text PDF, but pdfmux's fast re-derivation extracted < 200 chars):** 3 / 24 — `rfc9110-http-semantics` (39 chars), `ar-morocco-petitions-44-14` (73 chars), `ar-morocco-langculture-04-16` (137 chars). These are NOT image-only; the text layer is present and correct, but pdfmux's audit could not read it (see §6).

## 6. Findings & caveats (reported, never patched)

Per the patent red line, nothing below was fixed in the engine — it is measured and written up:

- **clean-gt hard-FP rate = 6.7%** (1/15 verifiable docs). On documents it can read, the verifier hard-`FAIL`s a faithful, independent human transcription at this rate.
  - **`irs-f1040` (root cause):** verdict `FAIL` via `low_alignment, low_coverage, severe_content_loss`. pdfmux's own source re-derivation of this page is **19× larger** than the correct extraction (111,734 vs 5,987 chars) — pymupdf4llm expands this dense form into empty table-cell scaffolding (`|Col2|Col3|…`), so a concise extraction scores only 5% coverage / 3% alignment against that inflated 'truth' and trips `severe_content_loss`. The verifier's ground truth is the bug, not the extraction. A real, publishable calibration weakness on dense forms.
- **Over-flagging: 3 correct extractions drew a soft `REVIEW`** (`irs-fw9`, `irs-fw4`, `rfc1951-deflate`) — short / boxed-form / monospace pages where a correct extraction's character-coverage or alignment lands just under a threshold. Non-fatal (REVIEW, not FAIL), but they would show as amber on a public /audit tool rather than a clean pass.
- **clean-pdfmux hard-FP rate = 6.7%** (1/15), plus 4 soft `REVIEW`s. pdfmux verifying pdfmux is **not** a free pass: the production self-audit path hard-`FAIL`s its OWN correct extraction of `irish-census-1926` (the landscape multi-column census schedule), because the cheap fast-pass source re-derivation disagrees with its own full extraction on that geometry. On a self-audit, the two passes should never diverge this far — worth understanding before shipping /audit on real files.
- **Blind spot: partial table truncation, FN rate 80.0%** (4/5 tabled docs). Deleting up to half a table's rows passed clean because (a) the integrity check (`_has_table`) tests *presence* (≥2 pipe rows), not completeness, and (b) the deleted cells are a small fraction of the page's total tokens, so coverage/alignment stay above threshold. The certifier catches a whole dropped page but not a quietly shortened table.
- **Caught reliably (0% FN): `drop_page`, `inject_offsource`, `severe_loss`.** Whole-page silent drops, off-source hallucination, and catastrophic (~90%) content loss are detected on every verifiable document.
- **Pure scans are refused, not mis-passed.** The 4th `unverifiable` state does its job: image-only scans (telegram, tribunal typescripts, cursive letters) never receive a false PASS or a false FAIL — the verifier declines to certify what it cannot re-derive.
- **Source re-derivation fails on Arabic RTL and some subsetted-font PDFs (`rfc9110-http-semantics`, `ar-morocco-petitions-44-14`, `ar-morocco-langculture-04-16`).** These are valid, correct text PDFs, but pdfmux's fast audit pass (pymupdf4llm + column-reorder + heading-injection) extracted only tens of characters from a full page — Arabic right-to-left text and the RFC 9110 font subset defeat it. Because the re-derived 'source truth' is nearly empty, the verifier cannot judge these documents at all: it neither certifies nor faults them, but a downstream /audit user would get an unhelpful near-empty comparison. A calibration gap to disclose alongside the scan limitation — the certifier's blind spot is not only pixels; it is any text the fast pass can't read (RTL, exotic font encodings).
- **All five decision thresholds remain the hand-set v1.8.2 constants** (`LOW_COVERAGE_THRESHOLD=0.60`, `LOW_ALIGNMENT_THRESHOLD=0.55`, `HIGH_HALLUCINATION_THRESHOLD=0.45`, `LOW_CONFIDENCE_THRESHOLD=0.55`, `SILENT_DROP_RECOVERY_THRESHOLD=0.50`). This corpus measures them; it does not tune them.

---

### Methodology notes

- Each source PDF is clipped to its manifest `pages` range before verification, so the source the verifier re-derives matches the pages the ground truth was authored from.
- `clean-gt` and every seeded defect are fed as page-aligned JSON so the verifier runs its segmented (Mode A) per-page path — the same path that detects real silent drops.
- Defect seeding is a pure function of the doc id (`seed_for`); the seeding tests (`pdfmux-bench/tests/test_seeding.py`) assert two runs are identical.

