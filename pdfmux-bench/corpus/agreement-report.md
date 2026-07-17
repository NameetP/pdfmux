# GT-0 Corpus — Inter-annotator Agreement Report

Eight documents (one per category) were **double-annotated**: transcribed twice, in two
independent passes, each pass authored by reading the 200-DPI rendered page image and
transcribing from the image only (never from any extractor's output). This report gives the
per-dimension agreement between the two passes and logs the adjudications that produced the
final committed ground truth.

## Method

- **Annotator model:** single annotator (this run), two passes separated by the full
  first-pass corpus authoring. This is therefore a **test–retest (intra-annotator)**
  reliability measure — an *upper bound* on true multi-annotator agreement, not a substitute
  for it. It still surfaces where transcription is genuinely ambiguous.
- **Dimensions** (computed with the benchmark's own `scoring.py`, so the yardstick matches the
  leaderboard):
  - `text_sim` — normalized character edit-distance similarity (`text_similarity`), 1.0 = identical.
  - `reading_order` — block-order agreement (`reading_order`), Kendall-tau based.
  - `heading_f1` — F1 over Markdown `#` headings (`heading_f1`).
  - `table_cell_f1` — order-independent multiset F1 over all table cell contents.
- **Regenerate:** `python agreement.py` (build tooling; consumes `annot1/` = pass 1 and
  `annot2/` = pass 2).

## Per-document agreement (pass 1 vs pass 2, before adjudication)

| Document | Category | text_sim | reading_order | heading_f1 | table_cell_f1 | has table |
|---|---|---:|---:|---:|---:|:---:|
| irs-fw9 | forms | 1.000 | 1.000 | 1.000 | 1.000 | no |
| arxiv-1512.03385-resnet | academic | 1.000 | 1.000 | 1.000 | 1.000 | no |
| gao-24-106214 | complex-tables | 1.000 | 1.000 | 1.000 | 1.000 | yes |
| rfc2616-http | digital-native | 1.000 | 1.000 | 1.000 | 1.000 | no |
| ar-morocco-petitions-44-14 | rtl | 1.000 | 1.000 | 1.000 | 1.000 | no |
| telegram-garfield-1881 | scanned | 0.978 | 0.923 | 1.000 | 1.000 | no |
| letter-peabody-1863 | handwriting | 0.999 | 0.750 | 1.000 | 1.000 | no |
| statute-1-1789 | degraded | 1.000 | 1.000 | 1.000 | 1.000 | no |

**Means:** text_sim 0.997 · reading_order 0.959 · heading_f1 1.000 · table_cell_f1 (over the
1 tabled doc, GAO) 1.000.

### Reading of these numbers

- **Clean printed/typeset/born-digital text (6 of 8) reaches 1.000 on every dimension** — forms,
  academic, the GAO complex tables, the RFC title page, the Arabic RTL legal decree, and the 1776
  Declaration of Independence print all transcribe unambiguously.
- **All disagreement is concentrated in the two handwriting/manuscript documents** — exactly
  where a human transcriber would expect it. This is the honest signal the double-annotation is
  meant to expose: the corpus labels are highly reproducible on machine-readable pages and only
  become ambiguous on cursive/handwritten sources.
- `reading_order` is the most sensitive dimension on short documents because it is computed over
  a small number of blocks (a single re-ordered or re-split block moves it a lot); on the Peabody
  letter one word was split differently across an annotation (see adjudication P-1), which is why
  its `reading_order` reads 0.750 while `text_sim` is 0.999.

## Adjudication log

Disagreements were resolved by returning to the page image and choosing the reading best
supported by the ink. The resolutions below were applied to the **final committed** GT; the
agreement numbers above are pre-adjudication.

| ID | Document | Disagreement | Adjudication | Applied to committed GT |
|---|---|---|---|---|
| T-1 | telegram-garfield-1881 | Pass 1 left the **"To"** line blank; Pass 2 read the recipient as `F. H. Silsby`. | The recipient name is legibly written on the "To" line of the form. Pass 2 is correct. | Yes — "To: F. H. Silsby" added. |
| P-1 | letter-peabody-1863 | `some thing else` (Pass 1, two words) vs `something else` (Pass 2). | The word is written as one continuous cursive form; modern normalization is one word. | Yes — resolved to "something else". |

No table-cell, heading, or numeric-value disagreements were found in the eight double-annotated
documents. The handwritten telegram's message body (`President Garfield just shot assassinated /
he is dead / R Williams / (should be R. E. Selway)`) and the Peabody letter body agreed verbatim
across both passes apart from the two items above.

## Caveat carried into the validation

`telegram-garfield-1881` and the two Garrison manuscript letters are **pure image-only scans with
no digital text layer**. pdfmux's verifier cannot re-derive their text, so it correctly returns
`unverifiable` for them regardless of GT quality (see `../VERIFIER-VALIDATION.md`). Their GT is a
faithful transcription for the bench label, but it does not (and cannot) participate in the
verifier's PASS/FAIL false-positive measurement.
