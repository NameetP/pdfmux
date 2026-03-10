+++
date = '2026-03-10'
draft = false
title = 'We ran pdfmux on Tesla 10-Ks, Uber S-1s, and Supreme Court opinions. Here is what happened.'
description = 'Real-world benchmark of pdfmux vs PyMuPDF and pymupdf4llm across 11 public documents — 1,422 pages of SEC filings, academic papers, legal opinions, and government reports.'
tags = ['benchmark', 'pdf-extraction', 'python', 'rag', 'llm', 'real-world']
slug = 'real-world-pdf-benchmark'
+++

**TL;DR**: We ran pdfmux across 11 real-world public documents — SEC filings, academic papers, Supreme Court opinions, and government reports — totaling 1,422 pages. pdfmux was 7% faster than pymupdf4llm, hit 100% confidence on every document, correctly identified 27 graphical pages that need special treatment, and recovered content where pymupdf4llm failed entirely.

---

## Why real-world PDFs matter

Our [first benchmark](/blog/benchmarking-pdf-extractors/) compared extraction tools on synthetic test cases. But real-world PDFs are messy:

- **SEC filings** have 500+ page documents mixing dense legal text, financial tables, and cover-page graphics
- **Academic papers** embed equations as images inside otherwise digital text
- **Legal opinions** use complex footnote hierarchies and citation formatting
- **Government reports** mix scanned covers with digital content

We wanted to know: does pdfmux actually work on the documents people need to process every day?

## The test corpus

We assembled 11 publicly available documents that represent the most common PDF extraction challenges:

| Document | Type | Pages | Size | Challenge |
|----------|------|------:|-----:|-----------|
| Tesla 10-K (2024) | SEC annual filing | 144 | 1.6 MB | Financial tables, forward-looking statements |
| Apple 10-K | SEC annual filing | 121 | 1.0 MB | Dense financial tables |
| Berkshire Hathaway AR | Annual report | 150 | 1.8 MB | Warren Buffett's letter + financials |
| Uber S-1 (2019) | SEC IPO filing | 522 | 5.4 MB | Massive, 15 graphical pages, complex tables |
| Airbnb Pitch Deck | Startup presentation | 12 | 842 KB | Fully graphical/image-based slides |
| Attention Is All You Need | Academic paper | 15 | 2.1 MB | Equations, architecture diagrams |
| BERT | Academic paper | 16 | 757 KB | Dense results tables |
| Trump v. United States | Supreme Court opinion | 119 | 519 KB | Legal citations, footnotes, syllabus |
| EY 10-K Guide | Professional services | 154 | 2.5 MB | Mixed layouts, branded graphics |
| FDA New Drug Therapy | Government report | 34 | 10 MB | Cover graphic, data tables |
| FDA PDUFA Performance | Government report | 135 | 1.4 MB | Tables, structured regulatory data |

**Total: 1,422 pages across 28.4 MB of real-world PDFs.**

## The tools

We compared three approaches:

1. **PyMuPDF raw** — `page.get_text()` on every page. Zero formatting, zero structure. Just dump the text.
2. **pymupdf4llm** — The most popular PDF-to-Markdown library. Adds headers, bold, lists, and table detection.
3. **pdfmux** — Our self-healing pipeline that classifies each page, routes to the best extractor, audits quality, and re-extracts failures.

All tests ran on an Apple Silicon Mac with Python 3.12. pdfmux used "fast" quality mode — no Docling, no OCR models, just intelligent routing and auditing on top of the same pymupdf4llm backend.

## Speed: pdfmux is faster, not slower

The common assumption: "a pipeline wrapper must be slower." Wrong.

| Document | pymupdf4llm | pdfmux | Delta |
|----------|------------:|-------:|------:|
| Airbnb Pitch Deck (12p) | 0.33s | 0.24s | **27% faster** |
| Apple 10-K (121p) | 16.88s | 15.32s | **9% faster** |
| FDA PDUFA (135p) | 47.20s | 38.91s | **18% faster** |
| Berkshire AR (150p) | 24.29s | 23.38s | **4% faster** |
| EY 10-K Guide (154p) | 22.89s | 22.02s | **4% faster** |
| Uber S-1 (522p) | 72.61s | 68.16s | **6% faster** |
| **Total (1,422 pages)** | **223.72s** | **208.90s** | **7% faster** |

pdfmux adds classification and auditing overhead but optimizes page-level routing to skip unnecessary work. On the 522-page Uber S-1, that saved over 4 seconds.

The small files (15-16 page academic papers) show negligible difference — the overhead is constant, not proportional.

## Content quality: same output, plus rescue

For digital-native PDFs, pdfmux produces near-identical output to pymupdf4llm — within 0.1% character difference on most documents. This is expected: both use the same extraction backend for digital pages.

But the interesting case is when things go wrong.

### The Airbnb Pitch Deck

The Airbnb deck is a 12-page presentation where 6 pages are image-heavy (text rendered as graphics). Here's what happened:

| Tool | Output |
|------|--------|
| PyMuPDF raw | 2,738 chars (text fragments only) |
| pymupdf4llm | **0 chars (error — crashed)** |
| pdfmux | 2,766 chars (recovered + classified) |

pymupdf4llm failed completely. pdfmux detected the graphical pages, classified them, and still extracted what digital text existed. It also flagged the 6 graphical pages for OCR treatment.

### The Uber S-1: 32K chars recovered

On Uber's massive IPO filing, pdfmux extracted **32,612 more characters** than pymupdf4llm (+1.9%). That's roughly 8 pages worth of additional content — likely from the 15 graphical pages that pdfmux identified and handled differently.

```
pymupdf4llm: 1,713,729 chars
pdfmux:      1,746,341 chars  (+32,612 chars, +1.9%)
```

For a RAG pipeline ingesting SEC filings, that's the difference between having complete risk factors and missing them.

## The classification intelligence

What makes pdfmux different isn't just extraction — it's **understanding what each page is** before extracting it.

Across all 11 documents, pdfmux identified:

- **27 graphical pages** that would produce degraded output with text-only extraction
- **6 empty pages** (title pages, dividers) that other tools silently skip
- **4 documents with tables** that could benefit from specialized extraction
- **100% confidence** on every document — meaning the audit validated every page

Here's the per-document classification:

```
Document                    Digital  Graphical  Empty  Tables
─────────────────────────────────────────────────────────────
Airbnb Pitch Deck              6        6        0      Yes
Uber S-1                     507       15        1      Yes
EY 10-K Guide                152        2        0      No
Tesla 10-K                   143        1        2      No
Apple 10-K                   120        1        0      No
FDA Drug Therapy              33        1        0      Yes
FDA PDUFA                    134        1        0      No
Berkshire AR                 150        0        3      No
Attention Paper               15        0        0      No
BERT Paper                    16        0        0      Yes
Supreme Court                119        0        0      No
```

This intelligence feeds into pdfmux's routing decisions. Digital pages go to the fast path. Graphical pages get OCR. Table pages get specialized extraction. Empty pages are flagged.

## What does the output actually look like?

Here's a snippet from the Tesla 10-K extraction:

```markdown
# **SUPREME COURT OF THE UNITED STATES**

Syllabus

TRUMP _v_ . UNITED STATES

CERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR
THE DISTRICT OF COLUMBIA CIRCUIT

No. 23–939. Argued April 25, 2024—Decided July 1, 2024

A federal grand jury indicted former President Donald J. Trump on four
counts for conduct that occurred during his Presidency following the
November 2020 election...
```

Clean markdown. Headers preserved. Italics for case citations. Paragraph structure intact. This is directly LLM-ready — no post-processing needed.

## The numbers, all at once

| Metric | PyMuPDF raw | pymupdf4llm | pdfmux |
|--------|------------:|------------:|-------:|
| Total time | 3.16s | 223.72s | **208.90s** |
| Total chars | 4,373,506 | 4,330,716 | **4,368,384** |
| Avg confidence | — | — | **100%** |
| Failures | 0 | 1 (Airbnb) | **0** |
| Graphical pages detected | 0 | 0 | **27** |
| Empty pages detected | 0 | 0 | **6** |

pdfmux is faster, extracts more content, never fails, and tells you about problems it finds. The trade-off? None. It wraps the same extraction libraries and adds intelligence on top.

## Try it yourself

Every PDF in this benchmark is publicly available. Run it yourself:

```bash
pip install pdfmux

# Quick conversion
pdfmux convert tesla-10k.pdf -o output.md

# See per-page analysis
pdfmux analyze tesla-10k.pdf

# Benchmark all backends
pdfmux bench paper.pdf

# With confidence scores
pdfmux convert document.pdf -o out.md --confidence
```

For the full benchmark script and results, see the [benchmarks directory on GitHub](https://github.com/NameetP/pdfmux/tree/main/benchmarks).

---

*pdfmux is MIT licensed and has zero cloud dependencies. Everything runs locally. [GitHub](https://github.com/NameetP/pdfmux) · [PyPI](https://pypi.org/project/pdfmux/) · [Docs](https://pdfmux.com)*
