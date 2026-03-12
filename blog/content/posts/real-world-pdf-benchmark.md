+++
date = '2026-03-10'
draft = false
title = 'We ran pdfmux on Tesla 10-Ks, Uber S-1s, and Supreme Court opinions. Here is what happened.'
description = 'Real-world benchmark of pdfmux vs PyMuPDF and pymupdf4llm across 11 public documents — 1,422 pages of SEC filings, academic papers, legal opinions, and government reports. Full results with every number.'
tags = ['benchmark', 'pdf-extraction', 'python', 'rag', 'llm', 'real-world', 'pymupdf4llm', 'sec-filings']
slug = 'real-world-pdf-benchmark'
og_image = '/og/benchmark.png'
+++

**TL;DR**: We ran pdfmux across 11 real-world public documents — SEC filings, academic papers, Supreme Court opinions, and government reports — totaling 1,422 pages. pdfmux processed them at **6.8 pages/sec**, was 7% faster than pymupdf4llm overall, extracted more content on 8 out of 11 documents, hit 100% confidence on every single one, and recovered content where pymupdf4llm crashed entirely. Zero cloud calls. Zero cost. Every number is below.

---

## The problem nobody talks about

Most PDF extraction benchmarks use synthetic test cases — clean, well-formatted documents designed to make tools look good. But the PDFs that actually break your RAG pipeline at 2am are the messy ones:

- **SEC filings** with 500+ pages mixing legal text, financial tables, and cover-page graphics
- **Academic papers** where equations are embedded as images inside otherwise digital text
- **Legal opinions** with footnote hierarchies three levels deep
- **Government reports** where the cover is a scanned image and the rest is digital

We built pdfmux because we kept hitting these edge cases. So we decided to benchmark it against the documents people actually need to process — not the ones that make for easy demos.

## The test corpus: 1,422 pages of real documents

We assembled 11 publicly available documents covering the most common PDF extraction challenges. Every document is downloadable — you can reproduce this entire benchmark yourself.

| Document | Type | Pages | Size | Why it's hard |
|----------|------|------:|-----:|---------------|
| Tesla 10-K (2024) | SEC annual filing | 144 | 1.6 MB | Financial tables, forward-looking statements |
| Apple 10-K | SEC annual filing | 121 | 1.0 MB | Dense financial tables, footnotes |
| Berkshire Hathaway AR | Annual report | 150 | 1.8 MB | Warren Buffett's letter + complex financials |
| Uber S-1 (2019) | SEC IPO filing | 522 | 5.4 MB | Massive document, 15 graphical pages, complex tables |
| Airbnb Pitch Deck | Startup presentation | 12 | 842 KB | Fully graphical/image-based slides |
| Attention Is All You Need | Academic paper | 15 | 2.1 MB | Equations, architecture diagrams |
| BERT | Academic paper | 16 | 757 KB | Dense results tables |
| Trump v. United States | Supreme Court opinion | 119 | 519 KB | Legal citations, footnotes, syllabus |
| EY 10-K Guide | Professional services | 154 | 2.5 MB | Mixed layouts, branded graphics |
| FDA New Drug Therapy | Government report | 34 | 10 MB | Cover graphic, data tables |
| FDA PDUFA Performance | Government report | 135 | 1.4 MB | Tables, structured regulatory data |

**Total: 1,422 pages across 28.4 MB of real-world PDFs.**

## The tools we compared

We tested three approaches on every document:

1. **PyMuPDF raw** — `page.get_text()` on every page. Zero formatting, zero structure. The baseline.
2. **pymupdf4llm** — The most popular PDF-to-Markdown library for LLM pipelines. Adds headers, bold, lists, and table detection.
3. **pdfmux** — Our self-healing pipeline that classifies each page, routes to the best extractor, audits quality, and re-extracts failures.

**Methodology:** All tests ran on an Apple Silicon Mac with Python 3.12. pdfmux used "fast" quality mode — no Docling, no OCR models, just intelligent routing and auditing on top of the same pymupdf4llm backend. Each document was processed 3 times; we report the median. Total cost: **$0** — everything runs locally.

> For context: LlamaParse charges per page, Unstructured has cloud-only features, and many "AI-powered" extractors require API keys. pdfmux wraps open-source extractors and adds intelligence on top — no API keys, no cloud calls, no per-page fees.

## Speed: every document, no cherry-picking

The common assumption: "a pipeline wrapper must be slower." Here's the full table — including the documents where pdfmux was slower:

| Document | pymupdf4llm | pdfmux | Delta |
|----------|------------:|-------:|------:|
| Airbnb Pitch Deck (12p) | 0.33s | 0.24s | **27% faster** |
| Attention Paper (15p) | 2.00s | 2.09s | ~same |
| BERT Paper (16p) | 3.65s | 4.24s | 16% slower |
| FDA Drug Therapy (34p) | 5.81s | 5.87s | ~same |
| Supreme Court (119p) | 6.13s | 6.45s | ~same |
| Apple 10-K (121p) | 16.88s | 15.32s | **9% faster** |
| FDA PDUFA (135p) | 47.20s | 38.91s | **18% faster** |
| Tesla 10-K (144p) | 21.93s | 22.22s | ~same |
| Berkshire AR (150p) | 24.29s | 23.38s | **4% faster** |
| EY 10-K Guide (154p) | 22.89s | 22.02s | **4% faster** |
| Uber S-1 (522p) | 72.61s | 68.16s | **6% faster** |
| **Total (1,422 pages)** | **223.72s** | **208.90s** | **7% faster** |

**That's 6.8 pages/sec** for pdfmux vs 6.4 pages/sec for pymupdf4llm.

The pattern is clear: on small files (15-16 pages), pdfmux's classification overhead is noticeable — BERT is 16% slower in wall clock time (0.6 seconds). On larger documents where it matters, pdfmux is consistently faster because page-level routing skips unnecessary work. The Uber S-1 saved 4.5 seconds. The FDA PDUFA report saved 8.3 seconds.

The overhead is constant (~0.5s for classification + audit). The savings scale with document size.

## Content quality: pdfmux wins on 8 out of 11 documents

This is where it gets interesting. We compared total extracted characters — more characters generally means more content recovered, fewer gaps in your RAG pipeline.

| Document | pymupdf4llm | pdfmux | Delta | Winner |
|----------|------------:|-------:|------:|--------|
| Airbnb Pitch Deck | **0 (crashed)** | 2,766 | +2,766 | pdfmux |
| Uber S-1 | 1,713,729 | 1,746,341 | +32,612 (+1.9%) | pdfmux |
| Attention Paper | 40,498 | 41,326 | +828 (+2.0%) | pdfmux |
| Supreme Court | 248,104 | 248,690 | +586 (+0.24%) | pdfmux |
| EY 10-K Guide | 528,984 | 529,390 | +406 (+0.08%) | pdfmux |
| Tesla 10-K | 485,141 | 485,534 | +393 (+0.08%) | pdfmux |
| Apple 10-K | 445,323 | 445,683 | +360 (+0.08%) | pdfmux |
| Berkshire AR | 498,793 | 498,986 | +193 (+0.04%) | pdfmux |
| BERT Paper | 66,230 | 66,200 | -30 (-0.05%) | pymupdf4llm |
| FDA Drug Therapy | 65,669 | 65,591 | -78 (-0.12%) | pymupdf4llm |
| FDA PDUFA | 238,245 | 237,877 | -368 (-0.15%) | pymupdf4llm |

**pdfmux extracts more content on 8 of 11 documents.** The 3 losses are trivial — 30, 78, and 368 characters respectively (whitespace normalization differences). The wins include a complete crash recovery and 32K characters of additional content from a 522-page SEC filing.

## When pymupdf4llm crashes, pdfmux keeps going

The Airbnb pitch deck is a 12-page presentation where half the pages are image-heavy — text rendered as graphics. This is the kind of document that breaks extraction pipelines in production.

| Tool | Output | What happened |
|------|--------|---------------|
| PyMuPDF raw | 2,738 chars | Extracted text fragments, missed graphical content |
| pymupdf4llm | **0 chars** | **Threw an unhandled error and produced nothing** |
| pdfmux | 2,766 chars | Detected 6 graphical pages, extracted available text, flagged pages for OCR |

**In a production RAG pipeline, this is the difference between a silent failure and a graceful recovery.** pymupdf4llm returns nothing — your pipeline silently indexes an empty document. pdfmux recovers what it can, tells you exactly which pages need OCR treatment, and never crashes.

## The Uber S-1: 32,612 characters recovered

Uber's 522-page IPO filing is the stress test. It has dense legal text, complex financial tables, and 15 pages with heavy graphical content (charts, infographics, full-page images).

```
pymupdf4llm: 1,713,729 chars
pdfmux:      1,746,341 chars  (+32,612 chars, +1.9%)
```

That 32K character difference is roughly **8 pages worth of content** — likely from the 15 graphical pages that pdfmux identified and handled with its fallback extraction path. For a RAG pipeline ingesting SEC filings, that's the difference between having complete risk factors and missing them entirely.

## Page-level intelligence: what other tools don't tell you

What makes pdfmux different isn't just extraction — it's **understanding what each page is** before extracting it. Every other tool treats all pages the same. pdfmux classifies first, then routes.

Across all 11 documents, pdfmux identified:

- **27 graphical pages** that would produce degraded output with text-only extraction
- **6 empty pages** (title pages, dividers) that other tools silently skip or mishandle
- **4 documents with complex tables** that benefit from specialized extraction
- **100% confidence** on every document — the audit validated every single page

Here's the per-document breakdown:

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

This classification intelligence feeds into routing decisions. Digital pages go to the fast path. Graphical pages get flagged for OCR. Table pages get specialized extraction. Empty pages are reported, not silently dropped.

## What does the output actually look like?

Here's a snippet from the Supreme Court opinion extraction — clean, LLM-ready markdown with zero post-processing:

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

Headers preserved. Italics for case citations. Paragraph structure intact. Feed this directly to GPT-4, Claude, or any LLM — no regex cleanup needed.

## The full scorecard

| Metric | PyMuPDF raw | pymupdf4llm | pdfmux |
|--------|------------:|------------:|-------:|
| Total time (1,422 pages) | 3.16s | 223.72s | **208.90s** |
| Pages per second | 450.0 | 6.4 | **6.8** |
| Total chars extracted | 4,373,506 | 4,330,716 | **4,368,384** |
| Documents with more content | — | 3 of 11 | **8 of 11** |
| Avg confidence | — | — | **100%** |
| Failures | 0 | 1 (Airbnb crash) | **0** |
| Graphical pages detected | 0 | 0 | **27** |
| Empty pages detected | 0 | 0 | **6** |
| Cloud calls required | 0 | 0 | **0** |
| Cost | $0 | $0 | **$0** |

PyMuPDF raw is fast but produces unstructured text — no headers, no tables, no markdown. pymupdf4llm adds structure but crashes on edge cases and misses graphical content. pdfmux wraps the same extraction backends, adds classification and auditing intelligence, and comes out faster, more complete, and more reliable.

## What's next

This benchmark ran in **fast** mode — pdfmux's lightest configuration. We're working on:

- **Standard mode benchmarks** with Docling table extraction and OCR for graphical pages
- **RAG quality evaluation** — not just character counts, but retrieval accuracy on questions about these documents
- **Larger corpus** — 100+ documents across more industries (healthcare, insurance, real estate)
- **Head-to-head vs LlamaParse, Unstructured, and Marker** on the same corpus

Follow the [GitHub repo](https://github.com/NameetP/pdfmux) for updates.

## Try it yourself

Every PDF in this benchmark is publicly available. Reproduce the entire thing:

```bash
pip install pdfmux

# Quick conversion
pdfmux convert tesla-10k.pdf -o output.md

# See per-page classification
pdfmux analyze tesla-10k.pdf

# Benchmark all backends
pdfmux bench paper.pdf

# With confidence scores
pdfmux convert document.pdf -o out.md --confidence
```

For the full benchmark script and raw results, see the [benchmarks directory on GitHub](https://github.com/NameetP/pdfmux/tree/main/benchmarks).

---

*pdfmux is MIT licensed, runs entirely locally, and has zero cloud dependencies. [GitHub](https://github.com/NameetP/pdfmux) · [PyPI](https://pypi.org/project/pdfmux/) · [Docs](https://pdfmux.com)*
