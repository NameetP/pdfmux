<!-- page: 1 -->

Network Working Group — P. Deutsch
Request for Comments: 1951 — Aladdin Enterprises
Category: Informational — May 1996

# DEFLATE Compressed Data Format Specification version 1.3

## Status of This Memo

This memo provides information for the Internet community. This memo does not specify an Internet standard of any kind. Distribution of this memo is unlimited.

## IESG note:

The IESG takes no position on the validity of any Intellectual Property Rights statements contained in this document.

## Notices

Copyright © 1996 L. Peter Deutsch

Permission is granted to copy and distribute this document for any purpose and without charge, including translations into other languages and incorporation into compilations, provided that the copyright notice and this notice are preserved, and that any substantive changes or deletions from the original are clearly marked.

A pointer to the latest version of this and related documentation in HTML format can be found at the URL <ftp://ftp.uu.net/graphics/png/documents/zlib/zdoc-index.html>.

## Abstract

This specification defines a lossless compressed data format that compresses data using a combination of the LZ77 algorithm and Huffman coding, with efficiency comparable to the best currently available general-purpose compression methods. The data can be produced or consumed, even for an arbitrarily long sequentially presented input data stream, using only an *a priori* bounded amount of intermediate storage. The format can be implemented readily in a manner not covered by patents.

Deutsch — Informational — [Page 1]

<!-- page: 2 -->

RFC 1951 — DEFLATE Compressed Data Format Specification — April 1996

## Contents

1. Introduction — 2
   - 1.1 Purpose — 2
   - 1.2 Intended audience — 2
   - 1.3 Scope — 2
   - 1.4 Compliance — 3
   - 1.5 Definitions of terms and conventions used — 3
   - 1.6 Changes from previous versions — 3
2. Compressed representation overview — 3
3. Detailed specification — 4
   - 3.1 Overall conventions — 4
     - 3.1.1 Packing into bytes — 4
   - 3.2 Compressed block format — 5
     - 3.2.1 Synopsis of prefix and Huffman coding — 5
     - 3.2.2 Use of Huffman coding in the "deflate" format — 6
     - 3.2.3 Details of block format — 7
     - 3.2.4 Non-compressed blocks (BTYPE=00) — 9
     - 3.2.5 Compressed blocks (length and distance codes) — 9
     - 3.2.6 Compression with fixed Huffman codes (BTYPE=01) — 10
     - 3.2.7 Compression with dynamic Huffman codes (BTYPE=10) — 10
   - 3.3 Compliance — 12
4. Compression algorithm details — 12
5. References — 13
6. Security Considerations — 13
7. Source code — 13
8. Acknowledgements — 14
9. Author's Address — 14

## 1  Introduction

### 1.1  Purpose

The purpose of this specification is to define a lossless compressed data format that:

- Is independent of CPU type, operating system, file system, and character set, and hence can be used for interchange;
- Can be produced or consumed, even for an arbitrarily long sequentially presented input data stream, using only an *a priori* bounded amount of intermediate storage, and hence can be used in data communications or similar structures such as Unix filters;

Deutsch — Informational — [Page 2]
