# pdfmux licensing

pdfmux is **open-core**. Here is exactly what that means, so there are no surprises.

## What's MIT (this repository) — free, forever

Everything in this repository is licensed under the [MIT License](LICENSE):

- the `pdfmux` Python library — extraction pipeline, per-page confidence scoring, OCR cascade, CLI;
- the pdfmux **MCP server**;
- the LangChain and LlamaIndex integrations.

Use it commercially, fork it, embed it — the MIT terms are the whole agreement for this code, and **every released version stays MIT permanently**.

## What's patent-pending and licensed separately — not in this repo

The **decision-trace** extraction method — the persisted per-page decision trace (including retained rejected candidates), the monotonic repair guard, and the runtime calibration loop, alone and in combination — is **patent-pending**: US Provisional Patent Application **No. 64/106,302** (priority date 2026-07-07).

That method is **not included in this MIT repository.** It ships in **pdfmux Cloud / Pro** under a separate commercial license. The MIT License above covers only the code in this repository and does **not** grant any license — patent or copyright — to the decision-trace method.

## Why the split

The open-source library is free so anyone can extract PDFs reliably, and so the ecosystem (LangChain, LlamaIndex, MCP hosts) can build on it without friction. The patented method funds the project and is available to teams that want the audit-transparent, self-healing extraction guarantees in production.

For commercial licensing, reach us via [pdfmux.com](https://pdfmux.com) or open a GitHub issue.
