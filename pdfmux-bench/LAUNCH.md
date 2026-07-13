# LAUNCH.md — making pdfmux-bench the category-standard scoreboard

This is the go-to-market for the **benchmark itself** — separate from pdfmux's own launch. The goal is narrow and specific: become the neutral scoreboard that people cite when they argue about document-extraction quality, the way ImageNet, GLUE, or SWE-bench became the reference in their fields.

> **Sequencing gate.** pdfmux is under a deliberate stay-dark posture (Gate 3 = billing-live + first customer). Do **not** Show HN or do broad public promotion of pdfmux-bench until that lifts. Everything below labeled `[stay-dark-ok]` can proceed now; everything labeled `[on-lift]` waits for the go-signal. The bench can be built, seeded with real numbers, and quietly PR'd to a few adapter authors before the public moment.

---

## The strategic bet

A benchmark becomes the standard when three things are true:

1. **It's the most complete.** It includes the tools people actually compare — especially the closed APIs (Reducto, LlamaParse, Mistral OCR) that academic benchmarks omit. Completeness is our moat: OmniDocBench won't add closed APIs (academic neutrality concerns), and vendors won't benchmark their competitors honestly. We will.
2. **It's trusted.** Reproducible by anyone, engine-agnostic scoring, losses published. The radical-honesty posture is not a nicety — it's the entire reason a neutral party would cite us over a vendor's self-benchmark.
3. **It's alive.** Refreshed quarterly, accepts adapter PRs, responds to disputes. RD-TableBench died by going stale; we win by not doing that.

pdfmux's credibility flywheel: pdfmux already publicly published "we're #2 on reading order." A company that benchmarks itself into second place in public is the *only* kind of company that can credibly run a neutral benchmark. That honesty is the wedge.

---

## Phase 0 — Seed with real numbers `[stay-dark-ok]`

You cannot launch a leaderboard with `PLACEHOLDER` cells. Before any promotion:

1. **Finish the corpus.** Pin the `needs_pin` records in `corpus/manifest.json` (SEC 10-K, Chronicling America page, LoC manuscript, Gutenberg book, verify the Arabic Wikipedia endpoint). Build + human-review every `<id>.gt.md`. Target: ≥3 documents per category, ≥24 total for a defensible v1.
2. **Run every engine.** Get keys for Reducto, LlamaParse, Mistral OCR. Install Docling/Marker/MinerU. Run the full matrix. This is the expensive, credibility-defining step — do it properly once.
3. **Publish the real `LEADERBOARD.md`** with a dated `results/` file behind it. Every number regenerable.
4. **Write the methodology post** (draft in the repo, not yet public): why closed APIs matter, why reproducibility matters, the exact metrics, and the losses — including any category pdfmux loses. Leading with a pdfmux loss is the most persuasive thing on the page.

**Do not skip straight to promotion.** A benchmark caught with a wrong or un-reproducible number dies on day one and never recovers trust.

---

## Phase 1 — Distribution `[on-lift]`

Order matters: earn technical credibility first, then broaden.

1. **Invite competitors to submit adapters — privately, first.** Email/DM the maintainers of Docling, Marker, MinerU and the closed-API vendors: "We built a neutral, reproducible extraction benchmark that includes your tool. Here are your current numbers and the exact command to reproduce them. If you think we've mis-configured your engine, send a PR." This does three things: sources corrections (so the public numbers are unimpeachable), creates buy-in, and plants the "if you think your tool wins, send a PR" frame before launch. Vendors who engage become distributors.
2. **PR into the awesome-lists.** `awesome-ocr`, `awesome-document-understanding`, `awesome-pdf`, `awesome-llm-eval`. One-line entry, links to the leaderboard. These are durable, high-intent discovery surfaces.
3. **Submit to the tracker ecosystem.** Register the task/leaderboard on Papers-with-Code-style trackers and Hugging Face (a Space that renders `leaderboard.json`, or a dataset card for the corpus manifest + ground truth). Being in the tracker graph is how people find "the benchmark for X."
4. **Show HN** — "Show HN: A reproducible document-extraction benchmark that includes the closed APIs academic benchmarks skip." Lead with the gap (stale vendor benchmarks + academic benchmarks that omit commercial APIs), show the leaderboard with a pdfmux loss visible, link the one-command reproduction. HN rewards the honesty and the reproducibility; it punishes anything that smells like a vendor ad — so the post is about the *benchmark*, and pdfmux is just one row.
5. **Reddit / community**: r/MachineLearning (as a resource, not a plug), r/LocalLLaMA (the OSS-engine crowd cares about Docling/Marker/MinerU numbers), relevant Discords.

---

## Phase 2 — Become the citation `[on-lift]`

1. **Quarterly refresh, publicly.** The scheduled CI job regenerates the board; announce each refresh with a short "what changed" note (new engines, corpus additions, rank movements). Cadence is what separates a standard from a one-off.
2. **Make it embeddable.** A small badge (`pdfmux-bench: #N`) and a hosted HTML leaderboard that engines can link from their own READMEs. When a competitor links our board from their README, we've won — they're citing us as neutral ground.
3. **Publish rank-movement stories.** "MinerU jumped 3 spots on complex-tables after v2" is content that the engine's own community amplifies. The benchmark becomes news infrastructure for the whole category.
4. **Defend neutrality loudly.** When someone accuses the bench of pdfmux bias (they will), the answer is a link: here's the scoring code, here's the corpus, here's the command, here's the category pdfmux loses — rerun it yourself. Every such exchange, handled in public, compounds trust.

---

## What "won" looks like

- A third party arguing about extraction quality links **our** leaderboard, not Reducto's.
- A competitor's README says "ranked #N on pdfmux-bench."
- Academic papers cite the corpus/methodology as the commercial-inclusive complement to OmniDocBench.
- pdfmux's ranking on its own neutral board is the single most-cited proof point in its sales/marketing — precisely *because* the board also shows where pdfmux loses.

## Non-goals / guardrails

- **Never juice pdfmux's numbers.** The moment the board is caught favoring pdfmux, the asset is worth zero. The reproducibility is the receipt — protect it above all.
- **Don't gate the corpus.** Public, license-clean, downloadable. A benchmark you can't reproduce isn't one.
- **Don't let it go stale.** A dead benchmark is a liability that reminds everyone RD-TableBench died. Refresh or retire — never rot.
- **Respect the stay-dark gate.** No broad public launch of the bench that spotlights pdfmux before the founder lifts the freeze.
