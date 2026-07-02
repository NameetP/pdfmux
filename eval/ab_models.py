"""Model A/B for pdfmux's LLM extraction path — cost x quality, one command.

Runs each named model over the 20-doc image-only ground-truth set
(eval/ab_datasets, built by build_ab_dataset.py), scoring extraction quality
against ground truth (pdfmux.eval.metrics) plus estimated cost and wall-clock
latency, then ranks by a cost x quality objective and names the winner.

Design rules:
  - NEVER fabricate numbers. A model whose SDK isn't installed or whose key is
    missing/dead is reported as BLOCKED with the exact command to unblock it —
    it gets no quality/cost row.
  - Real numbers only for models that actually ran this session.

Model specs (comma-separated in --models):
  none              baseline: quality=fast, native text only, no LLM, $0
  ocr               baseline: quality=standard, RapidOCR, no LLM, $0
  <provider>:<id>   quality=high, PDFMUX_LLM_PROVIDER=<provider>
                    PDFMUX_LLM_MODEL=<id>  (e.g. claude:claude-sonnet-5,
                    gemma:gemma-4-31b-it, openai:gpt-4o, glm:glm-5.2,
                    kimi:kimi-k2.7)

    python eval/ab_models.py --models "none,ocr,gemma:gemma-4-31b-it,claude:claude-sonnet-5,glm:glm-5.2,kimi:kimi-k2.7"

Writes eval/outputs/ab_report.md and eval/outputs/ab_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATASET_DIR = EVAL_DIR / "ab_datasets"
OUT_DIR = EVAL_DIR / "outputs"

# Tokens assumed per image page for cost estimation (input image + prompt, output text).
IMG_INPUT_TOKENS = 1200
OUT_TOKENS = 400
QUALITY_FLOOR = 0.60  # a model must clear this mean-overall to win on value


@dataclass
class DocScore:
    name: str
    overall: float = 0.0
    text_accuracy: float = 0.0
    structure: float = 0.0
    hallucination: float = 0.0
    error: str = ""


@dataclass
class ModelResult:
    spec: str
    status: str = "ran"  # ran | blocked
    reason: str = ""  # why blocked
    unblock_cmd: str = ""
    n_docs: int = 0
    mean_overall: float = 0.0
    mean_accuracy: float = 0.0
    mean_structure: float = 0.0
    mean_hallucination: float = 0.0
    est_cost_usd: float = 0.0
    est_cost_per_doc: float = 0.0
    seconds: float = 0.0
    sec_per_doc: float = 0.0
    errors: int = 0
    value_score: float = 0.0  # quality per dollar (see _rank)
    docs: list[DocScore] = field(default_factory=list)


def _load_dataset() -> list[tuple[str, Path, str]]:
    if not DATASET_DIR.is_dir():
        raise SystemExit(
            f"Dataset not found at {DATASET_DIR}. Run: python eval/build_ab_dataset.py"
        )
    items = []
    for pdf in sorted(DATASET_DIR.glob("*.pdf")):
        gt = pdf.with_suffix("").with_suffix(".gt.md")
        if gt.is_file():
            items.append((pdf.stem, pdf, gt.read_text(encoding="utf-8")))
    if not items:
        raise SystemExit(f"No <name>.pdf + <name>.gt.md pairs in {DATASET_DIR}.")
    return items


def _model_cost_per_page(provider, model_id: str) -> float | None:
    """Est. $ for one image page on this provider/model, from ModelInfo pricing."""
    try:
        for m in provider.supported_models():
            if m.id == model_id and m.input_cost_per_mtok is not None:
                out_cost = m.output_cost_per_mtok or 0.0
                return (
                    IMG_INPUT_TOKENS * m.input_cost_per_mtok + OUT_TOKENS * out_cost
                ) / 1_000_000
    except Exception:
        pass
    # Fall back to the provider's own estimator (uses its default model pricing).
    try:
        return provider.estimate_cost(0).cost_usd
    except Exception:
        return None


def _check_llm(provider_name: str, model_id: str) -> tuple[object | None, str, str]:
    """Return (provider_or_None, reason, unblock_cmd). provider is None if blocked."""
    from pdfmux.providers._discovery import discover_all_providers

    providers = discover_all_providers()
    p = providers.get(provider_name)
    install_hint = {
        "gemini": 'pip install "pdfmux[llm]"  # + export GEMINI_API_KEY',
        "gemma": 'pip install "pdfmux[llm]"  # + export GEMINI_API_KEY',
        "claude": 'pip install "pdfmux[llm-claude]"  # + export a LIVE ANTHROPIC_API_KEY',
        "openai": 'pip install "pdfmux[llm-openai]"  # + export OPENAI_API_KEY',
        "glm": "add glm to ~/.pdfmux/providers.yaml (see .pdfmux.example.yaml) + export ZHIPU_API_KEY",
        "kimi": "add kimi to ~/.pdfmux/providers.yaml (see .pdfmux.example.yaml) + export MOONSHOT_API_KEY",
    }.get(provider_name, f"configure provider '{provider_name}'")
    cmd = (
        f"{install_hint}  &&  PDFMUX_LLM_PROVIDER={provider_name} "
        f'PDFMUX_LLM_MODEL={model_id} python eval/ab_models.py --models "{provider_name}:{model_id}"'
    )
    if p is None:
        return None, f"provider '{provider_name}' not configured", cmd
    if not p.sdk_installed():
        return None, f"SDK not installed for '{provider_name}'", cmd
    if not p.has_credentials():
        return None, f"no/invalid credentials for '{provider_name}'", cmd
    return p, "", cmd


def _run_spec(spec: str, dataset: list[tuple[str, Path, str]]) -> ModelResult:
    from pdfmux.eval.metrics import hallucination_rate, structure_preservation, text_accuracy

    res = ModelResult(spec=spec)

    # Resolve quality + provider/model, and gate LLM specs on availability.
    provider = None
    model_id = ""
    if spec == "none":
        quality = "fast"
        os.environ.pop("PDFMUX_LLM_PROVIDER", None)
        os.environ.pop("PDFMUX_LLM_MODEL", None)
    elif spec == "ocr":
        quality = "standard"
        os.environ.pop("PDFMUX_LLM_PROVIDER", None)
        os.environ.pop("PDFMUX_LLM_MODEL", None)
    else:
        if ":" not in spec:
            res.status = "blocked"
            res.reason = f"bad spec {spec!r} (want provider:model)"
            return res
        provider_name, model_id = spec.split(":", 1)
        provider, reason, cmd = _check_llm(provider_name, model_id)
        if provider is None:
            res.status = "blocked"
            res.reason = reason
            res.unblock_cmd = cmd
            return res
        quality = "high"
        os.environ["PDFMUX_LLM_PROVIDER"] = provider_name
        os.environ["PDFMUX_LLM_MODEL"] = model_id

    # Deferred import so env vars above are seen by the pipeline.
    from pdfmux.pipeline import process

    t0 = time.time()
    ocr_pages_total = 0
    for name, pdf, gt in dataset:
        ds = DocScore(name=name)
        try:
            # use_cache=False is REQUIRED: the smart result cache keys on file
            # content, not on the model, so leaving it on makes model B silently
            # read model A's cached extraction (identical scores, ~0s runtime).
            out = process(
                file_path=str(pdf), output_format="markdown", quality=quality, use_cache=False
            )
            text = out.text or ""
            ocr_pages_total += len(out.ocr_pages or []) or (1 if quality != "fast" else 0)
            acc = text_accuracy(text, gt)
            struct = structure_preservation(text, gt)
            halluc = hallucination_rate(text, gt)
            overall = 0.5 * acc + 0.2 * struct + 0.3 * (1.0 - halluc)
            ds.text_accuracy, ds.structure, ds.hallucination = (
                round(acc, 4),
                round(struct, 4),
                round(halluc, 4),
            )
            ds.overall = round(overall, 4)
        except Exception as e:  # per-doc failure is data, not a crash
            ds.error = f"{type(e).__name__}: {str(e)[:160]}"
            res.errors += 1
        res.docs.append(ds)
        print(f"    [{spec:24}] {name:12} overall={ds.overall if not ds.error else 'ERR'}")

    res.seconds = round(time.time() - t0, 2)
    ok = [d for d in res.docs if not d.error]
    res.n_docs = len(ok)
    if ok:
        res.mean_overall = round(sum(d.overall for d in ok) / len(ok), 4)
        res.mean_accuracy = round(sum(d.text_accuracy for d in ok) / len(ok), 4)
        res.mean_structure = round(sum(d.structure for d in ok) / len(ok), 4)
        res.mean_hallucination = round(sum(d.hallucination for d in ok) / len(ok), 4)
    res.sec_per_doc = round(res.seconds / max(len(dataset), 1), 3)

    # Cost (estimated). Baselines are $0 (no model call).
    if provider is not None:
        per_page = _model_cost_per_page(provider, model_id)
        if per_page is not None:
            res.est_cost_usd = round(per_page * max(ocr_pages_total, len(dataset)), 6)
            res.est_cost_per_doc = round(res.est_cost_usd / max(len(dataset), 1), 6)
    return res


# Cost tolerance per doc. Cost at COST_REF halves a model's value vs. running
# it free; cheaper barely dents it. Keeps free and paid models on ONE comparable
# scale (a free, top-quality model wins outright — quality-per-dollar alone would
# blow up for any paid model and wrongly beat a better free one).
COST_REF = 0.01


def _rank(results: list[ModelResult]) -> None:
    """value = quality penalized by cost/doc, on one scale for free and paid alike."""
    for r in results:
        if r.status != "ran" or r.n_docs == 0:
            continue
        r.value_score = round(r.mean_overall / (1.0 + (r.est_cost_per_doc / COST_REF)), 4)


def _pick_winner(results: list[ModelResult]) -> ModelResult | None:
    ran = [r for r in results if r.status == "ran" and r.n_docs > 0]
    if not ran:
        return None
    # Prefer the best-value model that clears the quality floor; else best quality.
    eligible = [r for r in ran if r.mean_overall >= QUALITY_FLOOR]
    pool = eligible or ran
    key = (lambda r: r.value_score) if eligible else (lambda r: r.mean_overall)
    return max(pool, key=key)


def _watchdog_context() -> str | None:
    """Best-effort: latest agent-cost-watchdog weekly spend, for cost context."""
    for base in (
        os.environ.get("PDFMUX_WATCHDOG_DIR"),
        str(Path.home() / "Business Agents" / "shared" / "agent-cost"),
    ):
        if not base:
            continue
        d = Path(base)
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.jsonl"))
        if not files:
            continue
        total = 0.0
        rows = 0
        for line in files[-1].read_text(encoding="utf-8").splitlines():
            try:
                total += float(json.loads(line).get("cost_usd", 0) or 0)
                rows += 1
            except Exception:
                continue
        return f"{files[-1].stem}: ${total:.2f} across {rows} metered runs"
    return None


def _write_report(results: list[ModelResult], winner: ModelResult | None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ab_report.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )

    ran = [r for r in results if r.status == "ran" and r.n_docs > 0]
    blocked = [r for r in results if r.status == "blocked"]
    ran.sort(key=lambda r: (-(r.value_score or 0), -r.mean_overall))

    lines = ["# pdfmux model A/B — cost x quality", ""]
    lines.append(
        "20 image-only docs (`eval/ab_datasets`), scored vs ground truth. "
        "Quality = 0.5·accuracy + 0.2·structure + 0.3·(1−hallucination)."
    )
    wd = _watchdog_context()
    if wd:
        lines.append(f"\nagent-cost-watchdog latest week — {wd}")
    lines.append("")
    if winner:
        lines.append(
            f"**WINNER (ran this session): `{winner.spec}`** — "
            f"quality {winner.mean_overall:.3f}, est ${winner.est_cost_usd:.4f} "
            f"({winner.n_docs} docs), value {winner.value_score}."
        )
    lines.append("")
    lines.append("## Ran (real numbers)")
    lines.append("")
    lines.append(
        "| Model | Quality | Accuracy | Struct | Halluc | Est $/20 | Est $/doc | Value | Sec | Errors |"
    )
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in ran:
        lines.append(
            f"| `{r.spec}` | {r.mean_overall:.3f} | {r.mean_accuracy:.3f} | {r.mean_structure:.3f} | "
            f"{r.mean_hallucination:.3f} | ${r.est_cost_usd:.4f} | ${r.est_cost_per_doc:.5f} | "
            f"{r.value_score} | {r.seconds:.1f} | {r.errors} |"
        )
    if blocked:
        lines.append("")
        lines.append("## Blocked (no fabricated numbers — run the command to unblock)")
        lines.append("")
        for r in blocked:
            lines.append(f"- **`{r.spec}`** — {r.reason}")
            if r.unblock_cmd:
                lines.append(f"  ```\n  {r.unblock_cmd}\n  ```")
    lines.append("")
    lines.append(
        "_Costs are estimated from per-MTok pricing × assumed tokens/page; verify vendor "
        "pricing (especially GLM/Kimi) before treating as exact. Baselines run at $0._"
    )
    path = OUT_DIR / "ab_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="none,ocr,gemma:gemma-4-31b-it,openai:gpt-4o,claude:claude-sonnet-5,glm:glm-5.2,kimi:kimi-k2.7",
        help="comma-separated model specs",
    )
    args = parser.parse_args()
    specs = [s.strip() for s in args.models.split(",") if s.strip()]

    dataset = _load_dataset()
    print(f"Loaded {len(dataset)} docs. Running {len(specs)} model specs.\n")

    results = [_run_spec(s, dataset) for s in specs]
    _rank(results)
    winner = _pick_winner(results)
    report = _write_report(results, winner)

    print("\n" + "=" * 60)
    for r in results:
        if r.status == "ran" and r.n_docs > 0:
            print(
                f"RAN     {r.spec:24} q={r.mean_overall:.3f}  ${r.est_cost_usd:.4f}  "
                f"value={r.value_score}  {r.seconds:.1f}s  err={r.errors}"
            )
        else:
            print(f"BLOCKED {r.spec:24} {r.reason}")
    if winner:
        print(
            f"\nWINNER (ran): {winner.spec}  (quality {winner.mean_overall:.3f}, "
            f"est ${winner.est_cost_usd:.4f}, value {winner.value_score})"
        )
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
