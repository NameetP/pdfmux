"""Benchmark runner — test extractors against ground truth PDFs.

Usage:
    runner = BenchmarkRunner(dataset_dir="eval/datasets/")
    results = runner.run_all()
    runner.save_results(results)

Ground truth format:
    eval/datasets/
    ├── tables/
    │   ├── invoice.pdf
    │   └── invoice.gt.md      ← ground truth markdown
    ├── scanned/
    │   ├── letter.pdf
    │   └── letter.gt.md
    └── ...
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path

from pdfmux.eval.metrics import (
    hallucination_rate,
    structure_preservation,
    table_f1,
    text_accuracy,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractorScore:
    """Score for one extractor on one document."""

    extractor: str
    text_accuracy: float = 0.0
    structure_preservation: float = 0.0
    table_f1: float = 0.0
    hallucination_rate: float = 0.0
    overall: float = 0.0
    latency_ms: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class DocumentBenchmark:
    """Benchmark results for one document across all extractors."""

    pdf_path: str
    page_type: str  # directory name = page type
    page_count: int = 0
    scores: list[ExtractorScore] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    version: str = "1.0"
    timestamp: str = ""
    documents: list[DocumentBenchmark] = field(default_factory=list)

    def summary_by_type(self) -> dict[str, dict[str, dict[str, float]]]:
        """Aggregate scores by page_type → extractor → metric."""
        agg: dict[str, dict[str, list[ExtractorScore]]] = {}

        for doc in self.documents:
            if doc.page_type not in agg:
                agg[doc.page_type] = {}
            for score in doc.scores:
                if score.error:
                    continue
                if score.extractor not in agg[doc.page_type]:
                    agg[doc.page_type][score.extractor] = []
                agg[doc.page_type][score.extractor].append(score)

        result = {}
        for page_type, extractors in agg.items():
            result[page_type] = {}
            for ext_name, scores in extractors.items():
                n = len(scores)
                if n == 0:
                    continue
                result[page_type][ext_name] = {
                    "text_accuracy": sum(s.text_accuracy for s in scores) / n,
                    "structure_preservation": sum(s.structure_preservation for s in scores) / n,
                    "table_f1": sum(s.table_f1 for s in scores) / n,
                    "hallucination_rate": sum(s.hallucination_rate for s in scores) / n,
                    "overall": sum(s.overall for s in scores) / n,
                    "avg_latency_ms": sum(s.latency_ms for s in scores) / n,
                    "avg_cost_usd": sum(s.cost_usd for s in scores) / n,
                    "n_documents": n,
                }
        return result


class BenchmarkRunner:
    """Run benchmarks against ground truth datasets."""

    def __init__(self, dataset_dir: str | Path | None = None):
        if dataset_dir:
            self.dataset_dir = Path(dataset_dir)
        else:
            # Default: eval/datasets/ relative to repo root
            self.dataset_dir = Path(__file__).parent.parent.parent.parent / "eval" / "datasets"

    def discover_datasets(self) -> list[tuple[Path, Path, str]]:
        """Find all (pdf, ground_truth, page_type) tuples.

        Walks dataset_dir looking for .pdf files with matching .gt.md files.
        The parent directory name is the page_type.
        """
        datasets = []
        if not self.dataset_dir.is_dir():
            logger.warning("Dataset directory not found: %s", self.dataset_dir)
            return datasets

        for pdf_path in sorted(self.dataset_dir.rglob("*.pdf")):
            gt_path = pdf_path.with_suffix("").with_suffix(".gt.md")
            if not gt_path.is_file():
                # Try alternative naming
                gt_path = pdf_path.parent / (pdf_path.stem + ".gt.md")
            if not gt_path.is_file():
                logger.debug("No ground truth for %s — skipping", pdf_path.name)
                continue

            page_type = pdf_path.parent.name
            datasets.append((pdf_path, gt_path, page_type))

        return datasets

    def run_single(
        self,
        pdf_path: Path,
        ground_truth: str,
        page_type: str,
        extractors: list[str] | None = None,
    ) -> DocumentBenchmark:
        """Benchmark all available extractors on a single PDF."""
        from pdfmux.detect import classify

        classification = classify(pdf_path)

        # Default: test all available extractors
        if extractors is None:
            extractors = self._get_available_extractors()

        doc_result = DocumentBenchmark(
            pdf_path=str(pdf_path),
            page_type=page_type,
            page_count=classification.page_count,
        )

        for ext_name in extractors:
            score = self._benchmark_extractor(pdf_path, ground_truth, ext_name, page_type)
            doc_result.scores.append(score)

        return doc_result

    def run_all(
        self,
        extractors: list[str] | None = None,
    ) -> BenchmarkResult:
        """Run benchmarks on all datasets."""
        from datetime import datetime

        datasets = self.discover_datasets()
        if not datasets:
            logger.warning("No benchmark datasets found in %s", self.dataset_dir)
            return BenchmarkResult(timestamp=datetime.now(UTC).isoformat())

        result = BenchmarkResult(timestamp=datetime.now(UTC).isoformat())

        for pdf_path, gt_path, page_type in datasets:
            ground_truth = gt_path.read_text(encoding="utf-8")
            logger.info("Benchmarking %s (%s)", pdf_path.name, page_type)

            doc_benchmark = self.run_single(pdf_path, ground_truth, page_type, extractors)
            result.documents.append(doc_benchmark)

        return result

    def save_results(self, results: BenchmarkResult, output_path: Path | None = None) -> Path:
        """Save benchmark results to JSON.

        Default location: ~/.config/pdfmux/eval_results.json
        (This is where the router engine loads them from.)
        """
        if output_path is None:
            output_path = Path.home() / ".config" / "pdfmux" / "eval_results.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": results.version,
            "timestamp": results.timestamp,
            "summary": results.summary_by_type(),
            "documents": [asdict(d) for d in results.documents],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Benchmark results saved to %s", output_path)
        return output_path

    def _benchmark_extractor(
        self,
        pdf_path: Path,
        ground_truth: str,
        extractor_name: str,
        page_type: str,
    ) -> ExtractorScore:
        """Run one extractor on one PDF and compute metrics."""
        try:
            start = time.perf_counter()
            extracted_text = self._extract_with(pdf_path, extractor_name)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            # Compute metrics
            acc = text_accuracy(extracted_text, ground_truth)
            struct = structure_preservation(extracted_text, ground_truth)
            t_f1 = table_f1(extracted_text, ground_truth) if page_type == "tables" else 0.0
            halluc = hallucination_rate(extracted_text, ground_truth)

            # Overall score: weighted average
            # Tables get table_f1 weight, others don't
            if page_type == "tables":
                overall = 0.3 * acc + 0.2 * struct + 0.3 * t_f1 + 0.2 * (1.0 - halluc)
            else:
                overall = 0.4 * acc + 0.3 * struct + 0.3 * (1.0 - halluc)

            return ExtractorScore(
                extractor=extractor_name,
                text_accuracy=round(acc, 4),
                structure_preservation=round(struct, 4),
                table_f1=round(t_f1, 4),
                hallucination_rate=round(halluc, 4),
                overall=round(overall, 4),
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.warning("Extractor '%s' failed on %s: %s", extractor_name, pdf_path.name, e)
            return ExtractorScore(extractor=extractor_name, error=str(e))

    def _extract_with(self, pdf_path: Path, extractor_name: str) -> str:
        """Extract text from a PDF using a specific extractor."""
        if extractor_name == "pymupdf":
            from pdfmux.extractors.fast import FastExtractor

            ext = FastExtractor()
            pages = list(ext.extract(pdf_path))
            return "\n\n".join(p.text for p in pages)

        elif extractor_name == "multipass":
            from pdfmux.pipeline import process

            result = process(pdf_path, quality="standard")
            return result.text

        else:
            from pdfmux.extractors import get_extractor

            ext = get_extractor(extractor_name)
            pages = list(ext.extract(pdf_path))
            return "\n\n".join(p.text for p in pages)

    def _get_available_extractors(self) -> list[str]:
        """Get list of available extractor names."""
        extractors = ["pymupdf", "multipass"]  # always available

        try:
            from pdfmux.extractors import available_extractors

            for name, _ in available_extractors():
                if name not in ("fast",):  # fast is already pymupdf
                    extractors.append(name)
        except Exception:
            pass

        return extractors
