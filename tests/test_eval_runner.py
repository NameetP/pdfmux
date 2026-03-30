"""Tests for the benchmark runner."""

from __future__ import annotations

import json

from pdfmux.eval.runner import BenchmarkResult, BenchmarkRunner, DocumentBenchmark, ExtractorScore


class TestBenchmarkRunner:
    def test_discover_no_datasets(self, tmp_path):
        runner = BenchmarkRunner(dataset_dir=tmp_path)
        datasets = runner.discover_datasets()
        assert datasets == []

    def test_discover_with_ground_truth(self, tmp_path):
        # Create a mock dataset
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        (tables_dir / "test.pdf").write_bytes(b"%PDF-1.4 fake")
        (tables_dir / "test.gt.md").write_text("# Ground Truth\n\nSome text.")

        runner = BenchmarkRunner(dataset_dir=tmp_path)
        datasets = runner.discover_datasets()
        assert len(datasets) == 1
        assert datasets[0][2] == "tables"  # page_type from dir name

    def test_discover_skips_without_ground_truth(self, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        (tables_dir / "test.pdf").write_bytes(b"%PDF-1.4 fake")
        # No .gt.md file

        runner = BenchmarkRunner(dataset_dir=tmp_path)
        datasets = runner.discover_datasets()
        assert datasets == []

    def test_run_all_empty_dataset(self, tmp_path):
        runner = BenchmarkRunner(dataset_dir=tmp_path)
        results = runner.run_all()
        assert isinstance(results, BenchmarkResult)
        assert results.documents == []


class TestBenchmarkResult:
    def test_summary_by_type(self):
        result = BenchmarkResult(
            timestamp="2026-03-30T00:00:00Z",
            documents=[
                DocumentBenchmark(
                    pdf_path="test.pdf",
                    page_type="tables",
                    page_count=5,
                    scores=[
                        ExtractorScore(
                            extractor="pymupdf",
                            text_accuracy=0.80,
                            structure_preservation=0.70,
                            table_f1=0.60,
                            overall=0.70,
                            latency_ms=100,
                        ),
                        ExtractorScore(
                            extractor="docling",
                            text_accuracy=0.90,
                            structure_preservation=0.85,
                            table_f1=0.95,
                            overall=0.90,
                            latency_ms=2000,
                        ),
                    ],
                )
            ],
        )

        summary = result.summary_by_type()
        assert "tables" in summary
        assert "pymupdf" in summary["tables"]
        assert "docling" in summary["tables"]
        assert summary["tables"]["docling"]["overall"] == 0.90
        assert summary["tables"]["docling"]["table_f1"] == 0.95

    def test_summary_skips_errors(self):
        result = BenchmarkResult(
            timestamp="2026-03-30T00:00:00Z",
            documents=[
                DocumentBenchmark(
                    pdf_path="test.pdf",
                    page_type="scanned",
                    scores=[
                        ExtractorScore(extractor="llm", error="No API key"),
                        ExtractorScore(
                            extractor="pymupdf",
                            overall=0.50,
                            latency_ms=50,
                        ),
                    ],
                )
            ],
        )

        summary = result.summary_by_type()
        assert "llm" not in summary.get("scanned", {})
        assert "pymupdf" in summary.get("scanned", {})


class TestSaveResults:
    def test_save_and_load(self, tmp_path):
        result = BenchmarkResult(
            timestamp="2026-03-30T00:00:00Z",
            documents=[
                DocumentBenchmark(
                    pdf_path="test.pdf",
                    page_type="digital",
                    scores=[
                        ExtractorScore(
                            extractor="pymupdf",
                            overall=0.95,
                            latency_ms=10,
                        )
                    ],
                )
            ],
        )

        output_path = tmp_path / "results.json"
        runner = BenchmarkRunner(dataset_dir=tmp_path)
        saved = runner.save_results(result, output_path)

        assert saved.is_file()
        data = json.loads(saved.read_text())
        assert "summary" in data
        assert "documents" in data
        assert data["version"] == "1.0"


class TestExtractorScore:
    def test_creation(self):
        score = ExtractorScore(extractor="test", overall=0.85)
        assert score.extractor == "test"
        assert score.overall == 0.85
        assert score.error is None

    def test_error_score(self):
        score = ExtractorScore(extractor="llm", error="Not installed")
        assert score.error == "Not installed"
        assert score.overall == 0.0
