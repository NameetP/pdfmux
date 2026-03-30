"""Eval harness — benchmark extractors against ground truth.

Usage:
    from pdfmux.eval import BenchmarkRunner, BenchmarkResult

    runner = BenchmarkRunner()
    results = runner.run_all()
"""

from pdfmux.eval.metrics import (
    hallucination_rate,
    structure_preservation,
    table_f1,
    text_accuracy,
)
from pdfmux.eval.runner import BenchmarkResult, BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "text_accuracy",
    "structure_preservation",
    "table_f1",
    "hallucination_rate",
]
