"""Head-to-head classifier benchmark: pdfmux detect.py vs pdf-inspector.

Runs both classifiers over the labeled corpus in manifest.csv, three timed
runs each, and writes results.json with per-doc predictions, per-run
latencies, confusion matrices on the shared 4-class axis, cost-weighted
error counts, page-level OCR-routing accuracy, and probe behavior.

Comparison axis (per PRD 2026-09-02-pdf-inspector-eval):
  shared 4-class doc axis: TextBased / Scanned / ImageBased / Mixed.
  pdfmux 7-class -> shared axis uses the production precedence from
  pipeline._classify_to_page_type (arabic > scanned > tables > graphical
  > mixed > digital) then maps arabic/tables/digital -> TextBased,
  scanned -> Scanned, graphical -> ImageBased, mixed -> Mixed.

Timing protocol:
  - classification only (pdfmux: detect.classify; pdf-inspector:
    classify_pdf — its documented lightweight classification entry point).
  - 3 runs x all docs x both tools, same process, same machine.
  - pdfmux's pdf_cache is cleared after every classify call so every
    measurement includes the document open (no cross-run cache leak).
  - run 1 is cold-ish (first touch of each file this process; OS page
    cache may hold fixtures), runs 2-3 are warm. Reported per-run.

pdfmux is imported straight from the read-only clone's src/ via a stub
package (skipping pdfmux/__init__.py, which pulls extraction-side deps
not needed to classify). detect.py itself and its imports (errors,
pdf_cache, arabic) execute unmodified.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent  # pdfmux clone root
MANIFEST = HERE / "manifest.csv"
RESULTS = HERE / "results.json"
N_RUNS = 3

# --- pdfmux import without package __init__ (read-only clone, no install) ---
_pkg = types.ModuleType("pdfmux")
_pkg.__path__ = [str(REPO / "src" / "pdfmux")]
sys.modules["pdfmux"] = _pkg
from pdfmux.detect import classify as pdfmux_classify  # noqa: E402
from pdfmux.pdf_cache import close_all as pdfmux_cache_clear  # noqa: E402

import pdf_inspector  # noqa: E402


def pdfmux_page_type(c) -> str:
    """Reimplements pipeline._classify_to_page_type (production precedence)."""
    if c.is_arabic:
        return "arabic"
    if c.is_scanned:
        return "scanned"
    if c.has_tables:
        return "tables"
    if c.is_graphical:
        return "graphical"
    if c.is_mixed:
        return "mixed"
    return "digital"


PDFMUX_TO_4CLASS = {
    "arabic": "TextBased",
    "tables": "TextBased",
    "digital": "TextBased",
    "scanned": "Scanned",
    "graphical": "ImageBased",
    "mixed": "Mixed",
}

PI_TO_4CLASS = {
    "text_based": "TextBased",
    "scanned": "Scanned",
    "image_based": "ImageBased",
    "mixed": "Mixed",
}


def run_pdfmux(path: str) -> dict:
    t0 = time.perf_counter()
    try:
        c = pdfmux_classify(path)
        dt = time.perf_counter() - t0
        pt = pdfmux_page_type(c)
        return dict(
            ok=True,
            latency_s=dt,
            raw_class=pt,
            pred=PDFMUX_TO_4CLASS[pt],
            confidence=c.confidence,
            page_count=c.page_count,
            ocr_pages=sorted(set(c.scanned_pages) | set(c.graphical_pages)),
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        return dict(ok=False, latency_s=dt, error=f"{type(e).__name__}: {e}"[:200])
    finally:
        try:
            pdfmux_cache_clear()
        except Exception:
            pass


def run_pi(path: str) -> dict:
    t0 = time.perf_counter()
    try:
        c = pdf_inspector.classify_pdf(path)
        dt = time.perf_counter() - t0
        return dict(
            ok=True,
            latency_s=dt,
            raw_class=c.pdf_type,
            pred=PI_TO_4CLASS.get(c.pdf_type, c.pdf_type),
            confidence=c.confidence,
            page_count=c.page_count,
            ocr_pages=sorted(c.pages_needing_ocr),
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        return dict(ok=False, latency_s=dt, error=f"{type(e).__name__}: {e}"[:200])


# Page-level OCR-needed ground truth, by construction (hybrid/scanned cells only).
def page_ocr_gt(category: str, page_count: int) -> list[int] | None:
    if category in ("gen_scanned", "image_only"):
        return list(range(page_count))
    if category == "gen_mixed":
        return [1, 3]
    if category == "gen_searchable_scan":
        return []  # complete invisible text layer present — text extractable
    return None


def summarize(latencies: list[float]) -> dict:
    s = sorted(latencies)
    p95 = s[max(0, int(round(0.95 * len(s))) - 1)]
    return dict(
        n=len(s),
        median_ms=round(statistics.median(s) * 1000, 2),
        p95_ms=round(p95 * 1000, 2),
        mean_ms=round(statistics.fmean(s) * 1000, 2),
        max_ms=round(max(s) * 1000, 2),
    )


def main() -> None:
    docs = list(csv.DictReader(open(MANIFEST, newline="")))
    for d in docs:
        d["abspath"] = str(REPO / d["file"])

    tools = {"pdfmux": run_pdfmux, "pdf_inspector": run_pi}
    per_doc: dict[str, dict] = {d["file"]: dict(d) for d in docs}
    latencies: dict[str, list[list[float]]] = {t: [[] for _ in range(N_RUNS)] for t in tools}

    for run in range(N_RUNS):
        for tool, fn in tools.items():
            for d in docs:
                r = fn(d["abspath"])
                latencies[tool][run].append(r["latency_s"])
                if run == 0:
                    per_doc[d["file"]][tool] = r
                else:
                    prev = per_doc[d["file"]][tool]
                    now_pred = r.get("pred", r.get("error"))
                    prev_pred = prev.get("pred", prev.get("error"))
                    if now_pred != prev_pred:
                        per_doc[d["file"]].setdefault("instability", []).append(
                            f"{tool} run{run}: {prev_pred} -> {now_pred}"
                        )

    # --- accuracy on classifiable tier ---
    classifiable = [d for d in docs if d["tier"] == "classifiable"]
    classes = ["TextBased", "Scanned", "ImageBased", "Mixed"]
    acc: dict[str, dict] = {}
    for tool in tools:
        conf: dict[str, Counter] = defaultdict(Counter)
        correct_by_class: Counter = Counter()
        total_by_class: Counter = Counter()
        silent = []  # gt Scanned/Mixed -> pred TextBased (silent bad extraction)
        wasted = []  # gt TextBased -> pred needs-OCR class (wasted OCR)
        errors = []
        for d in classifiable:
            r = per_doc[d["file"]][tool]
            gt = d["gt_4class"]
            pred = r.get("pred", "ERROR") if r.get("ok") else "ERROR"
            conf[gt][pred] += 1
            total_by_class[gt] += 1
            if pred == gt:
                correct_by_class[gt] += 1
            if gt in ("Scanned", "Mixed") and pred == "TextBased":
                silent.append(dict(file=d["file"], gt=gt, weight=3 if gt == "Scanned" else 2))
            if gt == "TextBased" and pred in ("Scanned", "ImageBased", "Mixed"):
                wasted.append(dict(file=d["file"], pred=pred))
            if pred == "ERROR":
                errors.append(dict(file=d["file"], error=r.get("error")))
        per_class_acc = {
            c: round(correct_by_class[c] / total_by_class[c], 4) if total_by_class[c] else None
            for c in classes
        }
        macro = round(
            statistics.fmean(v for v in per_class_acc.values() if v is not None), 4
        )
        micro = round(sum(correct_by_class.values()) / len(classifiable), 4)
        acc[tool] = dict(
            n=len(classifiable),
            per_class_accuracy=per_class_acc,
            macro_accuracy=macro,
            micro_accuracy=micro,
            confusion={g: dict(c) for g, c in conf.items()},
            silent_failures=silent,
            silent_failure_weighted=sum(x["weight"] for x in silent),
            wasted_ocr=wasted,
            errors_on_classifiable=errors,
        )

    # --- page-level OCR routing on hybrid/scanned cells ---
    page_ocr: dict[str, dict] = {}
    for tool in tools:
        tp = fp = fn_ = tn = 0
        details = []
        for d in classifiable:
            r = per_doc[d["file"]][tool]
            if not r.get("ok"):
                continue
            gt_pages = page_ocr_gt(d["category"], r["page_count"])
            if gt_pages is None:
                continue
            got = set(r["ocr_pages"])
            want = set(gt_pages)
            allp = set(range(r["page_count"]))
            tp += len(got & want)
            fp += len(got - want)
            fn_ += len(want - got)
            tn += len(allp - got - want)
            if got != want:
                details.append(dict(file=d["file"], want=sorted(want), got=sorted(got)))
        total_pages = tp + fp + fn_ + tn
        page_ocr[tool] = dict(
            pages=total_pages,
            accuracy=round((tp + tn) / total_pages, 4) if total_pages else None,
            missed_ocr_pages=fn_,
            unneeded_ocr_pages=fp,
            disagreements=details,
        )

    # --- probe behavior ---
    probes = []
    for d in docs:
        if d["tier"] != "probe":
            continue
        row = dict(file=d["file"], category=d["category"])
        for tool in tools:
            r = per_doc[d["file"]][tool]
            row[tool] = r.get("pred") if r.get("ok") else r.get("error")
        probes.append(row)

    # --- speed ---
    speed = {
        tool: {
            f"run{i+1}": summarize(latencies[tool][i]) for i in range(N_RUNS)
        }
        | {"pooled": summarize([x for run in latencies[tool] for x in run])}
        for tool in tools
    }

    env = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu=subprocess.run(
            ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
            capture_output=True, text=True,
        ).stdout.strip(),
        nproc=subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip(),
        mem=subprocess.run(
            ["sh", "-c", "free -h | awk '/^Mem/{print $2}'"], capture_output=True, text=True
        ).stdout.strip(),
        pdf_inspector_version=version_of("pdf_inspector"),
        pymupdf_version=version_of("pymupdf"),
        pdfmux_git_sha=subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        pdfmux_version=subprocess.run(
            ["sh", "-c", f"grep -m1 '^version' {REPO}/pyproject.toml"],
            capture_output=True, text=True,
        ).stdout.strip(),
    )

    out = dict(
        env=env,
        n_docs=len(docs),
        n_classifiable=len(classifiable),
        gt_composition=dict(Counter(d["gt_4class"] for d in classifiable)),
        accuracy=acc,
        page_ocr_routing=page_ocr,
        speed=speed,
        probes=probes,
        per_doc={
            k: {
                t: {kk: vv for kk, vv in v[t].items() if kk != "latency_s"}
                for t in tools if t in v
            }
            | dict(gt=v["gt_4class"], tier=v["tier"], instability=v.get("instability", []))
            for k, v in per_doc.items()
        },
    )
    RESULTS.write_text(json.dumps(out, indent=1))

    print(json.dumps(dict(env=env, gt=out["gt_composition"]), indent=1))
    for tool in tools:
        a = acc[tool]
        print(
            f"\n{tool}: macro={a['macro_accuracy']} micro={a['micro_accuracy']} "
            f"per-class={a['per_class_accuracy']} "
            f"silent={len(a['silent_failures'])} (weighted {a['silent_failure_weighted']}) "
            f"wasted_ocr={len(a['wasted_ocr'])} errors={len(a['errors_on_classifiable'])}"
        )
        print(f"  speed: {json.dumps(speed[tool])}")
        print(f"  page_ocr: {json.dumps({k: v for k, v in page_ocr[tool].items() if k != 'disagreements'})}")


def version_of(mod: str) -> str:
    import importlib.metadata

    for name in (mod, mod.replace("_", "-")):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "?"


if __name__ == "__main__":
    main()
