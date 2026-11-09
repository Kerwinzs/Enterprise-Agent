"""Calibrate a cosine evidence gate on development cases and evaluate the frozen test split."""

import hashlib
import json
from pathlib import Path

from app.config import PACKAGE_DIR, Settings
from app.embeddings import LocalChineseEncoder
from app.retrieval import DocumentIndex
from scripts.evaluate_retrieval import DEFAULT_CASES, ROOT, load_cases, relevance_metrics

CALIBRATION_CASES = ROOT / "benchmarks/evidence_gate_calibration.json"
REPORT = ROOT / "benchmarks/evidence_gate_results.json"


def summary(rows: list[dict], result_key: str) -> dict:
    answerable = [row for row in rows if row["relevant_ids"]]
    unanswerable = [row for row in rows if not row["relevant_ids"]]
    return {
        "answerable": len(answerable),
        "answerable_blocked": sum(not row[result_key] for row in answerable),
        "recall_at_3": sum(
            relevance_metrics(row[result_key], row["relevant_ids"])["recall_at_3"]
            for row in answerable
        )
        / len(answerable),
        "unanswerable": len(unanswerable),
        "unanswerable_with_candidates": sum(bool(row[result_key]) for row in unanswerable),
    }


def main() -> None:
    settings = Settings()
    encoder = LocalChineseEncoder(settings.data_dir / "models")
    index = DocumentIndex(
        PACKAGE_DIR / "sample_data/docs",
        encoder=encoder,
        cache_dir=settings.data_dir / "vectors",
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k,
    )
    calibration = load_cases(DEFAULT_CASES, index, "dev") + load_cases(
        CALIBRATION_CASES, index, "dev"
    )
    for row in calibration:
        row["best_cosine"] = index.search(row["query"], top_k=1, mode="dense")[0].score
    lowest_answerable = min(row["best_cosine"] for row in calibration if row["relevant_ids"])
    threshold = max(step / 100 for step in range(-100, 101) if step / 100 <= lowest_answerable)

    test = load_cases(DEFAULT_CASES, index, "test")
    if {row["query"] for row in calibration} & {row["query"] for row in test}:
        raise ValueError("Calibration and test questions must be disjoint")
    results = {}
    for mode in ("dense", "hybrid"):
        rows = []
        for case in test:
            baseline = index.search(case["query"], mode=mode)
            gated = index.search(case["query"], mode=mode, min_dense_score=threshold)
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "relevant_ids": case["relevant_ids"],
                    "baseline": [source.id for source in baseline],
                    "gated": [source.id for source in gated],
                }
            )
        results[mode] = {
            "baseline": summary(rows, "baseline"),
            "gated": summary(rows, "gated"),
            "cases": rows,
        }
    report = {
        "encoder": encoder.identity,
        "index_fingerprint": index.dense.fingerprint,
        "candidate_k": index.candidate_k,
        "rrf_k": index.rrf_k,
        "dataset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (DEFAULT_CASES, CALIBRATION_CASES)
        },
        "threshold_rule": "Highest hundredth preserving every answerable calibration question",
        "threshold": threshold,
        "calibration": calibration,
        "test": results,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Calibrated threshold: {threshold:.2f}")
    for mode, result in results.items():
        print(mode, json.dumps({key: result[key] for key in ("baseline", "gated")}))
    print(f"Report: {Path(REPORT).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
