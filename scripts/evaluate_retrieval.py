"""Reproducible retrieval-only benchmark. No LLM calls, downloads or fabricated metrics."""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from app.config import PACKAGE_DIR, Settings
from app.embeddings import LocalChineseEncoder
from app.retrieval import DocumentIndex

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "benchmarks/retrieval_cases.json"


def load_cases(path: Path, index: DocumentIndex, split: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = {chunk.id: chunk.text for chunk in index.chunks}
    cases = data["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate benchmark case id")
    if len({case["query"] for case in cases}) != len(cases):
        raise ValueError("Duplicate benchmark query")
    for case in cases:
        if case["split"] not in {"dev", "test"}:
            raise ValueError("Unknown split")
        if case["category"] not in {"exact", "paraphrase", "no_answer"}:
            raise ValueError("Unknown category")
        labels = case["relevant_ids"]
        if (case["category"] == "no_answer") != (not labels):
            raise ValueError("No-answer labels must be empty; answerable labels must be nonempty")
        if len(labels) != len(set(labels)) or any(label not in chunks for label in labels):
            raise ValueError("Invalid relevant chunk ids")
        if len(case["evidence_quotes"]) != len(labels):
            raise ValueError("Every relevant chunk needs an evidence quote")
        for label, quote in zip(labels, case["evidence_quotes"], strict=True):
            if not quote or quote not in chunks[label]:
                raise ValueError(f"Evidence missing from {label} for {case['id']}")
    selected = [case for case in cases if split == "all" or case["split"] == split]
    if not selected:
        raise ValueError("No cases for requested split")
    return selected


def relevance_metrics(ids: list[str], relevant: list[str]) -> dict:
    if not relevant:
        return {}
    expected = set(relevant)
    first = next((rank for rank, source_id in enumerate(ids[:3], 1) if source_id in expected), None)
    return {
        "hit_at_1": float(bool(ids) and ids[0] in expected),
        "recall_at_3": len(set(ids[:3]) & expected) / len(expected),
        "mrr_at_3": 1 / first if first else 0.0,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(rows: list[dict]) -> dict:
    output = {}
    groups = {
        "answerable": [row for row in rows if row["relevant_ids"]],
        "exact": [row for row in rows if row["category"] == "exact"],
        "paraphrase": [row for row in rows if row["category"] == "paraphrase"],
        "no_answer": [row for row in rows if not row["relevant_ids"]],
    }
    for name, group in groups.items():
        if not group:
            continue
        durations = [value for row in group for value in row["latencies_ms"]]
        summary = {
            "count": len(group),
            "p50_ms": round(statistics.median(durations), 3),
            "p95_ms": round(percentile(durations, 0.95), 3),
        }
        if name == "no_answer":
            summary["returned_candidates_rate"] = sum(bool(row["ids"]) for row in group) / len(
                group
            )
        else:
            for metric in ("hit_at_1", "recall_at_3", "mrr_at_3"):
                summary[metric] = sum(row["metrics"][metric] for row in group) / len(group)
        output[name] = summary
    return output


def evaluate(index: DocumentIndex, cases: list[dict], modes: list[str], repeats: int) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    results = {mode: {} for mode in modes}
    # Warm model/runtime separately; all measured queries still include query embedding time.
    for mode in modes:
        index.search("检索预热", mode=mode)
    rng = random.Random(42)
    for _ in range(repeats):
        work = [(mode, case) for mode in modes for case in cases]
        rng.shuffle(work)
        for mode, case in work:
            started = time.perf_counter()
            matches = index.search(case["query"], top_k=3, mode=mode)
            elapsed_ms = (time.perf_counter() - started) * 1000
            ids = [match.id for match in matches]
            existing = results[mode].get(case["id"])
            if existing is None:
                results[mode][case["id"]] = {
                    **case,
                    "ids": ids,
                    "scores": [match.score for match in matches],
                    "metrics": relevance_metrics(ids, case["relevant_ids"]),
                    "latencies_ms": [],
                }
            elif existing["ids"] != ids:
                raise ValueError("Non-deterministic ranking across repetitions")
            results[mode][case["id"]]["latencies_ms"].append(round(elapsed_ms, 4))
    return {
        mode: {
            "summary": summarize(list(rows.values())),
            "cases": [rows[case["id"]] for case in cases],
        }
        for mode, rows in results.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/retrieval_results.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["bm25", "dense", "hybrid"],
        default=["bm25", "dense", "hybrid"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--query-cache-size",
        type=int,
        default=0,
        help="Capacity for preloaded evaluation queries. Zero measures uncached queries.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.query_cache_size < 0:
        parser.error("--query-cache-size must be nonnegative")
    settings = Settings()
    started = time.perf_counter()
    encoder = (
        LocalChineseEncoder(settings.data_dir / "models")
        if any(mode != "bm25" for mode in args.modes)
        else None
    )
    index = DocumentIndex(
        PACKAGE_DIR / "sample_data/docs",
        encoder=encoder,
        cache_dir=settings.data_dir / "vectors",
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k,
        query_cache_size=args.query_cache_size,
    )
    setup_ms = (time.perf_counter() - started) * 1000
    cases = load_cases(args.cases, index, args.split)
    if encoder and args.query_cache_size:
        for case in cases:
            index.search(case["query"], top_k=1, mode="dense")
    source_digest = hashlib.sha256()
    for path in sorted(
        [
            *(PACKAGE_DIR / "sample_data/docs").glob("*.md"),
            ROOT / "app/retrieval.py",
            ROOT / "app/embeddings.py",
            ROOT / "app/vector_index.py",
            Path(__file__),
        ]
    ):
        source_digest.update(path.name.encode() + path.read_bytes())
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "metadata": {
            "split": args.split,
            "cases": len(cases),
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "repeats": args.repeats,
            "candidate_k": settings.retrieval_candidate_k,
            "rrf_k": settings.retrieval_rrf_k,
            "query_cache_size": args.query_cache_size,
            "dataset_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "source_and_corpus_sha256": source_digest.hexdigest(),
            "model": encoder.MODEL if encoder else None,
            "model_identity": encoder.identity if encoder else None,
            "fastembed_version": version("fastembed") if encoder else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "setup_ms": round(setup_ms, 3),
            "document_vector_cache_hit": index.dense.cache_hit if index.dense else None,
            "latency_scope": (
                "Sequential retrieval including query cache lookup or encoding, cosine scan "
                "and fusion. Excludes download, index construction, LLM and HTTP. "
                "Evaluation queries are preloaded when query caching is enabled."
            ),
            "limitations": (
                "Synthetic manually authored corpus/labels, no independent review; single-machine "
                "small-sample results, not production or end-to-end answer accuracy. "
                "No-answer candidate rate is diagnostic, not hallucination rate."
            ),
        },
        "results": evaluate(index, cases, list(dict.fromkeys(args.modes)), args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for mode, result in report["results"].items():
        print(mode, json.dumps(result["summary"], ensure_ascii=False))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
