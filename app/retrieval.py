"""BM25 baseline and optional dense/hybrid retrieval over the same source chunks."""

import math
import re
from collections import Counter
from pathlib import Path

from app.embeddings import Encoder
from app.schemas import Source
from app.vector_index import DenseIndex


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            tokens.append(run)
    return tokens


class DocumentIndex:
    def __init__(
        self,
        directory: Path,
        *,
        mode: str = "bm25",
        encoder: Encoder | None = None,
        cache_dir: Path | None = None,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("Retrieval constants must be positive")
        self.mode, self.candidate_k, self.rrf_k = mode, candidate_k, rrf_k
        self.dense = None
        self.chunks: list[Source] = []
        self.documents: list[dict] = []
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ")
            self.documents.append({"id": path.stem, "title": title, "content": text})
            # Overlapping character windows also bound the context supplied to the model.
            for number, start in enumerate(range(0, len(text), 420), 1):
                self.chunks.append(
                    Source(
                        id=f"{path.stem}:{number}",
                        document=path.stem,
                        title=title,
                        text=text[start : start + 480],
                        score=0,
                    )
                )
        self.frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.lengths = [sum(freq.values()) for freq in self.frequencies]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1) or 1
        self.document_frequency = Counter(token for freq in self.frequencies for token in freq)

        if encoder is not None:
            if cache_dir is None:
                raise ValueError("Vector cache directory is required")
            self.dense = DenseIndex(self.chunks, encoder, cache_dir)
        if mode != "bm25" and self.dense is None:
            raise ValueError("Dense and hybrid retrieval require an encoder")

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        mode: str | None = None,
        min_dense_score: float | None = None,
    ) -> list[Source]:
        selected = mode or self.mode
        if selected not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []
        if selected == "bm25":
            return self._bm25_search(query, top_k)
        if self.dense is None:
            raise ValueError("Dense retrieval is not configured")
        depth = top_k if selected == "dense" else max(self.candidate_k, top_k)
        dense_matches = self.dense.search(query, depth)
        if min_dense_score is not None and (
            not dense_matches or dense_matches[0].score < min_dense_score
        ):
            return []
        if selected == "dense":
            return dense_matches
        # Each leg fetches more than final top_k so cross-list agreement can survive.
        return reciprocal_rank_fusion(
            [self._bm25_search(query, depth), dense_matches],
            top_k=top_k,
            rank_constant=self.rrf_k,
        )

    def _bm25_search(self, query: str, top_k: int) -> list[Source]:
        ranked = []
        for index, freq in enumerate(self.frequencies):
            score = 0.0
            for token in set(tokenize(query)):
                tf = freq[token]
                if not tf:
                    continue
                df = self.document_frequency[token]
                idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                norm = tf + 1.5 * (0.25 + 0.75 * self.lengths[index] / self.average_length)
                score += idf * tf * 2.5 / norm
            if score > 0:
                ranked.append(self.chunks[index].model_copy(update={"score": round(score, 4)}))
        return sorted(ranked, key=lambda chunk: (-chunk.score, chunk.id))[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[Source]],
    *,
    top_k: int = 3,
    rank_constant: int = 60,
) -> list[Source]:
    """Fuse ranks rather than incompatible raw BM25 and cosine score scales."""
    if rank_constant < 1 or top_k < 1:
        raise ValueError("RRF constants must be positive")
    scores: dict[str, float] = {}
    sources: dict[str, Source] = {}
    for ranking in rankings:
        seen = set()
        for rank, source in enumerate(ranking, 1):
            if source.id in seen:
                continue
            seen.add(source.id)
            sources[source.id] = source
            scores[source.id] = scores.get(source.id, 0) + 1 / (rank_constant + rank)
    ids = sorted(scores, key=lambda source_id: (-scores[source_id], source_id))[:top_k]
    return [
        sources[source_id].model_copy(update={"score": round(scores[source_id], 8)})
        for source_id in ids
    ]
