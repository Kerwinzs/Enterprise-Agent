"""Exact cosine search and content-addressed document-vector caching for a small corpus."""

import hashlib
import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from app.embeddings import Encoder
from app.schemas import Source

DEFAULT_QUERY_CACHE_SIZE = 128


def normalize(vector: list[float], dimension: int) -> list[float]:
    if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
        raise ValueError("Invalid embedding dimension or non-finite value")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        raise ValueError("Embedding must be nonzero")
    return [value / norm for value in vector]


class DenseIndex:
    def __init__(
        self,
        chunks: list[Source],
        encoder: Encoder,
        cache_dir: Path,
        *,
        query_cache_size: int = DEFAULT_QUERY_CACHE_SIZE,
    ):
        self.chunks = chunks

        # Cache only query vectors so ranking and evidence checks run for every search.
        @lru_cache(maxsize=query_cache_size)
        def query_vector(query: str) -> list[float]:
            return normalize(encoder.embed_query(query), encoder.dimension)

        self._query_vector = query_vector
        payload = {
            "schema_version": 1,
            "encoder": encoder.identity,
            "dimension": encoder.dimension,
            "chunks": [{"id": chunk.id, "text": chunk.text} for chunk in chunks],
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"{self.fingerprint}.json"
        self.cache_hit = False
        try:
            saved = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if saved["fingerprint"] != self.fingerprint or len(saved["vectors"]) != len(chunks):
                raise ValueError("Cache does not match corpus")
            self.vectors = [normalize(row, encoder.dimension) for row in saved["vectors"]]
            self.cache_hit = True
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            rows = encoder.embed_documents([chunk.text for chunk in chunks]) if chunks else []
            if len(rows) != len(chunks):
                raise ValueError("Encoder returned wrong number of document embeddings") from None
            self.vectors = [normalize(row, encoder.dimension) for row in rows]
            self._save()

    def _save(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.cache_path.parent,
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {"fingerprint": self.fingerprint, "vectors": self.vectors},
                    stream,
                    allow_nan=False,
                )
            os.replace(temporary, self.cache_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def search(self, query: str, top_k: int) -> list[Source]:
        if not self.chunks:
            return []
        query_vector = self._query_vector(query)
        ranked = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=True):
            similarity = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            ranked.append(chunk.model_copy(update={"score": round(similarity, 8)}))
        # Nearest neighbors are candidates, not proof that the corpus contains an answer.
        return sorted(ranked, key=lambda chunk: (-chunk.score, chunk.id))[:top_k]
