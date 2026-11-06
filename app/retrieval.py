"""Small in-memory BM25 index with Chinese bigrams; no embedding API required."""

import math
import re
from collections import Counter
from pathlib import Path

from app.schemas import Source


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            tokens.append(run)
    return tokens


class DocumentIndex:
    def __init__(self, directory: Path):
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

    def search(self, query: str, top_k: int = 3) -> list[Source]:
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
