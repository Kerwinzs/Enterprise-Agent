"""Optional local CPU encoder. No model downloads happen on the serving path."""

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Protocol


class Encoder(Protocol):
    identity: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalChineseEncoder:
    MODEL = "BAAI/bge-small-zh-v1.5"
    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, cache_dir: Path, *, allow_download: bool = False):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                '向量检索需要可选依赖，请先运行 python -m pip install -e ".[retrieval]"'
            ) from exc
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.model = TextEmbedding(
                model_name=self.MODEL,
                cache_dir=str(cache_dir),
                threads=2,
                providers=["CPUExecutionProvider"],
                local_files_only=not allow_download,
            )
        except Exception as exc:
            raise RuntimeError(
                "中文向量模型未就绪，请联网运行 python -m scripts.prepare_retrieval；"
                "或显式设置 RETRIEVAL_MODE=bm25。服务不会自动降级。"
            ) from exc
        # Include actual local model/tokenizer artifacts, not just a mutable model name.
        digest = hashlib.sha256()
        for path in sorted(cache_dir.rglob("*")):
            if path.is_file() and path.suffix in {".onnx", ".onnx_data", ".json", ".txt"}:
                if ".cache" in path.parts:
                    continue
                digest.update(path.relative_to(cache_dir).as_posix().encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        self.artifact_sha256 = digest.hexdigest()
        self.dimension = 512
        self.identity = (
            f"{self.MODEL}|fastembed={version('fastembed')}|"
            f"artifacts={self.artifact_sha256}|query_prefix={self.QUERY_PREFIX}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in row] for row in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # This model's short-query instruction is applied only to the query.
        row = next(iter(self.model.embed([self.QUERY_PREFIX + text])))
        return [float(value) for value in row]
