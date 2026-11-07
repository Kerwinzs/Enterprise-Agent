"""Explicitly download the local Chinese model and build document embeddings."""

from app.config import PACKAGE_DIR, Settings
from app.embeddings import LocalChineseEncoder
from app.retrieval import DocumentIndex


def main():
    settings = Settings()
    encoder = LocalChineseEncoder(settings.data_dir / "models", allow_download=True)
    index = DocumentIndex(
        PACKAGE_DIR / "sample_data/docs",
        mode="hybrid",
        encoder=encoder,
        cache_dir=settings.data_dir / "vectors",
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k,
    )
    print(f"Prepared {len(index.chunks)} chunks; vector_cache_hit={index.dense.cache_hit}")
    print(f"Model: {encoder.MODEL}; artifact_sha256={encoder.artifact_sha256}")
    print("Set RETRIEVAL_MODE=hybrid in .env, then restart the server.")


if __name__ == "__main__":
    main()
