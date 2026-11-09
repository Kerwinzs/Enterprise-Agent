import json
import math

import pytest
from fastapi.testclient import TestClient

from app.config import PACKAGE_DIR, Settings
from app.retrieval import DocumentIndex, reciprocal_rank_fusion
from app.schemas import Source
from app.vector_index import DenseIndex, normalize
from scripts.evaluate_retrieval import DEFAULT_CASES, load_cases, relevance_metrics, summarize


class FakeEncoder:
    """Tiny deterministic semantic stand-in for unit tests, never used in benchmark reports."""

    identity = "test-encoder-v1"
    dimension = 3
    MODEL = "test-only"
    document_calls = 0
    query_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts):
        self.document_calls += 1
        return [
            [1, 0, 0]
            if text.startswith("# 差旅与报销")
            else [0, 1, 0]
            if "餐饮" in text
            else [0, 0, 1]
            for text in texts
        ]

    def embed_query(self, text):
        self.query_calls += 1
        return [1, 0, 0]


def source(source_id, text="text", score=1):
    return Source(
        id=source_id, document=source_id.split(":")[0], title=source_id, text=text, score=score
    )


@pytest.fixture
def small_corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hotel.md").write_text("# 差旅与报销\n一线城市住宿每晚上限600元。")
    (docs / "food.md").write_text("# 餐饮\n餐补每天100元。")
    return docs


def test_semantic_path_recovers_no_keyword_overlap(small_corpus, tmp_path):
    encoder = FakeEncoder()
    index = DocumentIndex(small_corpus, encoder=encoder, cache_dir=tmp_path / "vectors")
    assert index.search("住酒店") == []
    for mode in ("dense", "hybrid"):
        result = index.search("住酒店", top_k=1, mode=mode)
        assert result[0].id == "hotel:1"
    assert encoder.query_calls == 2


def test_rrf_uses_ranks_deduplicates_and_ignores_raw_score_scale():
    a, b, c = source("a:1", score=9999), source("b:1", score=0.001), source("c:1")
    result = reciprocal_rank_fusion([[a, b], [b, c]], top_k=3, rank_constant=60)
    assert [row.id for row in result] == ["b:1", "a:1", "c:1"]
    assert result[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert a.score == 9999
    duplicate = reciprocal_rank_fusion([[a, a], [b]], top_k=2)
    assert duplicate[0].score == duplicate[1].score
    assert reciprocal_rank_fusion([[], []]) == []


def test_cached_document_vectors_reused_and_invalidated(small_corpus, tmp_path):
    encoder = FakeEncoder()
    kwargs = {"encoder": encoder, "cache_dir": tmp_path / "vectors"}
    first = DocumentIndex(small_corpus, **kwargs)
    assert not first.dense.cache_hit
    again = DocumentIndex(small_corpus, **kwargs)
    assert again.dense.cache_hit
    assert encoder.document_calls == 1
    (small_corpus / "hotel.md").write_text("# 差旅与报销\n额度修改为700元。")
    changed = DocumentIndex(small_corpus, **kwargs)
    assert changed.dense.fingerprint != first.dense.fingerprint
    assert encoder.document_calls == 2
    encoder.identity = "different-model-or-tokenizer"
    changed_model = DocumentIndex(small_corpus, **kwargs)
    assert changed_model.dense.fingerprint != changed.dense.fingerprint
    assert encoder.document_calls == 3


@pytest.mark.parametrize(
    "bad_content",
    [
        "invalid json",
        '{"fingerprint":"wrong","vectors":[]}',
    ],
)
def test_corrupt_cache_is_rebuilt(small_corpus, tmp_path, bad_content):
    encoder = FakeEncoder()
    first = DocumentIndex(small_corpus, encoder=encoder, cache_dir=tmp_path / "vectors")
    first.dense.cache_path.write_text(bad_content)
    rebuilt = DocumentIndex(small_corpus, encoder=encoder, cache_dir=tmp_path / "vectors")
    assert not rebuilt.dense.cache_hit
    assert encoder.document_calls == 2


def test_wrong_dimensions_in_cache_are_rebuilt(tmp_path):
    encoder = FakeEncoder()
    index = DenseIndex([source("a:1")], encoder, tmp_path)
    index.cache_path.write_text(json.dumps({"fingerprint": index.fingerprint, "vectors": [[1, 2]]}))
    assert not DenseIndex([source("a:1")], encoder, tmp_path).cache_hit


@pytest.mark.parametrize("vector", [[0, 0], [1], [math.nan, 1], [math.inf, 0]])
def test_invalid_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        normalize(vector, 2)


def test_empty_queries_and_empty_corpus_do_not_embed(small_corpus, tmp_path):
    encoder = FakeEncoder()
    index = DocumentIndex(small_corpus, encoder=encoder, cache_dir=tmp_path / "vectors")
    assert index.search("  ", mode="hybrid") == []
    empty = DenseIndex([], encoder, tmp_path / "empty")
    assert empty.search("something", 3) == []
    assert encoder.query_calls == 0


def test_modes_require_explicit_encoder_and_valid_parameters(small_corpus):
    with pytest.raises(ValueError, match="require an encoder"):
        DocumentIndex(small_corpus, mode="hybrid")
    with pytest.raises(ValueError, match="Unknown"):
        DocumentIndex(small_corpus, mode="typo")
    with pytest.raises(ValueError):
        DocumentIndex(small_corpus, candidate_k=0)
    index = DocumentIndex(small_corpus)
    with pytest.raises(ValueError):
        index.search("q", mode="dense")
    with pytest.raises(ValueError):
        index.search("q", top_k=0)


def test_benchmark_labels_and_split():
    index = DocumentIndex(PACKAGE_DIR / "sample_data/docs")
    assert len(index.chunks) == 33
    test = load_cases(DEFAULT_CASES, index, "test")
    dev = load_cases(DEFAULT_CASES, index, "dev")
    assert len(test) == 50 and len(dev) == 10
    assert not {row["id"] for row in test} & {row["id"] for row in dev}
    assert len([row for row in test if row["category"] == "exact"]) == 20
    assert len([row for row in test if row["category"] == "paraphrase"]) == 20
    assert len([row for row in test if row["category"] == "no_answer"]) == 10


def test_bad_benchmark_label_fails_instead_of_changing_denominator(tmp_path):
    payload = json.loads(DEFAULT_CASES.read_text())
    payload["cases"][0]["relevant_ids"] = ["missing:1"]
    cases_path = tmp_path / "bad.json"
    cases_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid relevant"):
        load_cases(cases_path, DocumentIndex(PACKAGE_DIR / "sample_data/docs"), "test")


def test_metric_math_and_no_answer_report_separation():
    metrics = relevance_metrics(["bad:1", "a:1", "b:1"], ["a:1", "b:1", "c:1"])
    assert metrics == {"hit_at_1": 0.0, "recall_at_3": 2 / 3, "mrr_at_3": 0.5}
    assert relevance_metrics(["unrelated:1"], []) == {}
    rows = [
        {
            "category": "exact",
            "relevant_ids": ["a:1"],
            "ids": ["a:1"],
            "metrics": relevance_metrics(["a:1"], ["a:1"]),
            "latencies_ms": [1, 2],
        },
        {
            "category": "no_answer",
            "relevant_ids": [],
            "ids": ["unrelated:1"],
            "metrics": {},
            "latencies_ms": [3, 4],
        },
    ]
    summary = summarize(rows)
    assert summary["answerable"]["count"] == 1
    assert summary["answerable"]["recall_at_3"] == 1
    assert summary["no_answer"]["returned_candidates_rate"] == 1
    assert "recall_at_3" not in summary["no_answer"]


def test_hybrid_is_used_by_actual_chat_tool(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "LocalChineseEncoder", FakeEncoder)
    settings = Settings(
        _env_file=None,
        agent_mode="demo",
        retrieval_mode="hybrid",
        data_dir=tmp_path,
        openai_api_key="",
    )
    with TestClient(main_module.create_app(settings)) as client:
        config = client.get("/api/config").json()
        assert config["retrieval_mode"] == "hybrid"
        assert config["embedding_model"] == "test-only"
        result = client.post("/api/chat", json={"message": "住店花销"}).json()
        call = result["tool_calls"][0]
        assert call["name"] == "search_documents"
        assert call["result"]["retrieval_mode"] == "hybrid"
        assert call["result"]["score_kind"] == "rrf"
        assert result["sources"][0]["document"] == "reimbursement"


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_chat_evidence_gate_blocks_weak_matches_and_accepts_boundary(tmp_path, monkeypatch, mode):
    import app.main as main_module

    class GateEncoder(FakeEncoder):
        def embed_query(self, text):
            if text == "访客证补办费多少？":
                return [-1, -1, -1]
            return super().embed_query(text)

    monkeypatch.setattr(main_module, "LocalChineseEncoder", GateEncoder)
    settings = Settings(
        _env_file=None,
        agent_mode="demo",
        retrieval_mode=mode,
        retrieval_min_dense_score=1.0,
        data_dir=tmp_path,
        openai_api_key="",
    )
    with TestClient(main_module.create_app(settings)) as client:
        weak = client.post("/api/chat", json={"message": "访客证补办费多少？"}).json()
        assert weak["tool_calls"][0]["result"]["min_dense_score"] == 1.0
        assert weak["tool_calls"][0]["result"]["matches"] == []
        assert weak["sources"] == []
        assert "未找到足以支持回答的资料" in weak["answer"]

        strong = client.post("/api/chat", json={"message": "住店花销"}).json()
        assert strong["sources"][0]["id"] == "reimbursement:1"


def test_local_encoder_dependency_and_download_boundary(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from app.embeddings import LocalChineseEncoder

    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(RuntimeError, match="可选依赖"):
        LocalChineseEncoder(tmp_path)

    constructor_options = []

    class StubModel:
        def __init__(self, **kwargs):
            constructor_options.append(kwargs)

        def passage_embed(self, texts):
            return [[1] * 512 for _ in texts]

        def embed(self, texts):
            assert texts == [LocalChineseEncoder.QUERY_PREFIX + "酒店"]
            return iter([[1] * 512])

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=StubModel))
    # No optional library needs to be installed in the base CI job.
    monkeypatch.setattr("app.embeddings.version", lambda name: "test-version")
    encoder = LocalChineseEncoder(tmp_path)
    assert constructor_options[-1]["local_files_only"] is True
    assert len(encoder.embed_documents(["住宿"])) == 1
    assert len(encoder.embed_query("酒店")) == 512
    LocalChineseEncoder(tmp_path, allow_download=True)
    assert constructor_options[-1]["local_files_only"] is False


def test_unavailable_model_has_actionable_error_and_no_silent_fallback(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from app.embeddings import LocalChineseEncoder

    def unavailable(**kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("private library details")

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=unavailable))
    with pytest.raises(RuntimeError, match="scripts.prepare_retrieval") as error:
        LocalChineseEncoder(tmp_path)
    assert "private library" not in str(error.value)
