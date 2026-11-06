import json
import sqlite3

import pytest

from app.config import PACKAGE_DIR
from app.database import OrderStore, initialize_database
from app.retrieval import DocumentIndex
from app.tools import ToolRegistry, tool_definitions


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "orders.db"
    initialize_database(path)
    initialize_database(path)  # Initialization is idempotent, never duplicates seed orders.
    return ToolRegistry(DocumentIndex(PACKAGE_DIR / "sample_data/docs"), OrderStore(path))


def test_retrieval_ranks_reimbursement_first_and_handles_no_match(registry):
    hits = registry.index.search("一线城市出差住宿报销标准", 2)
    assert hits[0].document == "reimbursement"
    assert "600" in hits[0].text
    assert hits[0].id == "reimbursement:1"
    assert registry.index.search("quantumunicornxyz") == []


def test_unpaid_includes_overdue_and_summary_is_not_limited(registry):
    _, result = registry.invoke("query_orders", {"status": "unpaid", "limit": 1})
    assert result["matched_count"] == 4
    assert result["returned_count"] == 1
    assert result["truncated"] is True
    assert result["total_amount_cents"] == 5435000
    assert result["rows"][0]["status"] == "overdue"


def test_customer_is_exact_match_and_sql_injection_is_inert(registry):
    _, result = registry.invoke("query_orders", {"customer": "星河科技", "status": "unpaid"})
    assert result["matched_count"] == 2
    assert result["total_amount_cents"] == 2270000
    _, result = registry.invoke("query_orders", {"customer": "' OR 1=1 --"})
    assert result["matched_count"] == 0
    assert registry.orders.query("all", None, 50)["matched_count"] == 6


@pytest.mark.parametrize(
    "name,args",
    [
        ("query_orders", {"status": "DROP TABLE orders"}),
        ("query_orders", {"sql": "SELECT * FROM orders"}),
        ("query_orders", {"limit": 51}),
        ("query_orders", {"limit": True}),
        ("query_orders", {"limit": "10"}),
        ("search_documents", {"query": "test", "top_k": 0}),
        ("search_documents", {"query": "x" * 501}),
    ],
)
def test_tool_argument_validation(registry, name, args):
    _, result = registry.invoke(name, args)
    assert result["error"] == "invalid_arguments"


def test_unknown_tool_and_malformed_json(registry):
    assert registry.execute("run_shell", "{}")[1]["error"] == "unknown_tool"
    for raw in ("{bad", "[]", "null", "x" * 4001):
        assert registry.execute("query_orders", raw)[1]["error"] == "invalid_arguments"


def test_readonly_connection_and_seed_integrity(registry):
    with sqlite3.connect(registry.orders.path.as_uri() + "?mode=ro", uri=True) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM orders")
    assert registry.orders.query("all", None, 50)["matched_count"] == 6


def test_strict_function_schemas():
    for tool in tool_definitions():
        function = tool["function"]
        schema = function["parameters"]
        assert function["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        json.dumps(tool, ensure_ascii=False)
