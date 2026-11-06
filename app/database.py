"""Only predefined parameterized SELECT statements reach the read-only database."""

import json
import sqlite3
from pathlib import Path

from app.config import PACKAGE_DIR


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY, customer TEXT NOT NULL,
            amount_cents INTEGER NOT NULL CHECK(amount_cents >= 0),
            status TEXT NOT NULL CHECK(status IN ('paid','unpaid','overdue')),
            created_at TEXT NOT NULL, due_date TEXT NOT NULL
        )""")
        rows = json.loads((PACKAGE_DIR / "sample_data/orders.json").read_text(encoding="utf-8"))
        conn.executemany(
            "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?)",
            [
                (
                    row["order_id"],
                    row["customer"],
                    row["amount_cents"],
                    row["status"],
                    row["created_at"],
                    row["due_date"],
                )
                for row in rows
            ],
        )


class OrderStore:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def query(self, status: str, customer: str | None, limit: int) -> dict:
        clauses, params = [], []
        if status == "unpaid":
            clauses.append("status IN ('unpaid', 'overdue')")
        elif status != "all":
            clauses.append("status = ?")
            params.append(status)
        if customer:
            clauses.append("customer = ?")
            params.append(customer)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        # mode=ro is the enforcement boundary; model output never becomes SQL.
        with sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            summary = conn.execute(
                "SELECT COUNT(*) AS count, COALESCE(SUM(amount_cents),0) AS total_cents "
                "FROM orders" + where,
                params,
            ).fetchone()
            rows = conn.execute(
                "SELECT order_id, customer, amount_cents, status, created_at, due_date "
                "FROM orders" + where + " ORDER BY due_date, order_id LIMIT ?",
                [*params, limit],
            ).fetchall()
        return {
            "rows": [dict(row) for row in rows],
            "matched_count": summary["count"],
            "total_amount_cents": summary["total_cents"],
            "returned_count": len(rows),
            "truncated": summary["count"] > len(rows),
            "currency": "CNY",
            "snapshot_date": "2026-11-01",
            "data_note": "虚构演示快照；逾期状态非实时计算",
        }
