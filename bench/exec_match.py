"""Execution-based correctness: run predicted and gold SQL, compare result sets.

Rules (same spirit as the Spider execution metric):
- If the gold query has ORDER BY, row order matters; otherwise compare multisets.
- Floats are rounded so 3.0000001 == 3.0.
- Invalid SQL, or a query that runs past the timeout, counts as wrong, with the
  failure kind recorded so error breakdowns are possible.
"""
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

TIMEOUT_S = 10.0


@dataclass
class MatchResult:
    correct: bool
    kind: str  # "match" | "wrong_rows" | "invalid_sql" | "timeout" | "gold_error"
    detail: str = ""


def _norm(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, bytes):
        return v.hex()
    return v


def run_query(db_file: str, sql: str, timeout_s: float = TIMEOUT_S):
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    # Returning non-zero from the progress handler aborts the query.
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        rows = con.execute(sql).fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise TimeoutError(sql) from e
        raise
    finally:
        con.close()
    return [tuple(_norm(v) for v in r) for r in rows]


def has_order_by(sql: str) -> bool:
    return "order by" in " ".join(sql.lower().split())


def compare(db_file: str, pred_sql: str, gold_sql: str) -> MatchResult:
    try:
        gold = run_query(db_file, gold_sql)
    except Exception as e:  # dataset problem, not a model problem
        return MatchResult(False, "gold_error", str(e))
    if not pred_sql.strip():
        return MatchResult(False, "invalid_sql", "empty prediction")
    try:
        pred = run_query(db_file, pred_sql)
    except TimeoutError:
        return MatchResult(False, "timeout")
    except Exception as e:
        return MatchResult(False, "invalid_sql", str(e)[:200])
    same = pred == gold if has_order_by(gold_sql) else Counter(pred) == Counter(gold)
    return MatchResult(same, "match" if same else "wrong_rows")
