"""Execution-based correctness: run predicted and gold SQL, compare result sets.

Rules (ported from the text-to-sql-finetune repo's `src/eval/execution.py`, so this benchmark scores
predictions exactly the way the adapter's 74.6% was scored):
- If the gold query has ORDER BY, row order matters; otherwise compare multisets.
- Column order does NOT matter: a correct answer whose SELECT columns are permuted still counts
  (permutations are tried up to 6 columns). A second, stricter number (`strict`: identical column order)
  is recorded alongside, and reported as `exec_accuracy_strict`.
- Floats are rounded to 4 places so 3.00001 == 3.0.
- Invalid SQL, or a query that runs past the timeout, counts as wrong, with the
  failure kind recorded so error breakdowns are possible.
"""
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from itertools import permutations

TIMEOUT_S = 30.0  # same as the training repo's evaluator
FLOAT_PLACES = 4
MAX_PERM_COLS = 6


@dataclass
class MatchResult:
    correct: bool
    kind: str  # "match" | "wrong_rows" | "invalid_sql" | "timeout" | "gold_error"
    detail: str = ""
    strict: bool = False  # also correct with identical column order (stricter than `correct`)


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
    order = has_order_by(gold_sql)
    strict = pred == gold if order else Counter(pred) == Counter(gold)
    same = result_sets_match(gold, pred, order)
    return MatchResult(same, "match" if same else "wrong_rows", strict=strict and same)


def _round(v):
    return round(v, FLOAT_PLACES) if isinstance(v, float) else v


def result_sets_match(gold_rows, pred_rows, order_matters: bool) -> bool:
    """Result-set equality, tolerant of column order (ported from the training repo's evaluator).

    Rows are compared as an ordered list when the gold query has ORDER BY, otherwise as a multiset.
    If the column counts match (and are <= MAX_PERM_COLS), every column permutation of the prediction is tried.
    """
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    n_cols = len(gold_rows[0])
    if n_cols != len(pred_rows[0]):
        return False
    gold = [tuple(_round(c) for c in r) for r in gold_rows]
    pred = [tuple(_round(c) for c in r) for r in pred_rows]
    perms = permutations(range(n_cols)) if n_cols <= MAX_PERM_COLS else [tuple(range(n_cols))]
    gold_counter = None if order_matters else Counter(gold)
    for perm in perms:
        permuted = [tuple(r[i] for i in perm) for r in pred]
        if (gold == permuted) if order_matters else (gold_counter == Counter(permuted)):
            return True
    return False
