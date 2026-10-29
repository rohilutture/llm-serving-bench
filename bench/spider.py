"""Load Spider examples and render database schemas for prompting."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Example:
    idx: int
    db_id: str
    question: str
    gold_sql: str
    difficulty: str = "unknown"


def db_path(db_root: Path, db_id: str) -> Path:
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def load_examples(dev_json: Path, limit: int | None = None,
                  difficulty_file: Path | None = None) -> list[Example]:
    """Read Spider dev.json. Difficulty labels are optional (a JSON list of
    strings aligned with dev.json, e.g. exported from your text-to-SQL repo's
    harness) so per-tier accuracy can be compared with earlier results."""
    rows = json.loads(Path(dev_json).read_text(encoding="utf-8"))
    tiers = None
    if difficulty_file:
        tiers = json.loads(Path(difficulty_file).read_text(encoding="utf-8"))
        if len(tiers) != len(rows):
            raise ValueError("difficulty file length does not match dev.json")
    out = []
    for i, r in enumerate(rows[:limit] if limit else rows):
        out.append(Example(i, r["db_id"], r["question"], r["query"],
                           tiers[i] if tiers else "unknown"))
    return out


@lru_cache(maxsize=512)
def schema_ddl(db_file: str) -> str:
    """CREATE TABLE statements straight from sqlite_master — the most faithful
    schema text, including foreign keys, which help schema linking."""
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    finally:
        con.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


@lru_cache(maxsize=8)
def _tables(tables_json: str) -> dict:
    from .train_format import load_tables
    return load_tables(tables_json)


@lru_cache(maxsize=512)
def schema_text(db_file: str, fmt: str | None = None) -> str:
    """Schema text for the prompt. Default format = the exact one the adapter was trained with
    (`bench.train_format.serialize_schema`, with sample rows, as in the training run).

    BENCH_SCHEMA_FORMAT=ddl switches to raw CREATE TABLE statements (explicit opt-in only).
    BENCH_INCLUDE_SAMPLES=0 turns sample rows off (only if training was run without --include-samples).
    A missing tables.json is an error, never a silent fallback to a different format.
    """
    import os
    fmt = fmt or os.environ.get("BENCH_SCHEMA_FORMAT", "train")
    if fmt == "ddl":
        return schema_ddl(db_file)
    if fmt != "train":
        raise ValueError(f"unknown schema format {fmt!r} (use 'train' or 'ddl')")
    from .train_format import serialize_schema
    p = Path(db_file)
    db_id, tables_json = p.stem, p.parents[2] / "tables.json"
    if not tables_json.exists():
        raise FileNotFoundError(
            f"{tables_json} not found. The training-format schema is built from Spider's tables.json "
            "(next to dev.json). Pass the Spider root that contains it, or set BENCH_SCHEMA_FORMAT=ddl "
            "if you really want a different prompt format (this will not match how the adapter was trained).")
    entry = _tables(str(tables_json))[db_id]
    include_samples = os.environ.get("BENCH_INCLUDE_SAMPLES", "1") != "0"
    return serialize_schema(entry, db_path=str(p), include_samples=include_samples)
