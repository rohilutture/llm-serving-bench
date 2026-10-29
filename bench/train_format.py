"""The exact prompt and schema serialization the text-to-SQL adapter is trained with.

Vendored verbatim from https://github.com/rohilutture/text-to-sql-finetune
(`src/prompts.py` for SYSTEM_PROMPT / USER_TEMPLATE, `src/data/schema.py` for the schema text).
A model fine-tuned on one prompt format and served with another silently loses accuracy, which would make every
speed/accuracy comparison in this repo meaningless. So the benchmark defaults to this format; the DDL format in
`bench.spider.schema_ddl` is an explicit opt-in (BENCH_SCHEMA_FORMAT=ddl), never a silent fallback.
If the training repo's prompt or schema code changes, re-vendor this file.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite SQL. "
    "Given a database schema and a question, output a single valid SQLite query "
    "that answers it. Output only the SQL, with no explanation and no markdown."
)

USER_TEMPLATE = """Schema:
{schema}

Question: {question}

SQL:"""


def load_tables(tables_json_path: str) -> dict[str, dict]:
    """Load tables.json and index it by db_id."""
    with open(tables_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["db_id"]: e for e in entries}


def _sample_rows(db_path: str, table: str, n: int) -> list[tuple]:
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM "{table}" LIMIT {n}')
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def serialize_schema(
    entry: dict,
    db_path: Optional[str] = None,
    include_samples: bool = False,
    n_samples: int = 3,
) -> str:
    """Render one database's schema as text (identical to the training repo's serializer)."""
    tables = entry["table_names_original"]
    columns = entry["column_names_original"]
    col_types = entry["column_types"]
    pk_set = set(entry.get("primary_keys", []))

    cols_by_table: dict[int, list[tuple[int, str, str, bool]]] = {
        i: [] for i in range(len(tables))
    }
    for col_idx, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx < 0:
            continue
        cols_by_table[tbl_idx].append(
            (col_idx, col_name, col_types[col_idx], col_idx in pk_set)
        )

    lines: list[str] = [f"# Database: {entry['db_id']}", ""]
    for tbl_idx, table in enumerate(tables):
        lines.append(f"Table: {table}")
        for _, col_name, col_type, is_pk in cols_by_table[tbl_idx]:
            tag = " [PK]" if is_pk else ""
            lines.append(f"  - {col_name} ({col_type}){tag}")
        if include_samples and db_path:
            rows = _sample_rows(db_path, table, n_samples)
            if rows:
                lines.append(f"  sample rows: {rows}")
        lines.append("")

    fks = entry.get("foreign_keys", [])
    if fks:
        lines.append("Foreign keys:")
        for a, b in fks:
            ta, ca = columns[a]
            tb, cb = columns[b]
            lines.append(f"  - {tables[ta]}.{ca} -> {tables[tb]}.{cb}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
