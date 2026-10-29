import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def spider_dir(tmp_path):
    """A two-question Spider-format dataset with one real SQLite database."""
    root = tmp_path / "spider"
    db_dir = root / "database" / "concert"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "concert.sqlite")
    con.executescript("""
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER, country TEXT);
        INSERT INTO singer VALUES (1,'Ann',30,'France'),(2,'Bo',45,'USA'),(3,'Cy',22,'USA');
    """)
    con.commit()
    con.close()
    dev = [
        {"db_id": "concert", "question": "How many singers are there?",
         "query": "SELECT count(*) FROM singer"},
        {"db_id": "concert", "question": "Names of singers from the USA ordered by age.",
         "query": "SELECT name FROM singer WHERE country = 'USA' ORDER BY age"},
    ]
    (root / "dev.json").write_text(json.dumps(dev))
    tables = [{"db_id": "concert", "table_names_original": ["singer"],
               "column_names_original": [[-1, "*"], [0, "singer_id"], [0, "name"], [0, "age"], [0, "country"]],
               "column_types": ["text", "number", "text", "number", "text"],
               "primary_keys": [1], "foreign_keys": []}]
    (root / "tables.json").write_text(json.dumps(tables))
    return root
