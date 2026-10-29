"""The benchmark must prompt the model exactly the way the adapter was trained (see bench/train_format.py)."""
import pytest

from bench import prompts
from bench.spider import db_path, schema_text
from bench.train_format import SYSTEM_PROMPT, USER_TEMPLATE


@pytest.fixture(autouse=True)
def _fresh_cache():
    schema_text.cache_clear()
    yield
    schema_text.cache_clear()


def test_defaults_are_the_training_prompt():
    assert prompts.DEFAULT_SYSTEM == SYSTEM_PROMPT
    assert prompts.DEFAULT_TEMPLATE == USER_TEMPLATE
    assert prompts.load_template("configs/prompt_template.txt") == USER_TEMPLATE  # file has no stray newline


def test_schema_text_is_training_format_with_samples(spider_dir):
    s = schema_text(str(db_path(spider_dir / "database", "concert")))
    assert s.startswith("# Database: concert\n\nTable: singer\n")
    assert "  - singer_id (number) [PK]" in s
    assert "  sample rows: [(1, 'Ann', 30, 'France')" in s
    assert "CREATE TABLE" not in s


def test_full_user_message_matches_training_layout(spider_dir):
    schema = schema_text(str(db_path(spider_dir / "database", "concert")))
    msg = prompts.user_message(prompts.DEFAULT_TEMPLATE, schema, "How many singers are there?")
    assert msg.startswith("Schema:\n# Database: concert")
    assert msg.endswith("\n\nQuestion: How many singers are there?\n\nSQL:")


def test_missing_tables_json_is_an_error_not_a_silent_fallback(spider_dir):
    (spider_dir / "tables.json").unlink()
    with pytest.raises(FileNotFoundError, match="tables.json"):
        schema_text(str(db_path(spider_dir / "database", "concert")))


def test_ddl_is_explicit_opt_in(spider_dir, monkeypatch):
    monkeypatch.setenv("BENCH_SCHEMA_FORMAT", "ddl")
    assert "CREATE TABLE singer" in schema_text(str(db_path(spider_dir / "database", "concert")))
