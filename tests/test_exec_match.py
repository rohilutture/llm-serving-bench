from bench.exec_match import compare
from bench.prompts import extract_sql
from bench.spider import schema_ddl


def db(spider_dir):
    return str(spider_dir / "database" / "concert" / "concert.sqlite")


def test_equivalent_sql_matches(spider_dir):
    r = compare(db(spider_dir), "SELECT COUNT(singer_id) FROM singer", "SELECT count(*) FROM singer")
    assert r.correct and r.kind == "match"


def test_unordered_gold_ignores_row_order(spider_dir):
    r = compare(db(spider_dir), "SELECT name FROM singer ORDER BY name DESC",
                "SELECT name FROM singer")
    assert r.correct


def test_ordered_gold_requires_order(spider_dir):
    r = compare(db(spider_dir), "SELECT name FROM singer WHERE country='USA' ORDER BY age DESC",
                "SELECT name FROM singer WHERE country = 'USA' ORDER BY age")
    assert not r.correct and r.kind == "wrong_rows"


def test_multiset_counts_duplicates(spider_dir):
    r = compare(db(spider_dir), "SELECT country FROM singer GROUP BY country",
                "SELECT country FROM singer")
    assert not r.correct


def test_invalid_sql(spider_dir):
    r = compare(db(spider_dir), "SELEC name FRM singer", "SELECT name FROM singer")
    assert r.kind == "invalid_sql"


def test_timeout(spider_dir, monkeypatch):
    import bench.exec_match as em
    monkeypatch.setattr(em, "TIMEOUT_S", 0.05)
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT max(x) FROM c")
    r = em.compare(db(spider_dir), slow, "SELECT 1")
    assert r.kind == "timeout"


def test_read_only(spider_dir):
    r = compare(db(spider_dir), "DELETE FROM singer", "SELECT 1")
    assert not r.correct
    assert compare(db(spider_dir), "SELECT count(*) FROM singer",
                   "SELECT count(*) FROM singer").correct


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT a FROM t;\n```\nThis returns a.") == "SELECT a FROM t"
    assert extract_sql("SQL: SELECT a\nFROM t; -- done") == "SELECT a FROM t"
    assert extract_sql("SELECT 1") == "SELECT 1"


def test_schema_ddl(spider_dir):
    assert "CREATE TABLE singer" in schema_ddl(db(spider_dir))


def test_column_order_is_ignored_but_recorded_as_not_strict(spider_dir):
    # Same result set, columns swapped: counts as correct (training-repo rule) but not as strict.
    r = compare(db(spider_dir), "SELECT age, name FROM singer", "SELECT name, age FROM singer")
    assert r.correct and r.kind == "match" and not r.strict


def test_identical_columns_are_strict(spider_dir):
    r = compare(db(spider_dir), "SELECT name, age FROM singer", "SELECT name, age FROM singer")
    assert r.correct and r.strict


def test_column_permutation_does_not_hide_wrong_values(spider_dir):
    r = compare(db(spider_dir), "SELECT age, age FROM singer", "SELECT name, age FROM singer")
    assert not r.correct and r.kind == "wrong_rows"


def test_column_count_mismatch_is_wrong(spider_dir):
    r = compare(db(spider_dir), "SELECT name FROM singer", "SELECT name, age FROM singer")
    assert not r.correct


def test_ordered_gold_with_permuted_columns(spider_dir):
    r = compare(db(spider_dir), "SELECT age, name FROM singer ORDER BY age",
                "SELECT name, age FROM singer ORDER BY age")
    assert r.correct and not r.strict


def test_float_rounding_to_four_places():
    from bench.exec_match import result_sets_match
    assert result_sets_match([(1.00004,)], [(1.00001,)], False)
    assert not result_sets_match([(1.0004,)], [(1.0001,)], False)
