"""End-to-end with the mock backend: prompts -> generate -> score -> summary -> report."""
import json

from bench import report, runner
from bench.backends.mock import MockBackend
from bench.prompts import DEFAULT_TEMPLATE, user_message
from bench.spider import db_path, load_examples, schema_ddl


def answers_for(spider_dir, examples, preds):
    out = {}
    for ex, sql in zip(examples, preds):
        schema = schema_ddl(str(db_path(spider_dir / "database", ex.db_id)))
        out[user_message(DEFAULT_TEMPLATE, schema, ex.question)] = sql
    return out


def run_mock(spider_dir, tmp_path, preds, name, spt=0.0):
    exs = load_examples(spider_dir / "dev.json")
    be = MockBackend(answers_for(spider_dir, exs, preds), seconds_per_token=spt)
    return runner.run(be, exs, spider_dir / "database", tmp_path / "results" / name,
                      DEFAULT_TEMPLATE, batch_size=1)


def test_perfect_run(spider_dir, tmp_path):
    s = run_mock(spider_dir, tmp_path, ["```sql\nSELECT count(*) FROM singer;```",
                                        "SELECT name FROM singer WHERE country='USA' ORDER BY age"],
                 "perfect")
    assert s["exec_accuracy"] == 1.0
    assert s["by_kind"] == {"match": 2}
    rows = [json.loads(l) for l in open(tmp_path / "results/perfect/predictions.jsonl")]
    assert [r["idx"] for r in rows] == [0, 1]


def test_order_preserved_when_sorted(spider_dir, tmp_path):
    exs = load_examples(spider_dir / "dev.json")
    be = MockBackend(answers_for(spider_dir, exs, ["A", "B"]))
    prompts = runner.build_prompts(be, exs, spider_dir / "database", DEFAULT_TEMPLATE)
    gens, timings = runner.generate_all(be, list(reversed(prompts)), 1, 16, sort_by_length=True)
    assert [g.text for g in gens] == ["B", "A"]
    assert sum(t["n"] for t in timings) == 2


def test_report_relative_and_pareto(spider_dir, tmp_path):
    run_mock(spider_dir, tmp_path, ["SELECT count(*) FROM singer", "SELECT 1"], "hf-bf16", spt=0.01)
    run_mock(spider_dir, tmp_path, ["SELECT count(*) FROM singer",
                                    "SELECT name FROM singer WHERE country='USA' ORDER BY age"],
             "vllm-bf16")
    out = tmp_path / "results" / "REPORT.md"
    report.main(["--results", str(tmp_path / "results"), "--baseline", "hf-bf16", "--out", str(out)])
    text = out.read_text()
    assert "vllm-bf16" in text and "hf-bf16" in text
    assert out.with_suffix(".png").exists()
    rows = report.load(tmp_path / "results")
    report.add_relative(rows, "hf-bf16")
    v = next(r for r in rows if r["run"] == "vllm-bf16")
    assert v["delta_acc_pts"] == 50.0
    assert report.pareto(rows) == ["vllm-bf16"]


def test_dpo_pair_selection_prefers_wrong_rows():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dpo"))
    from make_pairs import pick_pairs
    samples = [("good", True, "match"), ("syntax", False, "invalid_sql"),
               ("near-miss", False, "wrong_rows")]
    assert pick_pairs(samples, 1) == [("good", "near-miss")]
    assert pick_pairs([("a", True, "match")], 2) == []
