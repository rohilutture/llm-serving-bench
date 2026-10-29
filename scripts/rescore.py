"""Re-score saved predictions with the current metric (no generation, no GPU).

    python scripts/rescore.py --run results/hf-bf16-adapter --spider-dir /data/spider

Useful after the matching rules change: compares the stored verdicts with fresh ones, so a difference in the
accuracy number can be attributed to scoring vs. generation.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.exec_match import compare  # noqa: E402
from bench.spider import db_path  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="results/<name> containing predictions.jsonl")
p.add_argument("--spider-dir", default="data/spider")
a = p.parse_args()

rows = [json.loads(l) for l in Path(a.run, "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l]
db_root = Path(a.spider_dir) / "database"
kinds, correct, strict, changed = Counter(), 0, 0, []
scored = 0
for r in rows:
    m = compare(str(db_path(db_root, r["db_id"])), r["pred"], r["gold"])
    kinds[m.kind] += 1
    if m.kind == "gold_error":
        continue
    scored += 1
    correct += m.correct
    strict += m.strict
    if m.correct != r.get("correct"):
        changed.append(r["idx"])
old = sum(bool(r.get("correct")) for r in rows if r.get("kind") != "gold_error")
print(json.dumps({
    "n": len(rows), "n_scored": scored,
    "stored_exec_accuracy": round(old / max(scored, 1), 4),
    "rescored_exec_accuracy": round(correct / max(scored, 1), 4),
    "rescored_exec_accuracy_strict": round(strict / max(scored, 1), 4),
    "verdicts_changed": len(changed), "by_kind": dict(kinds)}, indent=2))
