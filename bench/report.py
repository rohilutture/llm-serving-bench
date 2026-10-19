"""Collect results/*/summary.json into a markdown table + accuracy/throughput plot.

    python -m bench.report --results results --out results/REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [("run", "Run"), ("exec_accuracy", "Exec acc"), ("output_tokens_per_s", "Out tok/s"),
        ("requests_per_s", "Req/s"), ("speedup", "Speed-up"), ("delta_acc_pts", "Δ acc (pts)"),
        ("model_size_gb", "Weights GB"), ("peak_memory_gb", "Peak mem GB")]


def load(results_dir: Path):
    rows = []
    for f in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(f.read_text())
        s["run"] = f.parent.name
        rows.append(s)
    return rows


def add_relative(rows, baseline: str | None):
    base = next((r for r in rows if r["run"] == baseline), rows[0] if rows else None)
    for r in rows:
        if base and base.get("output_tokens_per_s") and r.get("output_tokens_per_s"):
            r["speedup"] = f'{r["output_tokens_per_s"] / base["output_tokens_per_s"]:.2f}x'
        r["delta_acc_pts"] = round(100 * (r["exec_accuracy"] - base["exec_accuracy"]), 2) if base else None
    return base


def pareto(rows):
    """Runs not beaten on BOTH accuracy and throughput by some other run."""
    front = []
    for r in rows:
        dominated = any(o is not r and o["exec_accuracy"] >= r["exec_accuracy"]
                        and (o.get("output_tokens_per_s") or 0) >= (r.get("output_tokens_per_s") or 0)
                        and (o["exec_accuracy"], o.get("output_tokens_per_s")) !=
                        (r["exec_accuracy"], r.get("output_tokens_per_s"))
                        for o in rows)
        if not dominated:
            front.append(r["run"])
    return front


def table(rows):
    fmt = lambda v: "—" if v is None else str(v)
    lines = ["| " + " | ".join(h for _, h in COLS) + " |",
             "|" + "---|" * len(COLS)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k, _ in COLS) + " |")
    return "\n".join(lines)


def plot(rows, front, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        on = r["run"] in front
        ax.scatter(r.get("output_tokens_per_s") or 0, 100 * r["exec_accuracy"],
                   s=60, color="#2a6fdb" if on else "#9aa5b1", zorder=3)
        ax.annotate(r["run"], (r.get("output_tokens_per_s") or 0, 100 * r["exec_accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Output tokens / s (offline, Spider dev)")
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_title("Speed vs. accuracy (blue = Pareto-optimal)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--baseline", default="hf-bf16")
    p.add_argument("--out", default="results/REPORT.md")
    a = p.parse_args(argv)
    rows = load(Path(a.results))
    if not rows:
        raise SystemExit("no results/*/summary.json found")
    base = add_relative(rows, a.baseline)
    front = pareto(rows)
    plot(rows, front, Path(a.out).with_suffix(".png"))
    md = [f"# Serving benchmark\n\nBaseline: `{base['run']}`. Pareto-optimal: "
          + ", ".join(f"`{f}`" for f in front) + "\n", table(rows),
          f"\n![speed vs accuracy]({Path(a.out).with_suffix('.png').name})\n"]
    Path(a.out).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
