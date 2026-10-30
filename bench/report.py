"""Collect results/*/summary.json into a markdown table + accuracy/throughput plot.

    python -m bench.report --results results --out results/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COLS = [("run", "Run"), ("exec_accuracy", "Exec acc"), ("exec_accuracy_strict", "Strict acc"),
        ("output_tokens_per_s", "Out tok/s"),
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


def regressions(rows, max_drop_pts: float):
    """Runs that lose more than `max_drop_pts` accuracy points vs the baseline: fast but not usable."""
    return [r["run"] for r in rows if r["delta_acc_pts"] is not None and r["delta_acc_pts"] < -max_drop_pts]


def table(rows, bad=()):
    fmt = lambda v: "—" if v is None else str(v)
    lines = ["| " + " | ".join(h for _, h in COLS) + " |",
             "|" + "---|" * len(COLS)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k, _ in COLS) + " |")
    return "\n".join(lines)


def online_table(results_dir: Path) -> str:
    """Latency under concurrency, from results/<run>/load_test.json (bench.load_test)."""
    rows = []
    for f in sorted(results_dir.glob("*/load_test.json")):
        for r in json.loads(f.read_text()):
            rows.append((f.parent.name, r))
    if not rows:
        return ""
    head = ["Run", "Concurrency", "TTFT p50 (s)", "Latency p50 (s)", "Latency p95 (s)", "Latency p99 (s)", "Out tok/s"]
    lines = ["\n## Online latency (streaming, `vllm serve`)\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in sorted(rows, key=lambda x: (x[1]["concurrency"], x[0])):
        lines.append(f"| {name} | {r['concurrency']} | {r['ttft_p50']} | {r['latency_p50']} | {r['latency_p95']} | "
                     f"{r['latency_p99']} | {r['output_tok_per_s']} |")
    return "\n".join(lines) + "\n"


def plot(rows, front, path: Path, bad=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        on = r["run"] in front
        color = "#d64545" if r["run"] in bad else ("#2a6fdb" if on else "#9aa5b1")
        ax.scatter(r.get("output_tokens_per_s") or 0, 100 * r["exec_accuracy"], s=60, color=color, zorder=3)
        ax.annotate(r["run"], (r.get("output_tokens_per_s") or 0, 100 * r["exec_accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Output tokens / s (offline, Spider dev)")
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_title("Speed vs. accuracy (blue = Pareto-optimal, red = accuracy regression)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252 and cannot print the symbols below
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--baseline", default="hf-bf16")
    p.add_argument("--out", default="results/REPORT.md")
    p.add_argument("--max-acc-drop", type=float, default=2.0,
                   help="runs losing more than this many accuracy points vs the baseline are flagged and "
                        "left out of the recommended set")
    a = p.parse_args(argv)
    rows = load(Path(a.results))
    if not rows:
        raise SystemExit("no results/*/summary.json found")
    base = add_relative(rows, a.baseline)
    front = pareto(rows)
    bad = regressions(rows, a.max_acc_drop)
    usable = [f for f in front if f not in bad]
    plot(rows, front, Path(a.out).with_suffix(".png"), bad)
    names = lambda xs: ", ".join(f"`{x}`" for x in xs) or "none"
    md = [f"# Serving benchmark\n\nBaseline: `{base['run']}`.\n",
          f"- Pareto-optimal on (accuracy, throughput): {names(front)}",
          f"- Pareto-optimal AND within {a.max_acc_drop:g} accuracy points of the baseline (the usable set): {names(usable)}",
          f"- ⚠ Accuracy regression (more than {a.max_acc_drop:g} pts below baseline; fast but not usable): {names(bad)}\n",
          "`Exec acc` ignores column order (the training repo's rule); `Strict acc` requires identical column order. "
          "`Weights GB` is blank for runs that quantize at load time.\n",
          table(rows, bad),
          f"\n![speed vs accuracy]({Path(a.out).with_suffix('.png').name})\n",
          online_table(Path(a.results))]
    Path(a.out).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
