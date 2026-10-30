"""Online load test against an OpenAI-compatible server (vLLM `vllm serve`).

Offline throughput says how fast a batch job finishes; users of an API feel
latency. This sweeps concurrency and records, per level:
  TTFT (time to first token), end-to-end latency p50/p95/p99, throughput.
Requests are streamed so TTFT is measured, not inferred.

    vllm serve merged/ --enable-prefix-caching &
    python -m bench.load_test --url http://localhost:8000/v1 --model merged/ \
        --spider-dir data/spider --concurrency 1 4 16 64 --n 256 --name vllm-bf16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

from .prompts import DEFAULT_SYSTEM, load_template, user_message
from .spider import db_path, load_examples, schema_text


async def one_request(client, url, model, user, max_tokens):
    body = {"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": DEFAULT_SYSTEM},
                         {"role": "user", "content": user}],
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft, tokens = None, 0
    async with client.stream("POST", f"{url}/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if ttft is None and chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                ttft = time.perf_counter() - t0
    return {"ttft": ttft, "latency": time.perf_counter() - t0, "tokens": tokens}


async def level(url, model, users, concurrency, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300) as client:
        async def guarded(u):
            async with sem:
                return await one_request(client, url, model, u, max_tokens)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(guarded(u) for u in users))
        wall = time.perf_counter() - t0
    lat = np.array([r["latency"] for r in res])
    ttft = np.array([r["ttft"] for r in res if r["ttft"] is not None])
    pct = lambda a, q: round(float(np.percentile(a, q)), 4) if len(a) else None
    return {"concurrency": concurrency, "n": len(res), "wall_s": round(wall, 3),
            "req_per_s": round(len(res) / wall, 3),
            "output_tok_per_s": round(sum(r["tokens"] for r in res) / wall, 1),
            "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
            "latency_p50": pct(lat, 50), "latency_p95": pct(lat, 95),
            "latency_p99": pct(lat, 99)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="name the server registered")
    p.add_argument("--spider-dir", default="data/spider")
    p.add_argument("--prompt-template")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--name", required=True)
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the discarded warm-up pass (without it, later concurrency levels replay prompts "
                        "that earlier levels already put in the prefix cache)")
    a = p.parse_args(argv)

    spider = Path(a.spider_dir)
    tpl = load_template(a.prompt_template)
    exs = load_examples(spider / "dev.json")
    random.Random(0).shuffle(exs)
    users = [user_message(tpl, schema_text(str(db_path(spider / "database", e.db_id))), e.question)
             for e in exs[:a.n]]
    rows = []
    if not a.no_warmup:  # populate the prefix cache / autotune once, so every level starts from the same state
        asyncio.run(level(a.url, a.model, users, 16, a.max_tokens))
    for c in a.concurrency:
        r = asyncio.run(level(a.url, a.model, users, c, a.max_tokens))
        print(json.dumps(r))
        rows.append(r)
    out = Path("results") / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "load_test.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
