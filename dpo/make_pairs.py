"""Build execution-verified preference pairs for DPO.

For each Spider TRAIN question: sample k completions from the fine-tuned model,
execute each one, and pair a correct query (chosen) with an incorrect one
(rejected). The reward is the database itself, so no human labels or judge
model are needed — the same signal the eval harness uses.

Questions where all k samples agree (all right or all wrong) carry no
preference signal and are skipped.

    python dpo/make_pairs.py --model models/merged-bf16 --spider-dir data/spider \
        --k 8 --out dpo/pairs.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.exec_match import compare
from bench.prompts import DEFAULT_SYSTEM, extract_sql, load_template, render_chat, user_message
from bench.spider import db_path, schema_text


def pick_pairs(samples, max_pairs):
    """samples: list[(text, correct, kind)]. Prefer rejected answers that RUN but
    return wrong rows: they are the hard negatives (invalid SQL is already rare
    after SFT, and teaching 'don't emit syntax errors' is the easy part)."""
    good = [s for s in samples if s[1]]
    bad = sorted([s for s in samples if not s[1]], key=lambda s: s[2] != "wrong_rows")
    pairs, seen = [], set()
    for g in good:
        for b in bad:
            key = (g[0], b[0])
            if key not in seen:
                seen.add(key)
                pairs.append((g[0], b[0]))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--spider-dir", default="data/spider")
    p.add_argument("--prompt-template")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-questions", type=int, default=7000)
    p.add_argument("--pairs-per-question", type=int, default=2)
    p.add_argument("--out", default="dpo/pairs.jsonl")
    a = p.parse_args()

    from vllm import LLM, SamplingParams
    spider = Path(a.spider_dir)
    train = json.loads((spider / "train_spider.json").read_text(encoding="utf-8"))
    random.Random(0).shuffle(train)
    train = train[:a.max_questions]
    tpl = load_template(a.prompt_template)

    llm = LLM(model=a.model, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    prompts = [render_chat(tok, DEFAULT_SYSTEM, user_message(
        tpl, schema_text(str(db_path(spider / "database", r["db_id"]))), r["question"]))
        for r in train]
    outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=a.temperature,
                                                max_tokens=256, seed=0))
    n_pairs = n_mixed = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r, prompt, o in zip(train, prompts, outs):
            db = str(db_path(spider / "database", r["db_id"]))
            samples = []
            for c in o.outputs:
                sql = extract_sql(c.text)
                m = compare(db, sql, r["query"])
                samples.append((sql, m.correct, m.kind))
            pairs = pick_pairs(samples, a.pairs_per_question)
            n_mixed += bool(pairs)
            for chosen, rejected in pairs:
                f.write(json.dumps({"prompt": prompt, "chosen": chosen + tok.eos_token,
                                    "rejected": rejected + tok.eos_token}) + "\n")
                n_pairs += 1
    print(f"{n_mixed}/{len(train)} questions had mixed outcomes -> {n_pairs} pairs")


if __name__ == "__main__":
    main()
