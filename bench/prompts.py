"""Prompt construction and SQL extraction.

IMPORTANT: the fine-tuned model was trained on one specific prompt format. Serving
it with a different format silently costs accuracy and would make every
quantization comparison here misleading. Put the exact template from the
text-to-SQL repo in a file and pass it with --prompt-template. Placeholders:
{schema} and {question}.
"""
from __future__ import annotations

import re
from pathlib import Path

from .train_format import SYSTEM_PROMPT as DEFAULT_SYSTEM  # noqa: E402  (exact training prompt)
from .train_format import USER_TEMPLATE as DEFAULT_TEMPLATE  # noqa: E402


def load_template(path: str | None) -> str:
    if not path:
        return DEFAULT_TEMPLATE
    text = Path(path).read_text(encoding="utf-8")
    for key in ("{schema}", "{question}"):
        if key not in text:
            raise ValueError(f"prompt template is missing {key}")
    return text


def user_message(template: str, schema: str, question: str) -> str:
    return template.format(schema=schema, question=question)


def render_chat(tokenizer, system: str, user: str) -> str:
    """Apply the model's own chat template so HF and vLLM see identical text."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """Pull one statement out of a generation: prefer a fenced block, drop any
    trailing explanation, keep only the first statement."""
    m = _FENCE.search(text)
    sql = m.group(1) if m else text
    sql = sql.strip()
    # Some models prefix "SQL:"; strip it.
    sql = re.sub(r"^(sql|query)\s*:\s*", "", sql, flags=re.I)
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    return " ".join(sql.split())
