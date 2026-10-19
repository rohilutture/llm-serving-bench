"""A deterministic stand-in backend so the whole pipeline can be tested without a GPU.

It answers from a lookup table (prompt -> SQL) and sleeps a configurable amount
per token, which is enough to exercise accuracy scoring, timing, and reporting.
"""
from __future__ import annotations

import time

from .base import BackendInfo, Generation


class MockBackend:
    def __init__(self, answers: dict[str, str], seconds_per_token: float = 0.0,
                 default: str = "SELECT 1"):
        self.answers = answers
        self.spt = seconds_per_token
        self.default = default
        self.info = BackendInfo("mock", "lookup-table")

    def render(self, system: str, user: str) -> str:
        return user

    def generate(self, prompts, max_new_tokens):
        out = []
        for p in prompts:
            text = self.answers.get(p, self.default)
            n = len(text.split())
            if self.spt:
                time.sleep(self.spt * n)
            out.append(Generation(text, len(p.split()), n))
        return out

    def peak_memory_gb(self):
        return None
