from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class BatchTiming:
    """Wall-clock for one generate() call; offline throughput is computed from these."""
    n_requests: int
    seconds: float
    completion_tokens: int


@dataclass
class BackendInfo:
    name: str
    model: str
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    info: BackendInfo

    def render(self, system: str, user: str) -> str:
        """Turn a system+user message into the exact prompt string the model sees."""

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[Generation]:
        """Greedy decode every prompt. Must preserve order."""

    def peak_memory_gb(self) -> float | None:
        """Peak accelerator memory, if meaningful for this backend."""
