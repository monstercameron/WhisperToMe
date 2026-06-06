from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator


@dataclass(frozen=True)
class StageTiming:
    name: str
    elapsed_ms: float


@contextmanager
def timed_stage(name: str) -> Iterator[dict[str, float]]:
    start = perf_counter()
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        result["elapsed_ms"] = (perf_counter() - start) * 1000.0

