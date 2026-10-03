"""Per-request tracing: stage timings plus arbitrary counters (LLM calls, cache hits...)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class Tracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self.spans: list[dict] = []
        self.counters: dict[str, float] = {}

    @contextmanager
    def span(self, name: str, **attrs):
        start = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self.spans.append({"name": name, "ms": round(ms, 3), **attrs})

    def count(self, name: str, n: float = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def to_dict(self) -> dict:
        return {
            "total_ms": round((time.perf_counter() - self._t0) * 1000, 3),
            "spans": list(self.spans),
            "counters": dict(self.counters),
        }
