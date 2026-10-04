"""LLM access through any OpenAI-compatible /chat/completions endpoint (Groq, OpenRouter,
Ollama, ...), using only the standard library.

CachedLLM stores deterministic (temperature 0) responses on disk so benchmark and study runs
are repeatable, free to re-run, and unaffected by rate limits or model changes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

RETRY_STATUS = (429, 500, 502, 503, 504)


class LLMError(RuntimeError):
    """Any failure to get text from the model. Messages never contain the API key."""


class _Truncated(LLMError):
    """The model spent its whole output budget (usually on reasoning) before writing any text."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    cached: bool = False


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 600
    ) -> LLMResult: ...


def _retry_after(err: urllib.error.HTTPError) -> float | None:
    try:
        return min(float(err.headers.get("Retry-After", "")), 30.0)
    except (TypeError, ValueError):
        return None


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
        timeout: float = 60.0,
        retries: int = 3,
        sleep=time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout, self.retries, self._sleep = timeout, retries, sleep
        self._key = api_key

    def complete(self, messages, *, temperature: float = 0.0, max_tokens: int = 600) -> LLMResult:
        """Reasoning models can burn the entire budget thinking and return no text. When that
        happens, retry once with three times the budget (capped) instead of failing."""
        try:
            return self._complete_once(messages, temperature, max_tokens)
        except _Truncated:
            return self._complete_once(messages, temperature, min(max_tokens * 3, 3000))

    def _complete_once(self, messages, temperature: float, max_tokens: int) -> LLMResult:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        data = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "User-Agent": "calibrated-rag/0.1",  # some gateways reject the default urllib agent
        }
        payload = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data, headers=headers
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUS and attempt < self.retries:
                    self._sleep(_retry_after(e) or min(2**attempt, 8))
                    continue
                detail = e.read().decode(errors="replace")[:300]
                raise LLMError(f"LLM request failed: HTTP {e.code} {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    self._sleep(min(2**attempt, 8))
                    continue
                raise LLMError(f"LLM unreachable: {getattr(e, 'reason', e)}") from e
        try:
            choice = payload["choices"][0]
            text = (choice["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("LLM response had an unexpected shape") from e
        if not text:
            if choice.get("finish_reason") == "length":
                raise _Truncated(
                    "LLM returned no text (the output budget was used up by reasoning, even "
                    "after one retry with a larger budget; try LLM_REASONING_EFFORT=low)"
                )
            raise LLMError("LLM returned no text")
        return LLMResult(text, payload.get("model", self.model))


class CachedLLM:
    def __init__(self, inner: LLMClient, cache_dir: str | Path) -> None:
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def _key(self, messages, temperature, max_tokens) -> str:
        blob = json.dumps(
            {"model": self.model, "messages": messages, "t": temperature, "max": max_tokens},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, messages, *, temperature: float = 0.0, max_tokens: int = 600) -> LLMResult:
        if temperature != 0:  # sampled output is not reproducible, so never cached
            return self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
        path = self.dir / f"{self._key(messages, temperature, max_tokens)}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            return LLMResult(d["text"], d["model"], cached=True)
        result = self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
        self.misses += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"text": result.text, "model": result.model}, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written entry
        return result


class FakeLLM:
    """Scripted responses for tests; records every call."""

    model = "fake"

    def __init__(self, responses: list[str] | None = None, error: Exception | None = None) -> None:
        self.responses = list(responses or ["fake answer [1]"])
        self.error = error
        self.calls: list[list[dict]] = []

    def complete(self, messages, *, temperature: float = 0.0, max_tokens: int = 600) -> LLMResult:
        self.calls.append(messages)
        if self.error:
            raise self.error
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return LLMResult(text, self.model)


def build_llm(settings) -> LLMClient | None:
    """None when no key is configured: the app then serves sources without a written answer."""
    if not settings.llm_api_key:
        return None
    inner = OpenAICompatClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_reasoning_effort,
    )
    return CachedLLM(inner, Path(settings.data_dir) / "llm_cache")
