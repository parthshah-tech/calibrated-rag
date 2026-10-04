"""History-aware rewriting: turn a follow-up ("what about the second one?") into a standalone
search query using the conversation so far. No history means no model call."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLMClient, LLMError

MAX_TURNS = 6
MAX_CHARS = 600
SYSTEM_PROMPT = (
    "You turn a follow-up question into a standalone search query for a document search engine. "
    "Use the conversation to resolve references like 'it', 'the second one' or 'that'. "
    "If the question is already standalone, return it unchanged. "
    "Output only the query on one line, with no quotes and no explanation."
)


@dataclass
class Rewrite:
    query: str
    rewritten: bool  # the query actually changed
    cached: bool = False
    error: str | None = None


def trim_history(history: list[dict] | None) -> list[dict]:
    """Keep the last few well-formed turns, each cut to a bounded length."""
    out = []
    for turn in (history or [])[-MAX_TURNS:]:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content[:MAX_CHARS]})
    return out


def clean_query(text: str, fallback: str) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^(standalone query|query)\s*:\s*", "", line, flags=re.I).strip(" \"'`")
    return line[:300] or fallback


def rewrite_query(llm: LLMClient, question: str, history: list[dict] | None) -> Rewrite:
    turns = trim_history(history)
    if not turns:
        return Rewrite(question, False)
    convo = "\n".join(
        f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}" for t in turns
    )
    prompt = f"Conversation:\n{convo}\n\nLatest question: {question}\n\nStandalone query:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        result = llm.complete(messages, temperature=0.0, max_tokens=300)
    except LLMError as e:  # retrieval must survive: fall back to the question as typed
        return Rewrite(question, False, error=str(e))
    query = clean_query(result.text, question)
    return Rewrite(query, query.lower() != question.strip().lower(), result.cached)
