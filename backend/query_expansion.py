"""Multi-query expansion: ask the model for a few differently-worded versions of the question
so retrieval can match documents that use other words."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLMClient, LLMError

SYSTEM_PROMPT = (
    "You write alternative search queries for a document search engine. Given a question, write "
    "{n} different queries that would find passages answering it, using different words and "
    "angles (synonyms, more specific terms, related concepts). Output one query per line, with "
    "no numbering, no quotes and no explanation."
)


@dataclass
class Expansion:
    queries: list[str]
    cached: bool = False
    error: str | None = None


def parse_queries(text: str, original: str, n: int) -> list[str]:
    """One query per line; drops bullets, numbering, quotes, blanks, duplicates and the original."""
    seen = {original.strip().lower()}
    out: list[str] = []
    for line in text.splitlines():
        q = re.sub(r"^[\s\-*\d.)]+", "", line).strip(" \"'`")[:200]
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) == n:
            break
    return out


def expand_query(llm: LLMClient, query: str, n: int = 3) -> Expansion:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
        {"role": "user", "content": query},
    ]
    try:
        result = llm.complete(messages, temperature=0.0, max_tokens=300)
    except LLMError as e:
        return Expansion([], error=str(e))
    return Expansion(parse_queries(result.text, query, n), result.cached)
