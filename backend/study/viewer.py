"""Rebuild a document's text from its stored chunks for the plain document viewer.
Chunks are spans of the original text, so overlaps are trimmed and the rare dropped
whitespace-only gap becomes a newline."""

from __future__ import annotations

from ..interfaces import Chunk


def reconstruct(chunks: list[Chunk]) -> str:
    out: list[str] = []
    pos = 0
    for c in sorted(chunks, key=lambda c: (c.start, c.end)):
        if c.end <= pos:
            continue
        if c.start > pos:
            out.append("\n")
            pos = c.start
        out.append(c.text[pos - c.start :])
        pos = c.end
    return "".join(out)
