"""Chunking strategies. Every chunk is a span of the original text: text == doc[start:end].

fixed      sliding character window (baseline)
recursive  paragraph -> line -> sentence -> word -> char fallback, merged up to `size`
semantic   Phase 4 (embedding-based topic-shift boundaries)
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")


def _spans(text: str, bounds: list[tuple[int, int]]) -> list[Span]:
    return [Span(s, e, text[s:e]) for s, e in bounds if text[s:e].strip()]


def fixed_chunks(text: str, size: int = 512, overlap: int = 64) -> list[Span]:
    _validate(size, overlap)
    step = size - overlap
    bounds, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        bounds.append((start, end))
        if end == len(text):
            break
        start += step
    return _spans(text, bounds)


def _split(text: str, offset: int, size: int, seps: tuple[str, ...]) -> list[tuple[int, int]]:
    """Split `text` into contiguous pieces of at most `size` chars, preferring coarse separators."""
    if len(text) <= size:
        return [(offset, offset + len(text))]
    if not seps:
        return [(offset + i, offset + min(i + size, len(text))) for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    pieces, pos = [], 0
    while pos < len(text):
        nxt = text.find(sep, pos)
        end = len(text) if nxt == -1 else nxt + len(sep)  # keep the separator with the piece
        piece = text[pos:end]
        if len(piece) <= size:
            pieces.append((offset + pos, offset + end))
        else:
            pieces.extend(_split(piece, offset + pos, size, rest))
        pos = end
    return pieces


def recursive_chunks(
    text: str,
    size: int = 512,
    overlap: int = 64,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[Span]:
    _validate(size, overlap)
    pieces = _split(text, 0, size, separators)
    bounds: list[tuple[int, int]] = []
    cur: list[tuple[int, int]] = []
    for piece in pieces:
        if cur and piece[1] - cur[0][0] > size:
            bounds.append((cur[0][0], cur[-1][1]))
            # carry trailing pieces (up to `overlap` chars) into the next chunk
            carry: list[tuple[int, int]] = []
            for p in reversed(cur):
                if cur[-1][1] - p[0] > overlap:
                    break
                carry.insert(0, p)
            cur = carry
            while cur and piece[1] - cur[0][0] > size:  # carry must leave room for the piece
                cur.pop(0)
        cur.append(piece)
    if cur:
        bounds.append((cur[0][0], cur[-1][1]))
    return _spans(text, bounds)


def chunk_text(text: str, strategy: str = "recursive", size: int = 512, overlap: int = 64):
    if strategy == "fixed":
        return fixed_chunks(text, size, overlap)
    if strategy == "recursive":
        return recursive_chunks(text, size, overlap)
    if strategy == "semantic":
        raise NotImplementedError("semantic chunking is planned for Phase 4")
    raise ValueError(f"unknown chunk strategy: {strategy!r}")
