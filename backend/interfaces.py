"""Core data types and the small interfaces every swappable component implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>:<index>"
    doc_id: str
    text: str
    source: str  # original filename
    index: int
    start: int  # char offsets into the extracted document text
    end: int


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float  # retriever-native score; only the ranking is comparable across retrievers


@dataclass
class IngestResult:
    doc_id: str
    source: str
    n_chunks: int
    skipped: bool  # True when identical content was already ingested


@dataclass
class RetrievalResult:
    query: str
    mode: str
    dense: list[Hit] = field(default_factory=list)
    bm25: list[Hit] = field(default_factory=list)
    fused: list[Hit] = field(default_factory=list)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    confidence: object | None = None  # ConfidenceResult, set when both lists exist
    trace: dict = field(default_factory=dict)
    rewritten_query: str | None = None  # standalone query used for search, if rewritten
    expansions: list[str] = field(default_factory=list)  # extra queries searched
    notes: list[str] = field(default_factory=list)  # degraded steps, shown to the user


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, embedding: list[float], k: int) -> list[Hit]: ...
    def has_doc(self, doc_id: str) -> bool: ...
    def get(self, chunk_ids: list[str]) -> list[Chunk]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def delete_doc(self, doc_id: str) -> None: ...
    def count(self) -> int: ...
