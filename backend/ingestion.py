"""Text extraction and idempotent, content-hashed ingestion."""

from __future__ import annotations

import hashlib
import io
import threading

from .chunking import chunk_text
from .interfaces import Chunk, Embedder, IngestResult, VectorStore

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_text(filename: str, data: bytes) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"unsupported file type: {filename!r} (supported: {sorted(SUPPORTED)})")


class Ingestor:
    """Same bytes ingested twice are a no-op. The lock closes the check-then-add race
    within one process; multi-process deployments would need a store-level unique key."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        strategy: str = "recursive",
        size: int = 512,
        overlap: int = 64,
        batch_size: int = 64,
    ) -> None:
        self.store, self.embedder = store, embedder
        self.strategy, self.size, self.overlap = strategy, size, overlap
        self.batch_size = batch_size
        self._lock = threading.Lock()

    def ingest(self, filename: str, data: bytes, on_added=None) -> IngestResult:
        doc_id = content_hash(data)[:16]
        with self._lock:
            if self.store.has_doc(doc_id):
                return IngestResult(doc_id, filename, 0, skipped=True)
            text = extract_text(filename, data)
            spans = chunk_text(text, self.strategy, self.size, self.overlap)
            chunks = [
                Chunk(f"{doc_id}:{i}", doc_id, s.text, filename, i, s.start, s.end)
                for i, s in enumerate(spans)
            ]
            if chunks:
                embeddings: list[list[float]] = []
                for i in range(0, len(chunks), self.batch_size):
                    batch = chunks[i : i + self.batch_size]
                    embeddings.extend(self.embedder.embed([c.text for c in batch]))
                self.store.add(chunks, embeddings)
                if on_added:
                    on_added(chunks)
            return IngestResult(doc_id, filename, len(chunks), skipped=False)
