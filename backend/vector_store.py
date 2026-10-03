"""Vector stores. InMemoryVectorStore is exact brute-force cosine (tests and the baseline
for HNSW comparisons); ChromaVectorStore is the persistent default (HNSW, cosine)."""

from __future__ import annotations

import numpy as np

from .interfaces import Chunk, Hit


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._vecs = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        mat = np.asarray(embeddings, dtype=np.float32)
        self._vecs = mat if self._vecs.size == 0 else np.vstack([self._vecs, mat])
        for c in chunks:
            self._chunks[c.id] = c
            self._order.append(c.id)

    def query(self, embedding: list[float], k: int) -> list[Hit]:
        if not self._order:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        sims = self._vecs @ q / (np.linalg.norm(self._vecs, axis=1) * np.linalg.norm(q) + 1e-12)
        idx = np.lexsort((np.arange(len(sims)), -sims))[:k]  # stable tie-break by insert order
        return [Hit(self._order[i], float(sims[i])) for i in idx]

    def has_doc(self, doc_id: str) -> bool:
        return any(c.doc_id == doc_id for c in self._chunks.values())

    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunks[i] for i in chunk_ids if i in self._chunks]

    def all_chunks(self) -> list[Chunk]:
        return [self._chunks[i] for i in self._order]

    def delete_doc(self, doc_id: str) -> None:
        keep = [i for i, cid in enumerate(self._order) if self._chunks[cid].doc_id != doc_id]
        for cid in [c for c in self._order if self._chunks[c].doc_id == doc_id]:
            del self._chunks[cid]
        self._vecs = self._vecs[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self._order = [self._order[i] for i in keep]

    def count(self) -> int:
        return len(self._order)


class ChromaVectorStore:
    def __init__(self, path: str, collection: str = "chunks") -> None:
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _meta(c: Chunk) -> dict:
        return {
            "doc_id": c.doc_id,
            "source": c.source,
            "index": c.index,
            "start": c.start,
            "end": c.end,
        }

    @staticmethod
    def _chunk(cid: str, text: str, m: dict) -> Chunk:
        return Chunk(cid, m["doc_id"], text, m["source"], m["index"], m["start"], m["end"])

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[self._meta(c) for c in chunks],
        )

    def query(self, embedding: list[float], k: int) -> list[Hit]:
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=min(k, n))
        # Chroma returns cosine *distance*; convert to similarity
        return [Hit(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)]

    def has_doc(self, doc_id: str) -> bool:
        return bool(self._col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        res = self._col.get(ids=chunk_ids, include=["documents", "metadatas"])
        by_id = {
            i: self._chunk(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        }
        return [by_id[i] for i in chunk_ids if i in by_id]

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        chunks = [
            self._chunk(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]
        return sorted(chunks, key=lambda c: (c.doc_id, c.index))

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._col.count()
