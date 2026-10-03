import pytest
from conftest import DOC_A

from backend.embedder import CachedEmbedder, HashEmbedder
from backend.ingestion import content_hash, extract_text


def test_hash_is_stable_and_content_sensitive():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")


def test_reingesting_identical_bytes_is_a_noop(pipeline):
    first = pipeline.ingest("a.txt", DOC_A.encode())
    again = pipeline.ingest("renamed.txt", DOC_A.encode())  # same content, different name
    assert not first.skipped and first.n_chunks > 0
    assert again.skipped and again.doc_id == first.doc_id
    assert pipeline.store.count() == first.n_chunks
    assert len(pipeline.bm25) == first.n_chunks


def test_changed_content_is_a_new_document(pipeline):
    a = pipeline.ingest("a.txt", DOC_A.encode())
    b = pipeline.ingest("a.txt", (DOC_A + " Extra sentence.").encode())
    assert a.doc_id != b.doc_id and not b.skipped


def test_unsupported_type():
    with pytest.raises(ValueError):
        extract_text("x.exe", b"")


def test_docx_roundtrip():
    import io

    from docx import Document

    d = Document()
    d.add_paragraph("Hello docx world")
    buf = io.BytesIO()
    d.save(buf)
    assert "Hello docx world" in extract_text("t.docx", buf.getvalue())


def test_embedding_cache_counts_hits():
    emb = CachedEmbedder(HashEmbedder())
    emb.embed(["a b", "c d"])
    emb.embed(["a b", "e f"])
    assert (emb.hits, emb.misses) == (1, 3)
    assert emb.hit_rate == pytest.approx(0.25)
