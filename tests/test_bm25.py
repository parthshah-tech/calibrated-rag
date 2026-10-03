from backend.bm25_index import BM25Index, tokenize
from backend.interfaces import Chunk


def chunk(i, text):
    return Chunk(f"d:{i}", "d", text, "f.txt", i, 0, len(text))


def build():
    idx = BM25Index()
    idx.add(
        [
            chunk(0, "the cat sat on the mat"),
            chunk(1, "dogs chase cats in the park"),
            chunk(2, "quantum chromodynamics describes the strong interaction"),
            chunk(3, "error code XJ-4471 means the pump failed"),
        ]
    )
    return idx


def test_tokenizer():
    assert tokenize("Hello, World! XJ-4471") == ["hello", "world", "xj", "4471"]


def test_exact_term_ranks_first():
    hits = build().search("XJ-4471 pump", 3)
    assert hits[0].chunk_id == "d:3"


def test_only_matching_documents_returned():
    ids = [h.chunk_id for h in build().search("quantum", 10)]
    assert ids == ["d:2"]


def test_no_match_and_empty_cases():
    assert build().search("zebra", 5) == []
    assert build().search("", 5) == []
    assert BM25Index().search("anything", 5) == []


def test_k_limits_results_and_scores_descend():
    hits = build().search("the cat park", 2)
    assert len(hits) <= 2
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_incremental_add_and_rebuild_agree():
    inc = build()
    inc.add([chunk(4, "zebra stripes are unique")])
    rebuilt = BM25Index()
    rebuilt.rebuild_from(
        [
            chunk(0, "the cat sat on the mat"),
            chunk(1, "dogs chase cats in the park"),
            chunk(2, "quantum chromodynamics describes the strong interaction"),
            chunk(3, "error code XJ-4471 means the pump failed"),
            chunk(4, "zebra stripes are unique"),
        ]
    )
    assert [h.chunk_id for h in inc.search("zebra", 3)] == ["d:4"]
    assert inc.search("the cat", 5) == rebuilt.search("the cat", 5)
