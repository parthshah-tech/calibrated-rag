import pytest

from backend.chunking import chunk_text, fixed_chunks, recursive_chunks

TEXT = ("Alpha beta gamma. " * 12 + "\n\n") * 6 + "Tail without newline."


@pytest.mark.parametrize("fn", [fixed_chunks, recursive_chunks])
def test_spans_match_source_and_respect_size(fn):
    for c in fn(TEXT, 100, 20):
        assert TEXT[c.start : c.end] == c.text
        assert len(c.text) <= 100


@pytest.mark.parametrize("fn", [fixed_chunks, recursive_chunks])
def test_every_character_is_covered(fn):
    covered = set()
    for c in fn(TEXT, 100, 20):
        covered.update(range(c.start, c.end))
    uncovered = [i for i in range(len(TEXT)) if i not in covered and not TEXT[i].isspace()]
    assert uncovered == []


def test_fixed_overlap_is_exact():
    chunks = fixed_chunks("x" * 250, 100, 25)
    assert [(c.start, c.end) for c in chunks] == [(0, 100), (75, 175), (150, 250)]


def test_recursive_prefers_paragraph_boundaries():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = recursive_chunks(text, size=30, overlap=0)
    assert [c.text.strip() for c in chunks] == [
        "First paragraph here.",
        "Second paragraph here.",
        "Third paragraph here.",
    ]


def test_recursive_hard_splits_unbreakable_text():
    chunks = recursive_chunks("a" * 250, 100, 0)
    assert [len(c.text) for c in chunks] == [100, 100, 50]


def test_recursive_overlap_carries_trailing_context():
    text = "one two three four five six seven eight nine ten " * 6
    chunks = recursive_chunks(text, size=60, overlap=20)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start < prev.end  # overlapping
        assert nxt.start >= prev.start


def test_empty_and_whitespace_input():
    assert fixed_chunks("", 100, 10) == []
    assert recursive_chunks("   \n\n  ", 100, 10) == []


def test_invalid_parameters():
    with pytest.raises(ValueError):
        fixed_chunks("abc", 10, 10)
    with pytest.raises(ValueError):
        recursive_chunks("abc", 0, 0)


def test_strategy_dispatch():
    assert chunk_text("hello world", "fixed", 5, 0)
    with pytest.raises(NotImplementedError):
        chunk_text("x", "semantic")
    with pytest.raises(ValueError):
        chunk_text("x", "nope")
