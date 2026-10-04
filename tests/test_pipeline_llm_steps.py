from conftest import DOC_B, DOC_C

from backend.llm import FakeLLM, LLMError

HIST = [
    {"role": "user", "content": "Tell me about error codes"},
    {"role": "assistant", "content": "Codes like XJ-4471 mean the pump failed."},
]


def top(res):
    return res.chunks[res.fused[0].chunk_id].source


def test_rewrite_changes_what_is_searched(loaded):
    loaded.llm = FakeLLM(["XJ-4471 product code"])
    res = loaded.retrieve("what about that one?", 3, history=HIST)
    assert res.rewritten_query == "XJ-4471 product code" and top(res) == "b.txt"
    assert res.query == "what about that one?"  # the user's words are preserved
    assert res.trace["counters"]["llm_calls"] == 1
    assert any(s["name"] == "rewrite" for s in res.trace["spans"])


def test_rewrite_failure_is_a_note_not_an_error(loaded):
    loaded.llm = FakeLLM(error=LLMError("HTTP 429"))
    res = loaded.retrieve("fusion", 3, history=HIST)
    assert res.fused and res.rewritten_query is None
    assert any("rewrite skipped" in n and "429" in n for n in res.notes)


def test_expansion_finds_what_the_original_wording_misses(loaded):
    plain = loaded.retrieve("yummy loaf recipe", 1)
    assert top(plain) != "c.txt"  # no shared words with the bread document
    loaded.llm = FakeLLM(["sourdough bread flour starter\nbake hot oven"])
    res = loaded.retrieve("yummy loaf recipe", 3, expand=2)
    assert res.expansions == ["sourdough bread flour starter", "bake hot oven"]
    assert "c.txt" in [res.chunks[h.chunk_id].source for h in res.fused]
    names = {s["name"] for s in res.trace["spans"]}
    assert {"expand", "dense#1", "bm25#2"} <= names


def test_confidence_ignores_the_expansions(loaded):
    base = loaded.retrieve("yummy loaf recipe", 3).confidence
    loaded.llm = FakeLLM(["sourdough bread flour starter"])
    expanded = loaded.retrieve("yummy loaf recipe", 3, expand=1).confidence
    assert expanded == base  # the signal measures the question, not the paraphrases


def test_expansion_is_hybrid_only_and_needs_an_llm(loaded):
    res = loaded.retrieve("fusion", 3, expand=3)  # no LLM configured
    assert res.expansions == [] and res.notes == []
    loaded.llm = FakeLLM(["other words"])
    res = loaded.retrieve("fusion", 3, mode="dense", expand=3)
    assert res.expansions == [] and loaded.llm.calls == []


def test_expansion_failure_degrades_to_plain_retrieval(loaded):
    plain = [h.chunk_id for h in loaded.retrieve("reciprocal rank fusion", 3).fused]
    loaded.llm = FakeLLM(error=LLMError("down"))
    res = loaded.retrieve("reciprocal rank fusion", 3, expand=2)
    assert [h.chunk_id for h in res.fused] == plain
    assert any("expansion skipped" in n for n in res.notes)


def test_answer_passes_history_to_generation_and_counts_calls(loaded):
    loaded.llm = FakeLLM(["XJ-4471 product code", "It means the pump failed [1]."])
    res, ans, err = loaded.answer("what about that one?", 3, history=HIST)
    assert err is None and ans.cited == [1]
    gen_messages = loaded.llm.calls[1]
    assert any("XJ-4471 mean the pump failed" in m["content"] for m in gen_messages)
    assert res.trace["counters"]["llm_calls"] == 2


def test_docs_sanity():
    assert "XJ-4471" in DOC_B and "Sourdough" in DOC_C
