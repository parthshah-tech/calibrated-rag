from backend.history_aware import clean_query, rewrite_query, trim_history
from backend.llm import FakeLLM, LLMError
from backend.query_expansion import expand_query, parse_queries

HIST = [
    {"role": "user", "content": "What does the Pro plan cost?"},
    {"role": "assistant", "content": "The Pro plan costs 20 dollars a month."},
]


def test_no_history_means_no_model_call():
    llm = FakeLLM()
    r = rewrite_query(llm, "what about the second one?", [])
    assert r.query == "what about the second one?" and not r.rewritten and llm.calls == []


def test_follow_up_is_rewritten_using_the_conversation():
    llm = FakeLLM(["Query: Pro plan annual pricing"])
    r = rewrite_query(llm, "and per year?", HIST)
    assert r.query == "Pro plan annual pricing" and r.rewritten
    prompt = llm.calls[0][-1]["content"]
    assert (
        "User: What does the Pro plan cost?" in prompt
        and "Latest question: and per year?" in prompt
    )


def test_unchanged_query_is_not_flagged_as_rewritten():
    r = rewrite_query(
        FakeLLM(['"What does the Pro plan cost?"']), "what does the pro plan cost?", HIST
    )
    assert not r.rewritten


def test_model_failure_falls_back_to_the_typed_question():
    r = rewrite_query(FakeLLM(error=LLMError("HTTP 429")), "and per year?", HIST)
    assert r.query == "and per year?" and not r.rewritten and "429" in r.error


def test_clean_query_takes_the_first_line_and_strips_decoration():
    assert clean_query('\n  "pricing of Pro"\nexplanation', "x") == "pricing of Pro"
    assert clean_query("   ", "fallback") == "fallback"
    assert len(clean_query("a" * 500, "x")) == 300


def test_history_is_trimmed_and_malformed_turns_dropped():
    long = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    assert [t["content"] for t in trim_history(long)] == [f"q{i}" for i in range(4, 10)]
    messy = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "  "},
        {"role": "user"},
    ]
    assert trim_history(messy) == [] and trim_history(None) == []
    assert len(trim_history([{"role": "user", "content": "z" * 5000}])[0]["content"]) == 600


def test_parse_queries_cleans_dedupes_and_caps():
    text = (
        "1. sourdough starter feeding\n- 'Sourdough Starter Feeding'\n\n* bread flour ratio\n"
        "2) Original Question\nextra one\nextra two"
    )
    got = parse_queries(text, "original question", 3)
    assert got == ["sourdough starter feeding", "bread flour ratio", "extra one"]


def test_expand_query_calls_once_and_survives_failure():
    llm = FakeLLM(["alpha beta\ngamma delta"])
    ex = expand_query(llm, "question", 3)
    assert ex.queries == ["alpha beta", "gamma delta"] and len(llm.calls) == 1
    assert "3 different queries" in llm.calls[0][0]["content"]
    bad = expand_query(FakeLLM(error=LLMError("down")), "question", 3)
    assert bad.queries == [] and bad.error == "down"
