from backend.tracing import Tracer


def test_spans_and_counters():
    t = Tracer()
    with t.span("work", extra=1):
        pass
    t.count("llm_calls")
    t.count("llm_calls", 2)
    d = t.to_dict()
    assert d["spans"][0]["name"] == "work" and d["spans"][0]["extra"] == 1
    assert d["counters"]["llm_calls"] == 3
    assert d["total_ms"] >= d["spans"][0]["ms"]
