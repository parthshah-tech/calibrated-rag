from conftest import DOC_A, DOC_B
from fastapi.testclient import TestClient

from backend.main import create_app


def client(pipeline):
    return TestClient(create_app(pipeline))


def test_health(pipeline):
    assert client(pipeline).get("/health").json() == {"status": "ok", "llm_configured": False}


def test_ingest_then_search_and_idempotent_upload(pipeline):
    c = client(pipeline)
    r1 = c.post("/ingest", files={"file": ("a.txt", DOC_A.encode())}).json()
    r2 = c.post("/ingest", files={"file": ("a.txt", DOC_A.encode())}).json()
    c.post("/ingest", files={"file": ("b.txt", DOC_B.encode())})
    assert r1["skipped"] is False and r2["skipped"] is True
    out = c.post("/search", json={"query": "reciprocal rank fusion", "k": 2}).json()
    assert out["results"][0]["source"] == "a.txt"
    assert out["confidence"]["bucket"] in {"Low", "Medium", "High"}
    assert out["trace"]["total_ms"] >= 0


def test_rejects_bad_input(pipeline):
    c = client(pipeline)
    assert c.post("/ingest", files={"file": ("x.exe", b"zz")}).status_code == 415
    assert c.post("/search", json={"query": "x", "mode": "nope"}).status_code == 422
    assert c.post("/search", json={"query": ""}).status_code == 422


def test_root_serves_the_search_page(pipeline):
    r = client(pipeline).get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Ask your documents" in r.text and "/ask" in r.text


def test_ask_returns_sources_and_a_cited_answer(loaded):
    from backend.llm import FakeLLM

    loaded.llm = FakeLLM(["RRF fuses rankings [1]."])
    c = client(loaded)
    c.post("/ingest", files={"file": ("a.txt", DOC_A.encode())})
    out = c.post("/ask", json={"query": "reciprocal rank fusion", "k": 3}).json()
    assert out["answer"]["cited"] == [1] and out["answer_error"] is None
    assert out["results"] and out["confidence"]["bucket"] in {"Low", "Medium", "High"}
    assert c.get("/health").json()["llm_configured"] is True


def test_ask_without_llm_degrades_to_sources(loaded):
    out = client(loaded).post("/ask", json={"query": "fusion"}).json()
    assert out["answer"] is None and "LLM_API_KEY" in out["answer_error"] and out["results"]


def test_ask_survives_a_failing_llm_and_validates_input(loaded):
    from backend.llm import FakeLLM, LLMError

    loaded.llm = FakeLLM(error=LLMError("HTTP 401 bad key"))
    c = client(loaded)
    out = c.post("/ask", json={"query": "fusion"}).json()
    assert out["answer"] is None and "401" in out["answer_error"] and out["results"]
    assert c.post("/ask", json={"query": "x", "mode": "nope"}).status_code == 422


def test_ask_accepts_history_and_expand(loaded):
    from backend.llm import FakeLLM

    loaded.llm = FakeLLM(
        ["XJ-4471 product code", "sourdough bread flour", "It means the pump failed [1]."]
    )
    hist = [
        {"role": "user", "content": "error codes?"},
        {"role": "assistant", "content": "XJ-4471"},
    ]
    out = (
        client(loaded)
        .post("/ask", json={"query": "and that one?", "history": hist, "expand": 1})
        .json()
    )
    assert out["rewritten_query"] == "XJ-4471 product code"
    assert out["expansions"] == ["sourdough bread flour"] and out["notes"] == []
    assert out["answer"]["cited"] == [1]


def test_ask_validates_history_and_expand(loaded):
    c = client(loaded)
    bad_role = {"query": "x", "history": [{"role": "system", "content": "hi"}]}
    assert c.post("/ask", json=bad_role).status_code == 422
    assert c.post("/ask", json={"query": "x", "expand": 9}).status_code == 422
    too_long = {"query": "x", "history": [{"role": "user", "content": "a" * 5000}]}
    assert c.post("/ask", json=too_long).status_code == 422


def test_search_and_ask_accept_rerank(loaded):
    from backend.llm import FakeLLM
    from backend.reranker import LexicalOverlapReranker

    loaded.reranker = LexicalOverlapReranker()
    loaded.llm = FakeLLM(["Answer [1]."])
    c = client(loaded)
    s = c.post("/search", json={"query": "XJ-4471 code", "rerank": True}).json()
    assert s["reranked"] is True and s["trace"]["counters"]["rerank_pairs"] > 0
    a = c.post("/ask", json={"query": "XJ-4471 code", "rerank": True}).json()
    assert a["reranked"] is True and a["answer"]["cited"] == [1]
    assert c.post("/search", json={"query": "XJ-4471 code"}).json()["reranked"] is False
