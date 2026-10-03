from conftest import DOC_A, DOC_B
from fastapi.testclient import TestClient

from backend.main import create_app


def client(pipeline):
    return TestClient(create_app(pipeline))


def test_health(pipeline):
    assert client(pipeline).get("/health").json() == {"status": "ok"}


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
    assert "Ask your documents" in r.text and "/search" in r.text
