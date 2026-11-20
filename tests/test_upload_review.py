from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app


def test_upload_rejects_empty_unreadable_and_oversized_files(pipeline):
    app = create_app(pipeline, replace(pipeline.settings, max_upload_bytes=100))
    client = TestClient(app)
    for name, content, status in [
        ("empty.txt", b"", 422),
        ("spaces.txt", b"   \n", 422),
        ("broken.pdf", b"not a PDF", 422),
        ("broken.docx", b"not a DOCX", 422),
        ("large.txt", b"a" * 101, 413),
    ]:
        response = client.post("/ingest", files={"file": (name, content)})
        assert response.status_code == status, response.text
    assert client.get("/documents").json() == []


def test_document_view_removal_and_duplicate_at_limit(pipeline):
    client = TestClient(create_app(pipeline, replace(pipeline.settings, max_documents=1)))
    file = ("../rules.txt", b"The review meeting is on Thursday.")
    doc = client.post("/ingest", files={"file": file}).json()
    path = f"/documents/{doc['doc_id']}"
    assert client.get(path).json()["source"] == "rules.txt"
    assert "Thursday" in client.get(path).json()["text"]
    assert client.post("/ingest", files={"file": file}).json()["skipped"]
    assert (
        client.post("/ingest", files={"file": ("extra.txt", b"Another document")}).status_code
        == 413
    )
    assert client.delete(path).status_code == 200
    assert client.get(path).status_code == 404
    assert client.post("/search", json={"query": "Thursday"}).json()["results"] == []


@pytest.mark.parametrize("query", ["   ", "a" * 4001])
def test_query_limits(pipeline, query):
    client = TestClient(create_app(pipeline))
    assert client.post("/ask", json={"query": query}).status_code == 422


def test_visitors_cannot_read_search_or_delete_each_others_documents(tmp_path):
    settings = Settings(data_dir=str(tmp_path), embedder="hash", isolate_visitors=True)
    app = create_app(settings=settings)
    first, second = TestClient(app), TestClient(app)
    upload = first.post("/ingest", files={"file": ("private.txt", b"The secret code is WALNUT.")})
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["doc_id"]
    assert "HttpOnly" in upload.headers["set-cookie"]
    assert second.get("/documents").json() == []
    assert second.get(f"/documents/{doc_id}").status_code == 404
    assert second.delete(f"/documents/{doc_id}").status_code == 404
    query = {"query": "WALNUT", "doc_ids": [doc_id]}
    assert second.post("/search", json=query).json()["results"] == []
    assert first.post("/search", json=query).json()["results"]
    second.post("/ingest", files={"file": ("different-name.txt", b"The secret code is WALNUT.")})
    assert second.get("/documents").json()[0]["source"] == "different-name.txt"
    assert first.get("/documents").json()[0]["source"] == "private.txt"
    resumed = TestClient(create_app(settings=settings))
    resumed.cookies.update(first.cookies)
    assert resumed.get("/documents").json()[0]["source"] == "private.txt"
    assert resumed.delete(f"/documents/{doc_id}").status_code == 200
    assert second.get(f"/documents/{doc_id}").status_code == 200


def test_isolated_upload_rejects_cross_origin_form(tmp_path):
    client = TestClient(
        create_app(
            settings=Settings(
                data_dir=str(tmp_path),
                embedder="hash",
                isolate_visitors=True,
            )
        )
    )
    response = client.post(
        "/ingest",
        headers={"origin": "https://another.example"},
        files={"file": ("unexpected.txt", b"Unsolicited upload")},
    )
    assert response.status_code == 403
