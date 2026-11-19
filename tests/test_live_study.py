import io
import json
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.llm import FakeLLM
from backend.main import create_app
from backend.study.live_materials import load_materials


def uid():
    return uuid.uuid4().hex


@pytest.fixture
def live(pipeline, tmp_path):
    material = load_materials(Path(__file__).parents[1] / "study/live.example.json")
    data = material.model_dump()
    data["evaluations"] = [
        {
            "id": "e1",
            "query": "When is the deadline?",
            "answer": "30 April [1].",
            "score": 0.45,
            "passages": [{"name": "rules.md", "text": "Apply by 30 April."}],
            "answer_key": "PRIVATE KEY",
            "ai_correct": True,
        }
    ]
    path = tmp_path / "materials.json"
    path.write_text(json.dumps(data))
    settings = replace(
        pipeline.settings,
        study_mode=True,
        study_flow="live",
        study_task=str(path),
        study_db=str(tmp_path / "study.sqlite"),
        study_salt="test-salt",
        study_admin_token="test-admin",
        study_time_budget_min=1,
    )
    pipeline.llm = FakeLLM(["The deadline is 30 April [1]."])
    app = create_app(pipeline, settings)
    return app, settings, pipeline


def start(client):
    assert client.get("/study/info").status_code == 200
    response = client.post("/study/session", json={"consent": True})
    assert response.status_code == 200, response.text
    return response.json()


def submit(client, phase, **extra):
    return client.post("/study/submit", json={"request_id": uid(), "phase": phase, **extra})


def enter_task(client):
    start(client)
    assert submit(client, "instructions").json()["phase"] == "practice"
    result = submit(client, "practice", answers={"practice": {"text": "7.2", "confidence": 6}})
    assert result.status_code == 200, result.text
    return result.json()


def test_consent_required_and_regular_endpoints_closed(live):
    app, _, _ = live
    client = TestClient(app)
    assert client.get("/study").status_code == 200
    assert client.get("/app.css").status_code == 200
    assert client.get("/study/app.js").status_code == 200
    assert client.get("/study/session").json() == {"phase": "consent"}
    assert not app.state.live_store.participants()
    assert client.get("/study/documents").status_code == 403
    assert (
        client.post(
            "/study/ask", json={"request_id": uid(), "phase": "task", "query": "hi"}
        ).status_code
        == 403
    )
    assert client.post("/study/session", json={"consent": False}).status_code == 422
    assert client.get("/documents").status_code == 403
    assert client.post("/ingest", files={"file": ("x.txt", b"x")}).status_code == 403
    assert client.post("/ask", json={"query": "hi"}).status_code == 403


def test_balanced_assignment_hidden_scores_and_idempotent_asks(live):
    app, _, pipeline = live
    first, second = TestClient(app), TestClient(app)
    for client in (first, second):
        enter_task(client)
        payload = {"request_id": uid(), "phase": "task", "query": "When is the deadline?"}
        result = client.post("/study/ask", json=payload)
        assert result.status_code == 200, result.text
        call_count = len(pipeline.llm.calls)
        assert client.post("/study/ask", json=payload).json() == result.json()
        assert len(pipeline.llm.calls) == call_count
        session = client.get("/study/session").json()
        assert ("score" in result.json()) == ("confidence_note" in session)
        assert "PRIVATE KEY" not in json.dumps(session)
        if "confidence_note" not in session:
            assert '"score"' not in json.dumps(session)
            assert '"confidence"' not in json.dumps(result.json())
    assert {p["condition"] for p in app.state.live_store.participants()} == {"baseline", "badge"}
    assert len([e for e in app.state.live_store.events() if e["type"] == "live_query"]) == 2


def test_complete_without_asking_and_blind_export(live):
    app, _, _ = live
    client = TestClient(app)
    task = enter_task(client)
    answers = {item["id"]: {"text": "My conclusion", "confidence": 5} for item in task["items"]}
    request = {"phase": "task", "request_id": uid(), "answers": answers}
    evaluation = client.post("/study/submit", json=request).json()
    assert evaluation["phase"] == "evaluation"
    assert client.post("/study/submit", json=request).json()["phase"] == "evaluation"
    assert (
        client.post(
            "/study/ask", json={"query": "x", "phase": "task", "request_id": uid()}
        ).status_code
        == 409
    )
    rating = {"item_id": "e1", "judgment": "correct", "confidence": 6, "request_id": uid()}
    survey = client.post("/study/judge", json=rating).json()
    assert survey["phase"] == "survey"
    assert client.post("/study/judge", json=rating).json()["phase"] == "survey"
    values = {item["id"]: None for item in survey["survey"]}
    assert submit(client, "survey", survey=values).json()["phase"] == "interview"
    assert submit(client, "interview").json()["phase"] == "complete"
    assert client.get("/study/session").json()["phase"] == "complete"
    assert client.get("/study/admin/export.zip").status_code == 403
    archive = client.get("/study/admin/export.zip", headers={"x-admin-token": "test-admin"})
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        blind = z.read("human_answers_blind.csv").decode()
        columns = blind.splitlines()[0].split(",")
        assert "condition" not in columns and "score" not in columns
        assert len(blind.splitlines()) == 8  # header, practice, six scored decisions
        assert "n_main_queries" in z.read("sessions.csv").decode()
        assert (
            client.get("/study/session").json()["participant_id"] in z.read("sessions.csv").decode()
        )
        assert "live_query" not in z.read("events.jsonl").decode()
        assert client.cookies.get("rag_study") not in z.read("events.jsonl").decode()


def test_drafts_resume_and_session_access(live):
    app, settings, pipeline = live
    first, other = TestClient(app), TestClient(app)
    enter_task(first)
    data = {"phase": "task", "answers": {"eligibility": {"text": "Unfinished", "confidence": 4}}}
    assert first.post("/study/draft", json=data).status_code == 200
    resumed = TestClient(create_app(pipeline, settings))
    resumed.cookies.update(first.cookies)
    assert resumed.get("/study/session").json()["draft"]["eligibility"]["text"] == "Unfinished"
    assert other.get("/study/session").json()["phase"] == "consent"
    assert submit(first, "task", answers={}).status_code == 422
    assert first.post("/study/withdraw", json={}).json()["phase"] == "withdrawn"
    assert first.post("/study/draft", json=data).status_code == 409
    assert first.get("/study/documents").status_code == 409


def test_expired_task_allows_partial_submission_and_blocks_asks(live, monkeypatch):
    app, _, _ = live
    client = TestClient(app)
    state = enter_task(client)
    monkeypatch.setattr("backend.study.live_api.time.time", lambda: state["deadline"] + 1)
    assert (
        client.post(
            "/study/ask", json={"query": "x", "phase": "task", "request_id": uid()}
        ).status_code
        == 409
    )
    assert submit(client, "task", answers={}).json()["phase"] == "evaluation"
    event = [e for e in app.state.live_store.events() if e["type"] == "live_phase"][-1]
    assert event["payload"]["expired"] is True


def test_document_view_events_and_config_change(live, monkeypatch):
    app, settings, pipeline = live
    client = TestClient(app)
    enter_task(client)
    monkeypatch.setattr(
        pipeline, "ingest", lambda *_: pytest.fail("Reader must not index documents")
    )
    docs = client.get("/study/documents").json()
    payload = {"request_id": uid(), "kind": "document", "target": docs[0]["doc_id"]}
    expected = load_materials(settings.study_task).documents[0].model_dump()
    assert client.post("/study/view", json=payload).json()["passages"] == [expected]
    assert client.post("/study/view", json=payload).status_code == 200
    for _ in range(2):
        assert (
            client.post("/study/view/close", json={"request_id": payload["request_id"]}).status_code
            == 200
        )
    events = app.state.live_store.events()
    assert sum(e["type"] == "live_view_opened" for e in events) == 1
    assert sum(e["type"] == "live_view_closed" for e in events) == 1
    with pytest.raises(ValueError, match="configuration changed"):
        create_app(pipeline, replace(settings, confidence_metric="overlap"))
