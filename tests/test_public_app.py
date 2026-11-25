from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.public_app import PublicApp

AUTH = ("admin", "test-password")
ORIGIN = "https://shared.example.test"
RELAY = "https://127.0.0.1:8024"


def application(tmp_path):
    settings = Settings(
        embedder="hash",
        data_dir=str(tmp_path / "corpus"),
        study_db=str(tmp_path / "study.sqlite"),
        study_task=str(Path(__file__).parents[1] / "study/live.example.json"),
        study_salt="test-salt",
    )
    return PublicApp(settings, str(tmp_path / "rag"), ORIGIN, *AUTH)


def test_shared_routes_and_protected_admin(tmp_path):
    app = application(tmp_path)
    with TestClient(app, base_url=RELAY) as first:
        second = TestClient(app, base_url=RELAY)
        assert first.get("/").status_code == 200
        assert first.get("/study").status_code == 200
        for endpoint in ("/study/admin", "/study/admin/runs", "/study/admin/export.zip"):
            assert first.get(endpoint).status_code == 401
        assert first.get("/study/admin/db", headers={"X-Admin-Token": "fake"}).status_code == 401
        assert first.get("/study/admin", auth=AUTH).status_code == 200
        assert first.get("/study/admin/materials", auth=AUTH).headers["Cache-Control"] == "no-store"
        rejected = first.post(
            "/study/admin/materials",
            auth=AUTH,
            json={},
            headers={"Origin": "https://other.example.test", "Sec-Fetch-Site": "cross-site"},
        )
        assert rejected.status_code == 403

        upload = first.post(
            "/ingest",
            files={"file": ("example.txt", b"The library opens at 9:00 on weekdays.")},
            headers={"Origin": RELAY, "Sec-Fetch-Site": "same-origin"},
        )
        assert upload.status_code == 200, upload.text
        assert len(first.get("/documents").json()) == 1
        assert second.get("/documents").json() == []
        assert second.get("/study/documents").status_code == 403


def test_publication_pins_existing_sessions_and_versions_exports(tmp_path):
    app = application(tmp_path)
    with TestClient(app, base_url=RELAY) as existing:
        newcomer = TestClient(app, base_url=RELAY)
        info = existing.get("/study/info")
        original_title = info.json()["title"]
        assert info.headers["X-Study-Version"] == "1"
        assert newcomer.get("/study/info").headers["X-Study-Version"] == "1"
        assert (
            existing.post(
                "/study/session",
                json={"consent": True},
                headers={"X-Study-Version": "1"},
            ).status_code
            == 200
        )

        current = existing.get("/study/admin/materials", auth=AUTH).json()
        fields = (
            "title",
            "info",
            "instructions",
            "task",
            "practice",
            "items",
            "documents",
            "debrief",
        )
        edit = {key: current["materials"][key] for key in fields}
        edit.update({"revision": current["revision"], "title": "Updated example"})
        response = existing.post(
            "/study/admin/materials",
            auth=AUTH,
            json=edit,
            headers={"Origin": ORIGIN},
        )
        assert response.status_code == 200, response.text
        assert existing.get("/study/info").json()["title"] == original_title
        assert (
            newcomer.post(
                "/study/session",
                json={"consent": True},
                headers={"X-Study-Version": "1"},
            ).status_code
            == 409
        )
        assert newcomer.get("/study/info").json()["title"] == "Updated example"
        versions = existing.get("/study/admin/runs", auth=AUTH).json()["runs"]
        assert [run["id"] for run in versions] == [2, 1]
        assert existing.get("/study/admin/export.zip?run=1", auth=AUTH).status_code == 200
        assert existing.get("/study/admin/export.zip?run=0", auth=AUTH).status_code == 404

    with TestClient(application(tmp_path), base_url=RELAY) as restarted:
        assert restarted.get("/study/info").json()["title"] == "Updated example"
