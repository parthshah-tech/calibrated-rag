import csv
import json

import pytest
from conftest import DOC_A, DOC_B
from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.config import Settings
from backend.llm import FakeLLM
from backend.main import create_app
from backend.study import export as ex
from backend.study.questions import load_questions, ordered_questions, public_view, questions_hash
from backend.study.stimuli import FrozenStimuli, LiveStimuli, freeze
from backend.study.store import CONDITIONS, StudyStore
from backend.study.survey import DEFAULT_ITEMS, items_for, validate
from backend.study.viewer import reconstruct

QUESTIONS = [
    {
        "id": "p0",
        "question": "what is bm25",
        "source_file": "b.txt",
        "answerable": True,
        "gold_phrase": "sparse retrieval function",
        "practice": True,
    },
    {
        "id": "q1",
        "question": "how does reciprocal rank fusion work",
        "source_file": "a.txt",
        "gold_phrase": "Each document earns 1 over k",
        "answerable": True,
        "coverage": "well-covered",
        "answer": "It sums 1 over k plus rank.",
    },
    {
        "id": "q2",
        "question": "who won the world cup in 2010",
        "source_file": "",
        "gold_phrase": "",
        "answerable": False,
        "attention_check": True,
    },
    {
        "id": "q3",
        "question": "what does sourdough bread need",
        "source_file": "c.txt",
        "gold_phrase": "flour, water, salt",
        "answerable": True,
        "answer": "Flour, water, salt, starter.",
    },
]
TOKEN = "s3cret-token"


def make_settings(tmp_path, **kw):
    q = tmp_path / "questions.json"
    q.write_text(json.dumps(QUESTIONS))
    base = dict(
        study_mode=True,
        study_questions=str(q),
        study_db=str(tmp_path / "s.sqlite"),
        study_admin_token=TOKEN,
        study_salt="salt",
    )
    return Settings(**{**base, **kw})


@pytest.fixture
def study(tmp_path, loaded):
    app = create_app(loaded, make_settings(tmp_path))
    return TestClient(app), app.state.study_store, loaded, tmp_path


def start(c, store, code, condition):
    store.assign(code, [condition], "h")  # pin the condition for the test
    r = c.post("/study/start", json={"code": code})
    assert r.status_code == 200
    return r.json()


def post_event(c, code, type_, qid=None, **payload):
    return c.post(
        "/study/event", json={"code": code, "question_id": qid, "type": type_, "payload": payload}
    )


# ------------------------------------------------------------- store, questions, order
def test_assignment_is_balanced_idempotent_and_deterministic():
    st = StudyStore(":memory:")
    got = [st.assign(f"P{i:02d}", CONDITIONS, "h")[0] for i in range(9)]
    assert {c: got.count(c) for c in CONDITIONS} == {c: 3 for c in CONDITIONS}
    assert st.assign("P00", CONDITIONS, "h") == (got[0], False)
    other = StudyStore(":memory:")
    assert [other.assign(f"P{i:02d}", CONDITIONS, "h")[0] for i in range(9)] == got


def test_event_log_is_ordered_append_only_and_snapshots():
    st = StudyStore(":memory:")
    st.assign("P1", CONDITIONS, "h")
    a = st.log("P1", "q1", "question_shown", {}, ts=1.0)
    b = st.log("P1", "q1", "stated_confidence", {"value": 5}, ts=2.0)
    assert [e["id"] for e in st.events("P1")] == [a, b] and st.events("P1", "q2") == []
    assert not any(hasattr(st, n) for n in ("delete", "update", "clear", "remove"))
    assert st.serialize().startswith(b"SQLite format 3")


def test_questions_validate_and_public_view_hides_ground_truth(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(QUESTIONS))
    qs = load_questions(p)
    view = json.dumps(public_view(qs))
    assert "gold_phrase" not in view and "attention_check" not in view and "answer" not in view
    assert public_view(qs)[0]["practice"] is True and public_view(qs)[1]["practice"] is False
    assert questions_hash(qs) == questions_hash(load_questions(p))
    for bad in ([], [{"id": "x"}], [QUESTIONS[0], QUESTIONS[0]]):
        p.write_text(json.dumps(bad))
        with pytest.raises((ValueError, RuntimeError)):
            load_questions(p)
    with pytest.raises(RuntimeError, match="not found"):
        load_questions(tmp_path / "missing.json")


def test_order_is_seeded_practice_first_and_attention_checks_never_at_the_edges():
    many = [
        {
            "id": "p",
            "question": "p",
            "source_file": "",
            "gold_phrase": "",
            "answerable": True,
            "practice": True,
            "attention_check": False,
            "coverage": "",
        }
    ]
    for i in range(8):
        many.append(
            {
                "id": f"q{i}",
                "question": "x",
                "source_file": "",
                "gold_phrase": "",
                "answerable": True,
                "practice": False,
                "coverage": "",
                "attention_check": i in (0, 1),
            }
        )
    first = [q["id"] for q in ordered_questions("A1", many)]
    assert first == [q["id"] for q in ordered_questions("A1", many)]  # reproducible
    assert first[0] == "p" and sorted(first) == sorted(q["id"] for q in many)
    assert first != [q["id"] for q in ordered_questions("B2", many)]  # differs across people
    for n in range(200):
        order = ordered_questions(f"C{n}", many)[1:]
        assert not order[0]["attention_check"] and not order[-1]["attention_check"]


def test_survey_items_depend_on_condition_and_are_validated():
    ids = {c: {i["id"] for i in items_for(c, DEFAULT_ITEMS)} for c in CONDITIONS}
    assert "badge_1" not in ids["baseline"] and "badge_1" in ids["badge"]
    assert "reform_1" in ids["badge_reformulation"] and "reform_1" not in ids["badge"]
    good = {
        i["id"]: (4 if i["type"] == "scale" else i["options"][0])
        for i in items_for("baseline", DEFAULT_ITEMS)
    }
    validate(good, "baseline", DEFAULT_ITEMS)
    for bad in (
        {**good, "trust_1": 9},
        {**good, "extra": 1},
        {**good, "manip_check": "Maybe"},
        {k: v for k, v in good.items() if k != "trust_1"},
        {**good, "trust_1": True},
    ):
        with pytest.raises(ValueError):
            validate(bad, "baseline", DEFAULT_ITEMS)


# ------------------------------------------------------------- start and resume
def test_start_hides_the_condition_orders_questions_and_tailors_text(study):
    c, store, _, _ = study
    r = start(c, store, "P1", "baseline")
    blob = json.dumps(r)
    assert "condition" not in blob and "gold_phrase" not in blob and "attention" not in blob
    assert r["questions"][0]["id"] == "p0" and len(r["questions"]) == 4
    assert r["texts"]["badge_note"] == "" and r["resumed"] is True
    assert all(i["id"] != "badge_1" for i in r["survey"])
    b = start(c, store, "P2", "badge")
    assert "confidence indicator" in b["texts"]["badge_note"]
    assert any(i["id"] == "badge_1" for i in b["survey"])
    assert c.post("/study/start", json={"code": "bad code!"}).status_code == 422


def test_info_is_readable_before_consent_without_creating_a_participant(study):
    c, store, _, _ = study
    r = c.get("/study/info").json()
    assert "voluntary" in r["info"] and "agree to take part" in r["consent"]
    assert store.participants() == []


def test_progress_is_reported_so_a_refresh_resumes(study):
    c, store, _, _ = study
    start(c, store, "P3", "baseline")
    t = post_event(c, "P3", "phase_started").json()["ts"]
    post_event(c, "P3", "final_answer_submitted", "p0")
    again = c.post("/study/start", json={"code": "P3"}).json()
    assert again["phase_started_at"] == t and again["submitted"] == ["p0"]
    assert again["survey_done"] is False


# ------------------------------------------------------------- ask: conditions enforced
def test_baseline_never_receives_the_signal_but_it_is_logged(study):
    c, store, pipe, _ = study
    pipe.llm = FakeLLM(["Fusion combines rankings [1]."])
    start(c, store, "P1", "baseline")
    out = c.post("/study/ask", json={"code": "P1", "question_id": "q1"}).json()
    assert out["answer"]["cited"] == [1] and out["passages"]
    assert "confidence" not in out and "reformulations" not in out
    shown = [e for e in store.events("P1", "q1") if e["type"] == "answer_shown"][0]["payload"]
    assert shown["confidence"]["bucket"] in {"Low", "Medium", "High"} and not shown["shown_badge"]


def test_badge_condition_sees_confidence_but_no_reformulations(study):
    c, store, pipe, _ = study
    pipe.llm = FakeLLM(["Answer [1]."])
    start(c, store, "P2", "badge")
    out = c.post("/study/ask", json={"code": "P2", "question_id": "q1"}).json()
    assert out["confidence"]["bucket"] in {"Low", "Medium", "High"}
    assert "reformulations" not in out


def test_reformulations_only_in_condition_three_and_only_askable_if_shown(study):
    c, store, pipe, _ = study
    pipe.settings = Settings(
        **{**pipe.settings.__dict__, "confidence_low": 0.99, "confidence_high": 1.0}
    )
    pipe.llm = FakeLLM(
        ["Answer [1].", "reciprocal rank fusion rankings from several retrievers", "Answer [1]."]
    )
    start(c, store, "P3", "badge_reformulation")
    out = c.post("/study/ask", json={"code": "P3", "question_id": "q1"}).json()
    sug = out.get("reformulations", [])
    assert all({"text", "bucket", "score"} <= set(s) for s in sug) and len(sug) <= 3
    logged = [e for e in store.events("P3", "q1") if e["type"] == "reformulation_shown"]
    assert bool(sug) == bool(logged)
    if sug:
        ok = c.post("/study/ask", json={"code": "P3", "question_id": "q1", "text": sug[0]["text"]})
        assert ok.status_code == 200 and ok.json()["via_reformulation"] is True
        click = post_event(c, "P3", "reformulation_clicked", "q1", text=sug[0]["text"])
        assert click.status_code == 200
    bad = c.post("/study/ask", json={"code": "P3", "question_id": "q1", "text": "something else"})
    assert bad.status_code == 403


def test_free_text_is_refused_outside_condition_three(study):
    c, store, _, _ = study
    for code, cond in (("P4", "baseline"), ("P5", "badge")):
        start(c, store, code, cond)
        r = c.post("/study/ask", json={"code": code, "question_id": "q1", "text": "anything"})
        assert r.status_code == 403


def test_general_endpoints_are_closed_in_study_mode(study):
    c, _, _, _ = study
    assert c.post("/ask", json={"query": "x"}).status_code == 403
    assert c.post("/search", json={"query": "x"}).status_code == 403
    assert c.post("/ingest", files={"file": ("a.txt", b"x")}).status_code == 403
    assert c.get("/", follow_redirects=False).headers["location"] == "/study"


# ------------------------------------------------------------- events
def test_events_are_validated(study):
    c, store, _, _ = study
    start(c, store, "P6", "badge")
    for bad in (0, 8, "5", True, None):
        assert post_event(c, "P6", "stated_confidence", "q1", value=bad).status_code == 422
    assert post_event(c, "P6", "stated_confidence", "q1", value=5).status_code == 200
    assert post_event(c, "P6", "nonsense", "q1").status_code == 422
    assert post_event(c, "P6", "question_shown", "nope").status_code == 404
    assert post_event(c, "P6", "consent_given").status_code == 200  # no question needed
    assert post_event(c, "P6", "reformulation_clicked", "q1", text="never").status_code == 403
    assert post_event(c, "P6", "doc_viewer_opened", "q1", x="y" * 5000).status_code == 413
    assert c.post("/study/event", json={"code": "ghost", "type": "time_expired"}).status_code == 404


def test_final_answer_requires_accept_flag_and_text_when_overriding(study):
    c, store, _, _ = study
    start(c, store, "P7", "baseline")
    assert post_event(c, "P7", "final_answer", "q1", accepted="yes").status_code == 422
    assert post_event(c, "P7", "final_answer", "q1", accepted=False, text="  ").status_code == 422
    long = post_event(c, "P7", "final_answer", "q1", accepted=False, text="x" * 1001)
    assert long.status_code == 413
    ok = post_event(c, "P7", "final_answer", "q1", accepted=True, text="ignored")
    assert ok.status_code == 200
    stored = [e for e in store.events("P7") if e["type"] == "final_answer"][0]["payload"]
    assert stored == {"accepted": True, "text": ""}


def test_survey_must_match_the_condition(study):
    c, store, _, _ = study
    start(c, store, "P8", "baseline")
    items = items_for("baseline", DEFAULT_ITEMS)
    good = {i["id"]: (4 if i["type"] == "scale" else i["options"][0]) for i in items}
    assert post_event(c, "P8", "survey", values={**good, "badge_1": 4}).status_code == 422
    assert post_event(c, "P8", "survey", values=good).status_code == 200
    assert c.post("/study/start", json={"code": "P8"}).json()["survey_done"] is True


def test_document_viewer_and_config_hash(study):
    c, _, pipe, _ = study
    assert reconstruct([ch for ch in pipe.store.all_chunks() if ch.source == "a.txt"]) == DOC_A
    docs = c.get("/study/documents").json()
    a = next(d for d in docs if d["source"] == "a.txt")
    assert c.get(f"/study/documents/{a['doc_id']}").json()["text"] == DOC_A
    assert c.get("/study/documents/nope").status_code == 404
    h1 = c.get("/study/config").json()["config_hash"]
    assert c.get("/study/config").json()["config_hash"] == h1
    pipe.ingest("extra.txt", b"A new document that changes the corpus.")
    assert c.get("/study/config").json()["config_hash"] != h1


# ------------------------------------------------------------- frozen replay
def test_frozen_replay_matches_live_and_runs_without_any_pipeline(tmp_path, loaded, monkeypatch):
    settings = make_settings(tmp_path)
    qs = load_questions(settings.study_questions)
    loaded.llm = FakeLLM(["Fusion combines rankings [1]."])
    data = freeze(settings, lambda: loaded, qs)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(data))
    live = LiveStimuli(lambda: loaded, settings)
    frozen = FrozenStimuli(path, qs)
    for q in qs:
        saved = frozen.ask(q["id"], q["question"], True)["answer"]
        assert saved == live.ask(q["id"], q["question"], True)["answer"]
    with pytest.raises(KeyError):
        frozen.ask("q1", "a wording that was never saved")

    def boom(*a, **k):
        raise AssertionError("the study server must not build a pipeline in frozen mode")

    monkeypatch.setattr(main_mod, "build_pipeline", boom)
    s2 = make_settings(tmp_path, study_frozen=str(path), study_db=str(tmp_path / "f.sqlite"))
    app = create_app(None, s2)
    c, store = TestClient(app), app.state.study_store
    start(c, store, "F1", "badge")
    out = c.post("/study/ask", json={"code": "F1", "question_id": "q1"}).json()
    assert out["answer"]["text"] == "Fusion combines rankings [1]." and "confidence" in out
    docs = c.get("/study/documents").json()
    assert {d["source"] for d in docs} >= {"a.txt", "b.txt"}
    assert c.get(f"/study/documents/{docs[0]['doc_id']}").json()["text"]
    assert c.get("/study/config").json()["frozen"] is True
    assert c.get("/health").status_code == 200


def test_freezing_refuses_to_save_answers_that_could_not_be_written(tmp_path, loaded):
    settings = make_settings(tmp_path)
    loaded.llm = None  # no key: sources only, no answers
    with pytest.raises(RuntimeError, match="could not write"):
        freeze(settings, lambda: loaded, load_questions(settings.study_questions))


def test_a_frozen_file_from_other_questions_is_rejected(tmp_path, loaded):
    settings = make_settings(tmp_path)
    qs = load_questions(settings.study_questions)
    loaded.llm = FakeLLM(["Answer [1]."])
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(freeze(settings, lambda: loaded, qs)))
    changed = [dict(q) for q in qs]
    changed[1]["question"] = "a different question"
    with pytest.raises(RuntimeError, match="different question set"):
        FrozenStimuli(path, changed)


def test_freeze_produces_a_blind_grading_sheet_and_an_item_report(tmp_path, loaded):
    from backend.study.freeze import REPORT_COLUMNS, sheets

    settings = make_settings(tmp_path)
    qs = load_questions(settings.study_questions)
    loaded.llm = FakeLLM(["Fusion combines rankings [1]."])
    data = freeze(settings, lambda: loaded, qs)
    ai, report = sheets(data, qs)
    assert {r["question_id"] for r in ai} == {q["id"] for q in qs}
    assert all(set(r) == set(ex.AI_SHEET_COLUMNS) and r["final"] == "" for r in ai)
    blob = json.dumps(ai).lower()
    assert "bucket" not in blob and "badge" not in blob and "confidence" not in blob
    assert [r["question_id"] for r in report] == [q["id"] for q in qs]
    assert set(report[0]) == set(REPORT_COLUMNS) and report[0]["practice"] is True
    ids = {(r["question_id"], r["ai_answer_id"]) for r in ai}
    assert len(ids) == len(ai)  # one row per distinct answer


# ------------------------------------------------------------- export and grading
def run_session(c, store, pipe):
    pipe.llm = FakeLLM(["Fusion combines rankings [1].", "Sourdough needs yeast [1]."])
    start(c, store, "P9", "badge")
    log = lambda q, t, p=None, ts=None: store.log("P9", q, t, p or {}, ts)  # noqa: E731
    log("q1", "question_shown", ts=100.0)
    c.post("/study/ask", json={"code": "P9", "question_id": "q1"})
    log("q1", "doc_viewer_opened", ts=110.0)
    log("q1", "doc_viewer_closed", ts=125.0)
    log("q1", "final_answer", {"accepted": True, "text": ""}, ts=128.0)
    log("q1", "stated_confidence", {"value": 6}, ts=130.0)
    log("q1", "final_answer_submitted", ts=131.0)
    log("q3", "question_shown", ts=140.0)
    c.post("/study/ask", json={"code": "P9", "question_id": "q3"})
    log("q3", "final_answer", {"accepted": False, "text": "flour water salt starter"}, ts=150.0)
    log("q3", "stated_confidence", {"value": 3}, ts=155.0)
    log("q3", "final_answer_submitted", ts=156.0)
    items = items_for("badge", DEFAULT_ITEMS)
    vals = {i["id"]: (5 if i["type"] == "scale" else i["options"][0]) for i in items}
    log(None, "survey", {"values": vals})


def test_export_computes_reliance_correctness_timing_and_hides_identity(study):
    c, store, pipe, tmp = study
    run_session(c, store, pipe)
    qs = load_questions(tmp / "questions.json")
    rows = {r["question_id"]: r for r in ex.build_rows(store, qs, salt="salt")}
    assert set(rows["q1"]) == set(ex.COLUMNS) and rows["q1"]["participant_id"] != "P9"
    r1, r3 = rows["q1"], rows["q3"]
    assert r1["condition"] == "badge" and r1["accepted"] is True and r1["position"] >= 1
    assert r1["consulted_doc"] is True and r1["doc_seconds"] == 15.0
    assert r1["stated_confidence"] == 6 and r1["time_to_confidence_s"] == 30.0
    assert r1["time_total_s"] == 31.0 and r1["ai_correct_source"] == "auto"
    assert r1["ai_correct"] is True and r1["appropriate_reliance"] is True
    assert r3["accepted"] is False and r3["final_answer_text"] == "flour water salt starter"
    # the automatic judge only checks retrieval, so it calls the "yeast" answer correct; this is
    # exactly why real grades are merged in (next test) and why the sheets exist
    assert r3["ai_correct"] is True and r3["ai_correct_source"] == "auto"
    assert r3["underreliance"] is True and r3["appropriate_reliance"] is False
    assert r3["human_correct"] == "" and r3["final_correct"] == ""  # not graded yet
    assert rows["p0"]["unreached"] is True and rows["p0"]["n_asks"] == 0
    assert ex.build_rows(store, qs, keep_codes=True)[0]["participant_id"] == "P9"


def test_grading_sheets_are_blind_and_grades_flow_back_into_the_dataset(study):
    c, store, pipe, tmp = study
    run_session(c, store, pipe)
    qs = load_questions(tmp / "questions.json")
    ai, human = ex.ai_answer_sheet(store, qs), ex.human_answer_sheet(store, qs, "salt")
    for sheet in (ai, human):
        text = json.dumps(sheet)
        assert "badge" not in text.lower() and "P9" not in text and "bucket" not in text
    assert len(ai) == 2 and len(human) == 1 and human[0]["human_answer_text"].startswith("flour")
    ai_path, h_path = tmp / "ai.csv", tmp / "human.csv"
    sheets = ((ai_path, ex.AI_SHEET_COLUMNS, ai), (h_path, ex.HUMAN_SHEET_COLUMNS, human))
    for path, cols, rows in sheets:
        for r in rows:
            r["final"] = "1" if r.get("question_id") == "q1" or r.get("sheet_id") else "0"
        path.write_text(ex.to_csv(cols, rows))
    out = tmp / "out"
    paths = ex.write_export(store, qs, DEFAULT_ITEMS, out, "salt", False, str(ai_path), str(h_path))
    rows = {r["question_id"]: r for r in csv.DictReader(paths["per_question"].open())}
    assert rows["q1"]["ai_correct_source"] == "graded" and rows["q1"]["ai_correct"] == "True"
    assert rows["q3"]["ai_correct"] == "False" and rows["q3"]["accepted"] == "False"
    assert rows["q3"]["human_correct"] == "True" and rows["q3"]["final_correct"] == "True"
    assert rows["q3"]["appropriate_reliance"] == "True"  # rejected an incorrect answer
    assert "P9" not in paths["events"].read_text()
    survey = list(csv.DictReader(paths["survey"].open()))
    assert len(survey) == 1 and survey[0]["condition"] == "badge" and survey[0]["trust_1"] == "5"
    assert (out / "grading" / "ai_answers.csv").exists()


def test_a_question_with_no_answer_shown_is_never_correct(study):
    c, store, pipe, tmp = study  # no LLM: sources come back but no answer
    start(c, store, "P10", "baseline")
    c.post("/study/ask", json={"code": "P10", "question_id": "q1"})
    rows = ex.build_rows(store, load_questions(tmp / "questions.json"))
    row = {r["question_id"]: r for r in rows}["q1"]
    assert row["ai_correct"] is False and row["ai_answer_id"] == "none"


def test_refusing_an_unanswerable_question_counts_as_correct(study):
    c, store, pipe, tmp = study
    pipe.llm = FakeLLM(["NOT_IN_SOURCES: nothing about sport."])
    start(c, store, "P11", "baseline")
    c.post("/study/ask", json={"code": "P11", "question_id": "q2"})
    c.post("/study/ask", json={"code": "P11", "question_id": "q1"})
    rows = {
        r["question_id"]: r for r in ex.build_rows(store, load_questions(tmp / "questions.json"))
    }
    assert rows["q2"]["ai_correct"] is True and rows["q2"]["ai_refused"] is True
    assert rows["q1"]["ai_correct"] is False  # refused although the answer was in the sources


# ------------------------------------------------------------- admin
def test_admin_downloads_need_the_token_and_are_off_without_one(study, tmp_path, loaded):
    c, store, pipe, tmp = study
    run_session(c, store, pipe)
    assert c.get("/study/admin/per_question.csv").status_code == 403
    assert c.get("/study/admin/per_question.csv?token=nope").status_code == 403
    names = ("per_question.csv", "events.csv", "survey.csv", "ai_answers.csv", "human_answers.csv")
    for name in names:
        r = c.get(f"/study/admin/{name}", headers={"X-Admin-Token": TOKEN})
        assert r.status_code == 200 and "text/csv" in r.headers["content-type"], name
    q = c.get(f"/study/admin/per_question.csv?token={TOKEN}").text
    assert q.splitlines()[0].startswith("participant_id,condition") and "P9" not in q
    db = c.get(f"/study/admin/db?token={TOKEN}")
    assert db.status_code == 200 and db.content.startswith(b"SQLite format 3")
    off_settings = make_settings(
        tmp_path, study_admin_token="", study_db=str(tmp_path / "o.sqlite")
    )
    off = TestClient(create_app(loaded, off_settings))
    assert off.get("/study/admin/per_question.csv?token=x").status_code == 404


def test_fit_cutoffs_suggests_ordered_thresholds():
    from eval.fit_cutoffs import suggest

    recs = [
        {"mode": "hybrid", "strategy": "recursive", "success": i % 3 != 0, "conf_score": i / 60}
        for i in range(60)
    ]
    out = suggest(recs, "recursive")
    assert out["n"] == 60 and out["low"] < out["high"]
    with pytest.raises(SystemExit):
        suggest(recs[:5], "recursive")


def test_fixture_docs_sanity():
    assert "sparse retrieval function" in DOC_B
