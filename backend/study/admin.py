"""Token-protected downloads, so data can be pulled from a running server at any time.
Disabled (404) unless STUDY_ADMIN_TOKEN is set."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response

from . import export as ex

Token = Annotated[str | None, Header()]


def register_admin(app: FastAPI, settings, store, questions, items) -> None:
    def check(header: str | None, token: str | None) -> None:
        if not settings.study_admin_token:
            raise HTTPException(404, "not found")
        given = (header or token or "").encode()
        if not hmac.compare_digest(given, settings.study_admin_token.encode()):
            raise HTTPException(403, "bad token")

    def csv_response(name: str, columns, rows) -> Response:
        return Response(
            ex.to_csv(columns, rows),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    salt = settings.study_salt

    @app.get("/study/admin/per_question.csv", include_in_schema=False)
    def per_question(token: str | None = None, x_admin_token: Token = None):
        check(x_admin_token, token)
        return csv_response("per_question.csv", ex.COLUMNS, ex.build_rows(store, questions, salt))

    @app.get("/study/admin/events.csv", include_in_schema=False)
    def events(token: str | None = None, x_admin_token: Token = None):
        check(x_admin_token, token)
        return csv_response("events.csv", ex.EVENT_COLUMNS, ex.events_rows(store, salt))

    @app.get("/study/admin/survey.csv", include_in_schema=False)
    def survey(token: str | None = None, x_admin_token: Token = None):
        check(x_admin_token, token)
        cols, rows = ex.survey_rows(store, items, salt)
        return csv_response("survey.csv", cols, rows)

    @app.get("/study/admin/ai_answers.csv", include_in_schema=False)
    def ai_answers(token: str | None = None, x_admin_token: Token = None):
        check(x_admin_token, token)
        sheet = ex.ai_answer_sheet(store, questions)
        return csv_response("ai_answers.csv", ex.AI_SHEET_COLUMNS, sheet)

    @app.get("/study/admin/human_answers.csv", include_in_schema=False)
    def human_answers(token: str | None = None, x_admin_token: Token = None):
        check(x_admin_token, token)
        sheet = ex.human_answer_sheet(store, questions, salt)
        return csv_response("human_answers.csv", ex.HUMAN_SHEET_COLUMNS, sheet)

    @app.get("/study/admin/db", include_in_schema=False)
    def database(token: str | None = None, x_admin_token: Token = None):
        check(x_admin_token, token)
        return Response(
            store.serialize(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="study.sqlite"'},
        )
