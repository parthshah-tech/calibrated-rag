"""Serve document chat, the study, and protected administration on one origin."""

import os
import re
import secrets
from base64 import b64encode
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import replace
from hmac import compare_digest
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .config import Settings
from .main import create_app
from .study.versions import StudyVersions

RAG_ROUTES = {
    "GET": {"/", "/app.css", "/app-config", "/documents", "/health"},
    "POST": {"/ingest", "/ask", "/search"},
}
STUDY_ROUTES = {
    "GET": {"/study", "/study/app.js", "/study/info", "/study/session", "/study/documents"},
    "POST": {
        "/study/session",
        "/study/ask",
        "/study/draft",
        "/study/submit",
        "/study/judge",
        "/study/view",
        "/study/view/close",
        "/study/withdraw",
    },
}
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class PublicApp:
    def __init__(
        self, settings: Settings, rag_data: str, origin: str, username: str, password: str
    ):
        parsed = urlsplit(origin)
        if origin and (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("PUBLIC_ORIGIN must be an HTTPS origin, with no path.")
        self.origin = origin.rstrip("/")
        self.public_host = parsed.netloc
        self.credentials = b"Basic " + b64encode(f"{username}:{password}".encode())
        self.attempts = defaultdict(deque)
        self.study = StudyVersions(
            replace(
                settings,
                study_mode=True,
                study_flow="live",
                isolate_visitors=False,
                study_admin_token=settings.study_admin_token or secrets.token_urlsafe(32),
            )
        )
        self.rag = create_app(
            settings=replace(
                settings,
                study_mode=False,
                isolate_visitors=True,
                data_dir=rag_data,
            )
        )
        frontend = Path(__file__).resolve().parent.parent / "frontend"
        self.admin_page = (
            (frontend / "study-admin.html")
            .read_text("utf-8")
            .replace("<body>", '<body data-admin-mode="published">', 1)
        )
        self.lifespan_app = Starlette(lifespan=self.lifespan)

    @asynccontextmanager
    async def lifespan(self, app):
        async with self.study.lifespan(), self.rag.router.lifespan_context(self.rag):
            yield

    def allow_request(self, request: Request) -> bool:
        path = request.url.path
        buckets = {
            "/ask": "answer",
            "/study/ask": "answer",
            "/search": "answer",
            "/ingest": "upload",
        }
        bucket = buckets.get(path)
        if request.method != "POST" or bucket is None:
            return True
        now = monotonic()
        for key, attempts in list(self.attempts.items()):
            while attempts and now - attempts[0] >= 60:
                attempts.popleft()
            if not attempts:
                del self.attempts[key]
        client = request.client.host if request.client else "unknown"
        individual = (bucket, client)
        combined = (bucket, None)
        per_client, total = (18, 36) if bucket == "answer" else (12, 30)
        if len(self.attempts[individual]) >= per_client or len(self.attempts[combined]) >= total:
            return False
        self.attempts[individual].append(now)
        self.attempts[combined].append(now)
        return True

    def local_admin(self, request: Request) -> bool:
        loopback = {"127.0.0.1", "::1"}
        return (
            not self.origin
            and request.client is not None
            and request.client.host in loopback
            and request.url.hostname in loopback | {"localhost"}
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.lifespan_app(scope, receive, send)
            return

        request = Request(scope)
        path = request.url.path
        method = request.method
        actual_origin = str(request.base_url).rstrip("/")
        public_request = request.url.scheme == "https" and bool(self.origin)
        expected_origin = self.origin if public_request else actual_origin
        origin_header = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site")
        # Some relays rewrite Host and Origin. Same-origin fetch metadata is set by browsers.
        same_origin_relay = public_request and fetch_site == "same-origin"
        cross_origin = fetch_site not in {None, "same-origin", "none"} or (
            origin_header and origin_header != expected_origin and not same_origin_relay
        )

        if method in WRITE_METHODS and cross_origin:
            endpoint = JSONResponse({"detail": "Use the application's own page."}, 403)
        elif path == "/study/admin" or path.startswith("/study/admin/"):
            if request.url.scheme != "https" and not self.local_admin(request):
                endpoint = PlainTextResponse("Administrator sign-in requires HTTPS.", 403)
            elif not compare_digest(
                request.headers.get("authorization", "").encode(), self.credentials
            ):
                endpoint = PlainTextResponse(
                    "Administrator sign-in required.",
                    401,
                    headers={
                        "WWW-Authenticate": 'Basic realm="Study administration", charset="UTF-8"'
                    },
                )
            elif path == "/study/admin" and method == "GET":
                endpoint = HTMLResponse(self.admin_page)
            else:
                headers = [
                    (key, value)
                    for key, value in scope["headers"]
                    if key not in {b"authorization", b"x-admin-token"}
                ]
                headers.append((b"x-admin-token", self.study.settings.study_admin_token.encode()))
                scope = {**scope, "headers": headers}
                endpoint = self.study.admin
        elif path in STUDY_ROUTES.get(method, ()):
            endpoint = self.study.participant
        elif path in RAG_ROUTES.get(method, ()) or (
            method in {"GET", "DELETE"} and re.fullmatch(r"/documents/[0-9a-f]{16}", path)
        ):
            endpoint = self.rag
        else:
            endpoint = PlainTextResponse("Not found.", 404)

        if not (method in WRITE_METHODS and cross_origin) and not self.allow_request(request):
            endpoint = JSONResponse(
                {"detail": "Too many requests. Try again in a minute."},
                429,
                headers={"Retry-After": "60"},
            )

        if public_request:
            headers = [
                (key, value) for key, value in scope["headers"] if key not in {b"host", b"origin"}
            ]
            headers.append((b"host", self.public_host.encode("ascii")))
            if origin_header is not None or same_origin_relay:
                headers.append((b"origin", expected_origin.encode("ascii")))
            scope = {**scope, "headers": headers}

        async def protect_response(message):
            if message["type"] == "http.response.start":
                headers = [
                    (key, value) for key, value in message["headers"] if key != b"cache-control"
                ]
                headers.extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                    ]
                )
                if request.url.scheme == "https":
                    headers.append((b"strict-transport-security", b"max-age=86400"))
                message = {**message, "headers": headers}
            await send(message)

        await endpoint(scope, receive, protect_response)


def create_public_app():
    settings = Settings.from_env()
    username = os.environ.get("PUBLIC_ADMIN_USERNAME", "")
    password = os.environ.get("PUBLIC_ADMIN_PASSWORD", "")
    if not username or not password or ":" in username:
        raise ValueError("Set PUBLIC_ADMIN_USERNAME and PUBLIC_ADMIN_PASSWORD before starting.")
    rag_data = os.environ.get("PUBLIC_RAG_DATA_DIR", "./data/rag")
    if Path(rag_data).resolve() == Path(settings.data_dir).resolve():
        raise ValueError("Use separate PUBLIC_RAG_DATA_DIR and DATA_DIR directories.")
    return PublicApp(settings, rag_data, os.environ.get("PUBLIC_ORIGIN", ""), username, password)
