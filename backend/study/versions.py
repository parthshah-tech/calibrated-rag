"""Publish study materials while keeping existing sessions on their original version."""

import asyncio
import json
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import Settings
from ..main import create_app
from .live_admin import DraftUpdate, edit_materials, revision, write_atomic
from .live_api import COOKIE
from .live_materials import LiveMaterials, load_materials


@dataclass
class Version:
    app: FastAPI
    materials: LiveMaterials


class StudyVersions:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.directory = Path(settings.study_db).with_suffix(".versions")
        self.index = self.directory / "published.json"
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(self):
        async with AsyncExitStack() as stack:
            self.stack = stack
            self.published = (
                json.loads(self.index.read_text("utf-8")) if self.index.exists() else []
            )
            self.versions = [await self.open_version(self.settings, stack)]
            for directory in self.published:
                self.versions.append(
                    await self.open_version(self.version_settings(directory), stack)
                )
            yield

    def version_settings(self, version_id: str) -> Settings:
        directory = self.directory / version_id
        return replace(
            self.settings,
            study_task=str(directory / "materials.json"),
            study_db=str(directory / "study.sqlite"),
            data_dir=str(directory / "corpus"),
        )

    async def open_version(self, settings: Settings, stack: AsyncExitStack) -> Version:
        materials = load_materials(settings.study_task)
        app = create_app(settings=settings)
        await stack.enter_async_context(app.router.lifespan_context(app))
        return Version(app, materials)

    def participant_version(self, request: Request) -> int:
        token = request.cookies.get(COOKIE, "")
        if re.fullmatch(r"[0-9a-f]{64}", token):
            code = sha256(token.encode()).hexdigest()
            for index in range(len(self.versions) - 1, -1, -1):
                if self.versions[index].app.state.live_store.participant(code):
                    return index
        return len(self.versions) - 1

    async def participant(self, scope, receive, send):
        request = Request(scope)
        if request.method == "POST" and request.url.path == "/study/session":
            async with self.lock:
                index = self.participant_version(request)
                shown = request.headers.get("x-study-version")
                if shown is not None and shown != str(index + 1):
                    response = JSONResponse(
                        {"detail": "The study changed. Reload the page before continuing."}, 409
                    )
                    await response(scope, receive, send)
                    return
                await self.versions[index].app(scope, receive, send)
            return

        index = self.participant_version(request)
        if request.url.path == "/study/info":

            async def send_version(message):
                if message["type"] == "http.response.start":
                    message = {
                        **message,
                        "headers": [
                            *message["headers"],
                            (b"x-study-version", str(index + 1).encode()),
                        ],
                    }
                await send(message)

            await self.versions[index].app(scope, receive, send_version)
            return
        await self.versions[index].app(scope, receive, send)

    async def admin(self, scope, receive, send):
        request = Request(scope, receive)
        path = request.url.path
        if path == "/study/admin/runs" and request.method == "GET":
            runs = [
                {
                    "id": index,
                    "title": version.materials.title,
                    "preview": version.materials.preview,
                    "current": index == len(self.versions),
                }
                for index, version in enumerate(self.versions, start=1)
            ]
            response = JSONResponse({"runs": runs[::-1]})
        elif path == "/study/admin/materials" and request.method == "GET":
            response = JSONResponse(self.materials_result(self.versions[-1].materials))
        elif path == "/study/admin/materials" and request.method == "POST":
            try:
                edit = DraftUpdate.model_validate(await request.json())
                async with self.lock:
                    previous = self.versions[-1].materials
                    materials, reset = edit_materials(edit, previous, draft=False)
                    if materials != previous:
                        await self.publish(materials)
                response = JSONResponse(
                    {**self.materials_result(materials), "evaluation_reset": reset}
                )
            except ValueError:
                response = JSONResponse({"detail": "Submit a valid study setup."}, 422)
            except HTTPException as error:
                response = JSONResponse({"detail": error.detail}, error.status_code)
        else:
            index = len(self.versions) - 1
            if path in {"/study/admin/export.zip", "/study/admin/db"}:
                selected = request.query_params.get("run", str(index + 1))
                if not re.fullmatch(r"[0-9]{1,9}", selected) or not 1 <= int(selected) <= len(
                    self.versions
                ):
                    response = JSONResponse({"detail": "Study version not found."}, 404)
                    await response(scope, receive, send)
                    return
                index = int(selected) - 1
            await self.versions[index].app(scope, receive, send)
            return
        await response(scope, receive, send)

    def materials_result(self, materials: LiveMaterials):
        return {
            "revision": revision(materials),
            "materials": materials.model_dump(mode="json"),
            "saved": True,
        }

    async def publish(self, materials: LiveMaterials):
        version_id = uuid4().hex
        settings = self.version_settings(version_id)
        path = Path(settings.study_task)
        path.parent.mkdir(parents=True)
        write_atomic(path, materials.model_dump_json(indent=2))
        async with AsyncExitStack() as stack:
            version = await self.open_version(settings, stack)
            published = [*self.published, version_id]
            write_atomic(self.index, json.dumps(published))
            self.stack.push_async_callback(stack.pop_all().aclose)
        self.published = published
        self.versions.append(version)
