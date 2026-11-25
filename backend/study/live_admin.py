"""Edit the next study run without changing the active participant materials."""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Annotated

from fastapi import Depends, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import Field, ValidationError

from ..ingestion import SUPPORTED, extract_text
from .live_export import require_admin
from .live_materials import Document, LiveMaterials, Material, TaskItem, load_materials


class DraftUpdate(Material):
    revision: str
    title: str
    info: str
    instructions: str
    task: str
    practice: TaskItem
    items: list[TaskItem]
    documents: list[Document] = Field(min_length=1)
    debrief: str


def revision(materials):
    return sha256(materials.model_dump_json().encode()).hexdigest()


def edit_materials(edit: DraftUpdate, previous: LiveMaterials, *, draft: bool):
    if edit.revision != revision(previous):
        raise HTTPException(409, "Another edit was saved. Reopen the editor before saving.")
    changes = edit.model_dump(exclude={"revision"})
    before = previous.model_dump()
    evidence_fields = ("task", "practice", "items", "documents")
    reset_fields = ("title", *evidence_fields) if draft else evidence_fields
    reset = any(changes[key] != before[key] for key in reset_fields)
    data = {**before, **changes, "preview": draft or previous.preview or reset}
    if reset:
        data["evaluations"] = []
    if any(not doc.text.strip() for doc in edit.documents):
        raise HTTPException(422, "Each document needs readable text.")
    try:
        materials = LiveMaterials.model_validate(data)
    except ValidationError as error:
        messages = [
            " ".join([*(str(part) for part in item["loc"]), item["msg"]]) for item in error.errors()
        ]
        raise HTTPException(422, "; ".join(messages)) from error
    return materials, reset and bool(previous.evaluations)


def write_atomic(path: Path, contents: str):
    with NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w", delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(contents)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def register_admin(app, settings, active, frontend):
    path = Path(settings.study_db).with_name(Path(settings.study_db).name + ".draft.json")
    lock = Lock()

    def authorize(x_admin_token: Annotated[str | None, Header()] = None):
        require_admin(settings, x_admin_token)

    def current():
        if path.exists():
            return load_materials(path)
        return active.model_copy(update={"preview": True})

    @app.get("/study/admin", include_in_schema=False)
    def setup_page():
        if not settings.study_admin_token:
            raise HTTPException(404, "Not found.")
        return FileResponse(frontend / "study-admin.html")

    @app.get("/study/admin/setup.js", include_in_schema=False)
    def setup_script():
        return FileResponse(frontend / "study-admin.js")

    @app.get("/study/admin/materials", dependencies=[Depends(authorize)], include_in_schema=False)
    def read_draft():
        with lock:
            materials = current()
            return {"revision": revision(materials), "materials": materials, "saved": path.exists()}

    @app.post("/study/admin/materials", dependencies=[Depends(authorize)], include_in_schema=False)
    def save_draft(edit: DraftUpdate):
        with lock:
            materials, reset = edit_materials(edit, current(), draft=True)
            write_atomic(path, materials.model_dump_json(indent=2))
            return {
                "revision": revision(materials),
                "materials": materials,
                "saved": True,
                "evaluation_reset": reset,
            }

    @app.post("/study/admin/extract", dependencies=[Depends(authorize)], include_in_schema=False)
    def read_upload(file: UploadFile):
        name = Path((file.filename or "").replace("\\", "/")).name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED:
            raise HTTPException(415, "Choose a TXT, Markdown, PDF or DOCX file.")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "This file is too large.")
        try:
            text = extract_text(name, data).strip()
        except Exception as error:
            raise HTTPException(422, "Could not read this document.") from error
        if not text:
            raise HTTPException(422, "No text found. Use a text document or an OCR copy.")
        if len(text.encode()) > settings.max_upload_bytes:
            raise HTTPException(413, "The extracted text is too large.")
        if ext in {".pdf", ".docx"}:
            name = f"{Path(name).stem}.txt"
        return {"name": name, "text": text}
