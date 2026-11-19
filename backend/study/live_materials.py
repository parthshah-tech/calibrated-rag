"""Versioned materials for the voluntary-use study and its shared answer judgments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..generation import SYSTEM_PROMPT


class Material(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(Material):
    name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class TaskItem(Material):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    prompt: str = Field(min_length=1)
    answer_key: str = ""


class Evaluation(Material):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    passages: list[Document] = Field(min_length=1)
    answer_key: str = ""
    ai_correct: bool | None = None


class SurveyItem(Material):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    text: str = Field(min_length=1)
    type: Literal["scale", "choice"] = "scale"
    options: list[str] = Field(default_factory=list)
    low: str = "Strongly disagree"
    high: str = "Strongly agree"
    conditions: list[Literal["baseline", "badge"]] = ["baseline", "badge"]


class LiveMaterials(Material):
    title: str = Field(min_length=1)
    preview: bool = True
    info: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    task: str = Field(min_length=1)
    practice: TaskItem
    items: list[TaskItem] = Field(min_length=1)
    documents: list[Document] = Field(min_length=1)
    evaluations: list[Evaluation] = Field(default_factory=list)
    survey: list[SurveyItem] = Field(min_length=1)
    debrief: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        names = [document.name for document in self.documents]
        if len(names) != len(set(names)):
            raise ValueError("Study document names must be unique")
        if any(Path(name).suffix.lower() not in {".txt", ".md"} for name in names):
            raise ValueError("Study documents contain extracted text; use .txt or .md names")
        for values in [self.items, self.evaluations, self.survey]:
            ids = [v.id for v in values]
            if len(ids) != len(set(ids)):
                raise ValueError("Material IDs must be unique within each section")
        if self.practice.id in {item.id for item in self.items}:
            raise ValueError("Practice must have its own ID")
        for item in self.survey:
            if item.type == "choice" and not item.options:
                raise ValueError("Choice questions need options")
        return self


def load_materials(path):
    return LiveMaterials.model_validate_json(Path(path).read_text(encoding="utf-8"))


def config_hash(materials, settings):
    names = (
        "embedder",
        "embed_model",
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "confidence_metric",
        "confidence_k",
        "candidate_pool",
        "rrf_k",
        "llm_model",
        "llm_base_url",
        "llm_reasoning_effort",
        "study_time_budget_min",
    )
    value = {
        "flow": "voluntary-live-v1",
        "materials": materials.model_dump(),
        "settings": {key: getattr(settings, key) for key in names},
        "prompt": SYSTEM_PROMPT,
        "retrieval": {"k": 5, "mode": "hybrid", "history_turns": 6},
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
