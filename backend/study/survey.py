"""The post-task questionnaire. Defaults below; override by writing the same structure to
study/survey.json. `reverse` items are scored 8 minus the answer during analysis."""

from __future__ import annotations

import json
from pathlib import Path

ALL = ["baseline", "badge", "badge_reformulation"]
BADGE = ["badge", "badge_reformulation"]
AGREE = "scale"  # 1 (strongly disagree) to 7 (strongly agree)


def _agree(id_, text, conditions=ALL, reverse=False):
    item = {"id": id_, "text": text, "type": AGREE, "conditions": conditions}
    return {**item, "reverse": True} if reverse else item


DEFAULT_ITEMS = [
    _agree("trust_1", "I trusted the assistant's answers."),
    _agree("trust_2", "I could depend on the assistant to give correct answers."),
    _agree("trust_3", "I was wary of the assistant's answers.", reverse=True),
    _agree("trust_4", "The assistant's answers were reliable."),
    _agree("trust_5", "I felt I had to double-check the assistant's answers."),
    _agree("effort_1", "The task was mentally demanding."),
    _agree(
        "badge_1", "The confidence indicator helped me decide when to check the documents.", BADGE
    ),
    _agree("badge_2", "I understood what the confidence indicator meant.", BADGE),
    _agree("badge_3", "I mostly ignored the confidence indicator.", BADGE, reverse=True),
    _agree("reform_1", "The suggested rewordings were useful.", ["badge_reformulation"]),
    {
        "id": "manip_check",
        "text": "Did the assistant show a confidence indicator?",
        "type": "choice",
        "options": ["Yes", "No", "Not sure"],
        "conditions": ALL,
    },
    {
        "id": "ai_use",
        "text": "How often do you use AI chatbots?",
        "type": "choice",
        "options": ["Never", "Rarely", "Monthly", "Weekly", "Daily"],
        "conditions": ALL,
    },
]


def load_items(path: str | Path = "study/survey.json") -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else DEFAULT_ITEMS


def items_for(condition: str, items: list[dict]) -> list[dict]:
    return [i for i in items if condition in i["conditions"]]


def public_items(condition: str, items: list[dict]) -> list[dict]:
    keep = ("id", "text", "type", "options")
    return [{k: i[k] for k in keep if k in i} for i in items_for(condition, items)]


def validate(values: dict, condition: str, items: list[dict]) -> None:
    """Every item shown to this condition must be answered, and nothing else may be sent."""
    expected = {i["id"]: i for i in items_for(condition, items)}
    if set(values) != set(expected):
        raise ValueError("answers must cover exactly the questions shown")
    for k, v in values.items():
        item = expected[k]
        if item["type"] == AGREE:
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 7:
                raise ValueError(f"{k}: expected an integer from 1 to 7")
        elif v not in item["options"]:
            raise ValueError(f"{k}: not one of the options")
