"""Participant-facing wording. Defaults below; the team can override any key by putting a JSON
object in study/texts.json (for example to change the consent or debrief text)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {
    "info": (
        "This study looks at how people use an AI assistant that answers questions about "
        "documents. You will answer a series of questions about a few short documents with the "
        "help of the assistant, then fill in a short questionnaire. It takes about 40 minutes. "
        "Taking part is voluntary and you can stop at any time. We record your answers, your "
        "confidence ratings, which documents you open, and timings, linked only to a participant "
        "code and never to your name. Afterwards a researcher will ask you a few questions about "
        "your experience."
    ),
    "consent": "I have read the information above and agree to take part.",
    "instructions": (
        "For each question: (1) press Ask the assistant to see its answer and the passages it "
        "used. (2) You may open the source documents to check; the browser's find (Ctrl+F) works "
        "inside them. (3) Decide whether to accept the assistant's answer or give a different "
        "one, then rate how confident you are that your final answer is correct. You will not be "
        "told whether you were right. A timer shows how much time is left for the questions; any "
        "you do not reach are skipped. The first question is practice and is not scored."
    ),
    "badge_note": (
        "The assistant also shows a confidence indicator (Low, Medium or High). It shows how "
        "much two search methods agree. It is not a guarantee that the answer is right."
    ),
    "debrief": (
        "Thank you. This study compares versions of the assistant: some participants saw no "
        "confidence indicator, some saw an indicator, and some saw an indicator plus suggested "
        "rewordings of the question. We want to understand whether the indicator helps people "
        "trust the assistant's answers when they are right and check them when they are wrong. "
        "You were not told this beforehand so that it would not change how you worked. You were "
        "not told whether your answers were right; the assistant's answers were prepared in "
        "advance and some of them were incorrect, as they would be in real use. Your data is "
        "stored under your code only, and you may ask for it to be removed. Please tell the "
        "researcher that you have finished."
    ),
}


def load_texts(path: str | Path = "study/texts.json") -> dict:
    texts = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        loaded = json.loads(p.read_text(encoding="utf-8"))
        texts.update({k: v for k, v in loaded.items() if k in texts})
    return texts
