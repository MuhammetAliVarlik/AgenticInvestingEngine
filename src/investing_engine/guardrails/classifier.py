"""Optional model-based prompt-injection classifier (Llama Prompt Guard 2 on Groq).

The heuristic scanner is fast and explainable but pattern-bound; a small
dedicated classifier catches paraphrased attacks it would miss. The model
scores short chunks, so text is split into windows and any window above the
threshold is quarantined as a whole.

The classifier fails open with a logged warning: an outage of this optional
layer must not block analyses, and the heuristic layer still applies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
CHUNK_CHARS = 1500  # comfortably inside the classifier's 512-token window
_LABEL_SCORES = {"benign": 0.0, "malicious": 1.0, "injection": 1.0, "jailbreak": 1.0}


class InjectionClassifier(Protocol):
    def score(self, text: str) -> float:
        """Probability (0-1) that ``text`` contains a prompt-injection attempt."""
        ...


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries into windows of at most ``size`` characters."""
    windows: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        if len(paragraph) > size and current:
            windows.append(current)  # flush first so text order is preserved
            current = ""
        while len(paragraph) > size:
            windows.append(paragraph[:size])
            paragraph = paragraph[size:]
        if len(current) + len(paragraph) + 2 > size and current:
            windows.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current.strip():
        windows.append(current)
    return windows


def parse_score(content: str) -> float | None:
    """Interpret the classifier's reply, which may be a probability or a label."""
    text = content.strip().lower()
    try:
        return min(max(float(text), 0.0), 1.0)
    except ValueError:
        pass
    for label, score in _LABEL_SCORES.items():
        if label in text:
            return score
    return None


@dataclass
class GroqPromptGuard:
    api_key: str
    model: str = "meta-llama/llama-prompt-guard-2-86m"
    timeout: float = 10.0
    transport: httpx.BaseTransport | None = None

    def score(self, text: str) -> float:
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            response = client.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": text}]},
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        score = parse_score(str(content))
        if score is None:
            raise ValueError("Unrecognised classifier response")
        return score


def classify_chunks(
    classifier: InjectionClassifier, text: str, *, threshold: float
) -> tuple[str, int]:
    """Quarantine every chunk scoring at or above ``threshold``.

    Returns (text, number_of_flagged_chunks). Fails open on classifier errors.
    """
    from investing_engine.guardrails.injection import QUARANTINE_MARKER

    kept: list[str] = []
    flagged = 0
    for window in chunks(text):
        try:
            score = classifier.score(window)
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            logger.warning("Prompt Guard unavailable, continuing with heuristics: %s", exc)
            return text, 0
        if score >= threshold:
            flagged += 1
            kept.append(QUARANTINE_MARKER)
        else:
            kept.append(window)
    return "\n\n".join(kept), flagged
