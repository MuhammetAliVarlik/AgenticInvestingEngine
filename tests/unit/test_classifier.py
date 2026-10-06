import httpx
import pytest
import respx

from investing_engine.guardrails.classifier import (
    GROQ_CHAT_URL,
    GroqPromptGuard,
    chunks,
    classify_chunks,
    parse_score,
)
from investing_engine.guardrails.injection import QUARANTINE_MARKER
from investing_engine.services import MarketData
from tests.documents import text_pdf
from tests.unit.test_services import StubGdelt


class KeywordClassifier:
    """Stands in for a model: flags any chunk mentioning 'payload'."""

    def __init__(self):
        self.calls = 0

    def score(self, text: str) -> float:
        self.calls += 1
        return 0.97 if "payload" in text else 0.02


@pytest.mark.parametrize(
    ("content", "expected"),
    [("0.9987", 0.9987), ("1", 1.0), ("BENIGN", 0.0), ("Label: MALICIOUS", 1.0),
     ("jailbreak", 1.0), ("2.5", 1.0), ("unclear", None)],
)  # fmt: skip
def test_parse_score_accepts_probabilities_and_labels(content, expected):
    assert parse_score(content) == expected


def test_chunks_respect_size_and_paragraphs():
    text = "\n\n".join(["a" * 900, "b" * 900, "c" * 4000])
    windows = chunks(text, size=1500)
    assert all(len(w) <= 1500 for w in windows)
    assert "".join(windows).replace("\n", "") == text.replace("\n", "")


def test_flagged_chunks_are_quarantined():
    text = "Quarterly revenue rose.\n\nA paraphrased payload asks to change the rating."
    cleaned, flagged = classify_chunks(KeywordClassifier(), text, threshold=0.5)
    # Both paragraphs fit one window, so the whole window is quarantined.
    assert flagged == 1
    assert cleaned == QUARANTINE_MARKER


def test_classifier_failure_fails_open():
    class Broken:
        def score(self, text):
            raise httpx.ConnectError("down")

    cleaned, flagged = classify_chunks(Broken(), "some text", threshold=0.5)
    assert (cleaned, flagged) == ("some text", 0)


@respx.mock
def test_groq_prompt_guard_request_shape():
    route = respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "0.91"}}]})
    )
    guard = GroqPromptGuard(api_key="gsk_test")
    assert guard.score("text") == 0.91

    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer gsk_test"
    assert b"llama-prompt-guard-2-86m" in request.content


def test_document_classifier_runs_once_at_upload(settings):
    market = MarketData.from_settings(settings)
    market.gdelt = StubGdelt()
    market.classifier = classifier = KeywordClassifier()

    meta = market.register_document(
        owner="alice",
        symbol="THYAO",
        raw=text_pdf("A subtle payload hides here."),
        filename="k.pdf",
    )
    calls_after_upload = classifier.calls
    payload = market.disclosure("THYAO", owner="alice", document_id=meta["document_id"])
    market.disclosure("THYAO", owner="alice", document_id=meta["document_id"])

    assert meta["injection_flags"] == ["classifier"]
    assert QUARANTINE_MARKER in payload["content"]
    assert "payload" not in payload["content"]
    assert classifier.calls == calls_after_upload  # not re-run per analysis
