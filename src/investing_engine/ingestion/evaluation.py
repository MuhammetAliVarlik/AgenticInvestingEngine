"""OCR quality metrics."""

from __future__ import annotations

import re


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    """Edit distance with O(min(len(a), len(b))) memory."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit distance / reference length, after whitespace normalisation."""
    reference, hypothesis = _normalise(reference), _normalise(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words, hyp_words = _normalise(reference).split(), _normalise(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    # Map words to single code points so the character-level routine applies.
    vocabulary: dict[str, str] = {}

    def encode(words: list[str]) -> str:
        return "".join(vocabulary.setdefault(w, chr(0xE000 + len(vocabulary))) for w in words)

    return levenshtein(encode(ref_words), encode(hyp_words)) / len(ref_words)
