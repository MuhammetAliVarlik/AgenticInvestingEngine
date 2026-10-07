"""Deterministic checks on the generated report.

* **Numeric grounding** - every precise figure in the report (decimals or
  four-plus digits) must match a value some tool actually returned, allowing
  only for rounding to the precision shown. Figures the model invented or
  mangled are listed as ungrounded.
* **Scope** - the report may only contain sections for requested symbols.
* **Disclaimer** - the not-investment-advice notice is enforced.

These checks need no extra model calls, so they run on every analysis and
their scores are recorded with the trace.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

DISCLAIMER = (
    "**Disclaimer:** This report is generated for research and educational "
    "purposes and is not investment advice."
)
_DISCLAIMER_PROBE = re.compile(r"not\s+(an?\s+)?investment\s+advice", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^##\s+([A-Z0-9]{2,6})\b", re.MULTILINE)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,})(?![\w])")


@dataclass(slots=True)
class ReportCheck:
    grounding_score: float = 1.0
    checked_figures: int = 0
    ungrounded_figures: list[str] = field(default_factory=list)
    unexpected_symbols: list[str] = field(default_factory=list)
    disclaimer_added: bool = False

    @property
    def passed(self) -> bool:
        return not self.ungrounded_figures and not self.unexpected_symbols

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "grounding_score": self.grounding_score,
            "checked_figures": self.checked_figures,
            "ungrounded_figures": self.ungrounded_figures,
            "unexpected_symbols": self.unexpected_symbols,
            "disclaimer_added": self.disclaimer_added,
        }


def _parse(token: str) -> tuple[float, int]:
    """Return (value, decimals shown) for a number token like ``10,512.40``."""
    clean = token.replace(",", "")
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return float(clean), decimals


# Thousands separated by a space, a no-break space or a thin space
# ("12 374.26") are joined first, so they are checked as one figure.
_SPACED_THOUSANDS = re.compile(r"(?<![\w.,])\d{1,3}(?:[ \u00a0\u202f\u2009]\d{3})+(?!\d)")


def _join_spaced_thousands(text: str) -> str:
    return _SPACED_THOUSANDS.sub(lambda m: re.sub(r"\s", "", m.group(0)), text)


def numbers_in_text(text: str) -> list[float]:
    joined = _join_spaced_thousands(_DATE.sub(" ", text))
    return [_parse(m.group(0))[0] for m in _NUMBER.finditer(joined)]


def collect_facts(values: Iterable[Any]) -> set[float]:
    """Every finite number found in nested tool payloads and tool text."""
    facts: set[float] = set()
    stack = list(values)
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            if math.isfinite(item):
                facts.add(float(item))
        elif isinstance(item, str):
            facts.update(numbers_in_text(item))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return facts


def _is_grounded(value: float, decimals: int, facts: set[float]) -> bool:
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - fact) <= tolerance or abs(-value - fact) <= tolerance for fact in facts)


def check_report(
    report: str, *, requested: Sequence[str], facts: set[float]
) -> tuple[str, ReportCheck]:
    """Validate ``report`` and return it (with disclaimer enforced) plus findings."""
    check = ReportCheck()

    text = _join_spaced_thousands(_DATE.sub(" ", report))
    tokens = [m.group(0) for m in _NUMBER.finditer(text)]
    figures = [(t, *_parse(t)) for t in tokens]
    # Plain integers that are also years or small counts are not "precise figures".
    figures = [f for f in figures if f[2] > 0 or not 1900 <= f[1] <= 2100]
    check.checked_figures = len(figures)
    check.ungrounded_figures = sorted(
        {token for token, value, decimals in figures if not _is_grounded(value, decimals, facts)}
    )
    if figures:
        grounded = len(figures) - sum(t in check.ungrounded_figures for t, _, _ in figures)
        check.grounding_score = round(grounded / len(figures), 3)

    allowed = {s.upper() for s in requested}
    check.unexpected_symbols = sorted(
        {s for s in _SECTION_HEADER.findall(report) if s not in allowed}
    )

    if report and not _DISCLAIMER_PROBE.search(report):
        report = f"{report.rstrip()}\n\n{DISCLAIMER}"
        check.disclaimer_added = True
    return report, check
