"""Defences for third-party text entering the model's context.

Headlines and uploaded documents are attacker-controllable. Three layers
are applied before any of that text reaches an agent:

1. **Detection** - a fast heuristic scan (English and Turkish) for
   instruction-override phrasing, role/prompt manipulation, tool-call or
   markup smuggling and exfiltration attempts. An optional model-based
   classifier (Llama Prompt Guard) can be layered on top.
2. **Quarantine** - flagged lines are replaced with a marker, so the payload
   itself never reaches the model while the rest of the content still can.
3. **Spotlighting** - remaining text is wrapped in a per-request random
   boundary the model is told to treat as data only. Any attempt to forge
   that boundary inside the content is neutralised.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

_PATTERNS: tuple[tuple[str, str], ...] = (
    # Instruction override (EN)
    ("override", r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|system|your)\b.{0,30}\b(instructions?|prompts?|rules?|directions?|guidelines?)"),
    ("override", r"\bnew\s+(instructions?|task|rules?)\s*[:\-]"),
    ("override", r"\bfrom\s+now\s+on\b.{0,40}\b(you|respond|answer|act)\b"),
    # Instruction override (TR)
    ("override", r"(önceki|yukarıdaki|tüm|bütün)\s+(talimat|komut|kural|yönerge)\w*\s+(yok\s+say|unut|görmezden\s+gel|dikkate\s+alma)"),
    ("override", r"(talimat|komut|kural)\w*\s+(yok\s+say|unut|görmezden\s+gel)"),
    ("override", r"yeni\s+(talimat|görev)\w*\s*[:\-]"),
    # Role / prompt manipulation
    ("role", r"\byou\s+are\s+(now|no\s+longer)\b"),
    # "acts as the custodian" is normal filing language, so only address-the-model forms count.
    ("role", r"\b(pretend|roleplay)\b.{0,20}\b(you|to\s+be)\b|\byou\s+(must|should|will)\s+now\s+act\b"),
    ("role", r"\b(system|developer)\s*(prompt|message|instruction)s?\b"),
    ("role", r"(artık\s+sen|sen\s+artık|sistem\s+(mesajı|istemi|promptu))"),
    ("role", r"\b(jailbreak|DAN\s+mode|developer\s+mode)\b"),
    # Smuggled structure
    ("markup", r"<\s*/?\s*(system|assistant|user|untrusted_data|instructions?|tool_call)\b"),
    ("markup", r"\[/?(INST|SYS)\]|<<\s*/?SYS\s*>>|<\|(im_start|im_end|eot_id|start_header_id)\|>"),
    ("tool", r"\b(transfer_to_\w+|call\s+the\s+\w+\s+tool|function_call|tool_calls?)\b"),
    # Output steering / exfiltration
    ("steer", r"\b(always|must)\s+(recommend|rate|say|output|answer)\b.{0,30}\b(buy|sell|strong|10/10|1/10)\b"),
    ("steer", r"\b(reveal|print|repeat|show)\b.{0,30}\b(system\s+prompt|instructions|api\s*key|secret|password)\b"),
    ("exfil", r"!\[[^\]]*\]\(https?://|\bhttps?://\S+\?(\S*=)\S*(key|token|secret|prompt)"),
)  # fmt: skip

_COMPILED = tuple((label, re.compile(p, re.IGNORECASE | re.DOTALL)) for label, p in _PATTERNS)
_BOUNDARY_TAG = "untrusted_data"
QUARANTINE_MARKER = "[line removed: possible prompt injection]"


@dataclass(frozen=True, slots=True)
class Finding:
    category: str
    excerpt: str


@dataclass(slots=True)
class ScanResult:
    findings: list[Finding] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def categories(self) -> list[str]:
        return sorted({f.category for f in self.findings})


def scan(text: str) -> ScanResult:
    """Heuristically scan ``text`` for prompt-injection indicators."""
    result = ScanResult()
    for category, pattern in _COMPILED:
        for match in pattern.finditer(text):
            result.findings.append(Finding(category, match.group(0)[:80]))
    return result


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n(?=\s*[-•*])")


def _quarantine_paragraph(paragraph: str, result: ScanResult) -> str:
    """Scan sentence by sentence with soft line wraps joined.

    PDF extraction and OCR wrap lines mid-sentence, so a payload can span
    several lines; scanning (and removing) whole sentences means no tail of
    a flagged sentence survives on the next line.
    """
    if not scan(paragraph).flagged:
        return paragraph  # keep original layout (tables, lists) untouched
    sentences = [s for s in _SENTENCE_END.split(paragraph) if s.strip()]
    findings_before = len(result.findings)
    kept: list[str] = []
    for sentence in sentences:
        joined = " ".join(sentence.split())
        sentence_scan = scan(joined)
        if sentence_scan.flagged:
            result.findings.extend(sentence_scan.findings)
            if not kept or kept[-1] != QUARANTINE_MARKER:
                kept.append(QUARANTINE_MARKER)
        else:
            kept.append(joined)
    if len(result.findings) == findings_before:
        # Indicators only visible across sentence boundaries: drop the paragraph.
        result.findings.extend(scan(paragraph).findings)
        return QUARANTINE_MARKER
    return "\n".join(kept)


def quarantine(text: str) -> tuple[str, ScanResult]:
    """Replace every sentence containing an injection indicator with a marker."""
    result = ScanResult()
    paragraphs = [_quarantine_paragraph(p, result) for p in _PARAGRAPH_BREAK.split(text)]
    return "\n\n".join(paragraphs), result


def new_boundary() -> str:
    return secrets.token_hex(6)


def spotlight(text: str, *, source: str, boundary: str) -> str:
    """Wrap ``text`` in a boundary the model is instructed to treat as data only."""
    # Neutralise anything that looks like our tag so content cannot close
    # the block early or open a fake one.
    safe = re.sub(rf"<\s*/?\s*{_BOUNDARY_TAG}[^>]*>", "[tag removed]", text, flags=re.IGNORECASE)
    safe_source = re.sub(r"[^A-Za-z0-9 _.\-]", "", source)[:60]
    return (
        f'<{_BOUNDARY_TAG} id="{boundary}" source="{safe_source}">\n'
        f"{safe}\n"
        f'</{_BOUNDARY_TAG} id="{boundary}">'
    )


def guard_untrusted(text: str, *, source: str) -> tuple[str, ScanResult]:
    """Quarantine then spotlight third-party text. Returns (safe_text, scan)."""
    cleaned, result = quarantine(text)
    return spotlight(cleaned, source=source, boundary=new_boundary()), result
