"""Text extraction for user-uploaded disclosure documents.

Many KAP attachments (signed letters, financial statements, general-meeting
minutes) are scanned PDFs with no text layer, so extraction is two-stage:

1. Read the embedded text layer with ``pypdf``.
2. Pages without usable text are rendered with ``pypdfium2`` and passed
   through Tesseract OCR (Turkish + English when the language pack is
   installed), recording a per-page confidence score.

PNG and JPEG uploads go straight to OCR. Uploads are untrusted input, so
every stage is bounded: file size, page count, render resolution, decoded
image pixels and OCR wall-clock time.
"""

from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import pypdfium2 as pdfium
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS_PER_PAGE = 40
RENDER_SCALE = 300 / 72  # 300 DPI, the usual sweet spot for Tesseract
MAX_IMAGE_PIXELS = 40_000_000
TESSERACT_CONFIG = "--oem 1 --psm 6"

_PDF_MAGIC = b"%PDF-"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class DocumentError(ValueError):
    """The upload is not an acceptable document. Messages are user-safe."""


class ExtractionMethod(str, Enum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"


@dataclass(frozen=True, slots=True)
class PageText:
    number: int
    text: str
    method: ExtractionMethod
    confidence: float | None = None
    """Mean Tesseract word confidence (0-100) for OCR pages."""


@dataclass(slots=True)
class ExtractedDocument:
    filename: str
    media_type: str
    pages: list[PageText] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def text(self) -> str:
        return "\n\n".join(f"[Page {p.number}]\n{p.text}" for p in self.pages if p.text)

    @property
    def ocr_pages(self) -> int:
        return sum(p.method is ExtractionMethod.OCR for p in self.pages)

    def summary(self) -> dict[str, object]:
        confidences = [p.confidence for p in self.pages if p.confidence is not None]
        return {
            "filename": self.filename,
            "media_type": self.media_type,
            "pages": len(self.pages),
            "ocr_pages": self.ocr_pages,
            "mean_ocr_confidence": (
                round(sum(confidences) / len(confidences), 1) if confidences else None
            ),
            "characters": len(self.text),
            "warnings": self.warnings,
            "seconds": round(self.seconds, 2),
        }


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_pages: int = 30
    ocr_timeout_seconds: float = 30.0


@lru_cache(maxsize=1)
def ocr_languages() -> str:
    """Use Turkish + English when the Turkish pack is installed, else English."""
    try:
        installed = set(pytesseract.get_languages(config=""))
    except (pytesseract.TesseractNotFoundError, OSError):
        return ""
    if {"tur", "eng"} <= installed:
        return "tur+eng"
    return "eng" if "eng" in installed else ""


def detect_media_type(raw: bytes) -> str:
    """Identify the file by its magic bytes; the client's filename and MIME are ignored."""
    if raw.startswith(_PDF_MAGIC):
        return "application/pdf"
    if raw.startswith(_PNG_MAGIC):
        return "image/png"
    if raw.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    raise DocumentError("Unsupported file type: upload a PDF, PNG or JPEG document")


def _normalise(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess(image: Image.Image) -> Image.Image:
    """Clean a page before OCR: greyscale, remove speckle noise, stretch contrast.

    A 3x3 median filter removes the salt-and-pepper noise typical of
    photocopies and fax-quality scans without blurring glyph edges much.
    """
    grey = image.convert("L")
    return ImageOps.autocontrast(grey.filter(ImageFilter.MedianFilter(3)), cutoff=1)


def _ocr(image: Image.Image, *, timeout: float) -> tuple[str, float | None]:
    languages = ocr_languages()
    if not languages:
        raise DocumentError("OCR is not available on this server")
    try:
        data = pytesseract.image_to_data(
            preprocess(image),
            lang=languages,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
            timeout=timeout,
        )
    except RuntimeError as exc:  # pytesseract raises RuntimeError on timeout
        raise DocumentError("OCR timed out on a page") from exc

    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidence = float(data["conf"][i])
        if confidence >= 0:
            confidences.append(confidence)
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    mean = round(sum(confidences) / len(confidences), 1) if confidences else None
    return _normalise(text), mean


def _extract_pdf(raw: bytes, document: ExtractedDocument, limits: ExtractionLimits) -> None:
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise DocumentError("Encrypted PDFs are not supported")
        page_count = len(reader.pages)
    except PdfReadError as exc:
        raise DocumentError("The PDF could not be read") from exc

    if page_count > limits.max_pages:
        document.warnings.append(
            f"Only the first {limits.max_pages} of {page_count} pages were processed"
        )
    pdf: pdfium.PdfDocument | None = None
    try:
        for index in range(min(page_count, limits.max_pages)):
            try:
                text = _normalise(reader.pages[index].extract_text() or "")
            except Exception:  # malformed content streams are common in real filings
                text = ""
            if len(text) >= MIN_TEXT_CHARS_PER_PAGE:
                document.pages.append(PageText(index + 1, text, ExtractionMethod.TEXT_LAYER))
                continue

            if pdf is None:
                pdf = pdfium.PdfDocument(raw)
            bitmap = pdf[index].render(scale=RENDER_SCALE, grayscale=True)
            image = bitmap.to_pil()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                image.thumbnail((6000, 6000))
            ocr_text, confidence = _ocr(image, timeout=limits.ocr_timeout_seconds)
            document.pages.append(PageText(index + 1, ocr_text, ExtractionMethod.OCR, confidence))
    finally:
        if pdf is not None:
            pdf.close()


def _extract_image(raw: bytes, document: ExtractedDocument, limits: ExtractionLimits) -> None:
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # decompression-bomb guard
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            text, confidence = _ocr(image.convert("L"), timeout=limits.ocr_timeout_seconds)
    except Image.DecompressionBombError as exc:
        raise DocumentError("Image is too large to process") from exc
    except OSError as exc:
        raise DocumentError("The image could not be read") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    document.pages.append(PageText(1, text, ExtractionMethod.OCR, confidence))


def extract_document(
    raw: bytes, *, filename: str, limits: ExtractionLimits | None = None
) -> ExtractedDocument:
    """Extract page text from a PDF or image upload.

    Raises:
        DocumentError: for oversized, unsupported, encrypted or unreadable files.
    """
    limits = limits or ExtractionLimits()
    if len(raw) > limits.max_bytes:
        raise DocumentError(f"File exceeds the {limits.max_bytes // (1024 * 1024)} MiB limit")

    started = time.perf_counter()
    document = ExtractedDocument(filename=filename[:120], media_type=detect_media_type(raw))
    if document.media_type == "application/pdf":
        _extract_pdf(raw, document, limits)
    else:
        _extract_image(raw, document, limits)
    document.seconds = time.perf_counter() - started

    low_confidence = [
        p.number for p in document.pages if p.confidence is not None and p.confidence < 60
    ]
    if low_confidence:
        document.warnings.append(f"Low OCR confidence on pages {low_confidence}")
    if not document.text:
        document.warnings.append("No text could be extracted")
    logger.info("Extracted document", extra={"summary": document.summary()})
    return document
