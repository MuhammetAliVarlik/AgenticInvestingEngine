#!/usr/bin/env python3
"""OCR quality benchmark: character and word error rates on synthetic scans.

Renders Turkish and English disclosure-style passages as scanned pages at
several resolutions, runs them through the production extraction pipeline
and reports CER/WER against the ground truth. Install the Turkish language
pack (``tesseract-ocr-tur``) to evaluate Turkish text properly.

Usage:
    python scripts/ocr_eval.py
"""

from __future__ import annotations

import io
import random
import statistics
from collections.abc import Callable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from investing_engine.ingestion.documents import extract_document, ocr_languages
from investing_engine.ingestion.evaluation import character_error_rate, word_error_rate

PASSAGES = {
    "en-dividend": (
        "MATERIAL EVENT DISCLOSURE\n"
        "The Board of Directors resolved to distribute a gross\n"
        "cash dividend of TRY 2.50 per share for fiscal year 2025.\n"
        "The payment date has been set as 15 May 2026."
    ),
    "tr-temettu": (
        "ÖZEL DURUM AÇIKLAMASI\n"
        "Yönetim Kurulumuz, 2025 yılı karından pay başına\n"
        "brüt 2,50 TL nakit kâr payı dağıtılmasına karar vermiştir.\n"
        "Ödeme tarihi 15 Mayıs 2026 olarak belirlenmiştir."
    ),
    "tr-sermaye": (
        "SERMAYE ARTIRIMI\n"
        "Şirketimizin çıkarılmış sermayesinin 1.380.000.000 TL'den\n"
        "2.760.000.000 TL'ye bedelsiz olarak artırılmasına,\n"
        "Sermaye Piyasası Kurulu onayına sunulmasına karar verilmiştir."
    ),
}


def _noise(image: Image.Image) -> Image.Image:
    rng = random.Random(7)  # noqa: S311 - deterministic test noise, not cryptography
    pixels = image.load()
    for _ in range(image.width * image.height // 60):
        x, y = rng.randrange(image.width), rng.randrange(image.height)
        pixels[x, y] = 0 if rng.random() < 0.5 else 255
    return image


def _jpeg(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=35)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("L")


# Degradations typical of phone photos and office scanners.
CONDITIONS: dict[str, tuple[float, Callable[[Image.Image], Image.Image]]] = {
    "clean-200dpi": (1.0, lambda im: im),
    "clean-150dpi": (0.75, lambda im: im),
    "skew-1.5deg": (1.0, lambda im: im.rotate(1.5, expand=True, fillcolor=255)),
    "blur": (1.0, lambda im: im.filter(ImageFilter.GaussianBlur(1.2))),
    "noise": (1.0, _noise),
    "jpeg-q35": (1.0, _jpeg),
    "combined": (
        0.75,
        lambda im: _jpeg(
            _noise(im.rotate(1.0, expand=True, fillcolor=255).filter(ImageFilter.GaussianBlur(0.8)))
        ),
    ),
}


def unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A font with full Turkish glyph coverage; Pillow's default lacks ş, ğ, İ."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render(text: str, scale: float, degrade: Callable[[Image.Image], Image.Image]) -> bytes:
    font = unicode_font(int(44 * scale))
    width, line_height = int(1900 * scale), int(80 * scale)
    image = Image.new("L", (width, line_height * (text.count("\n") + 3)), color=255)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(text.splitlines()):
        draw.text((int(80 * scale), int(60 * scale) + i * line_height), line, fill=0, font=font)
    buffer = io.BytesIO()
    degrade(image).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    print(f"Tesseract languages: {ocr_languages() or 'none installed'}\n")
    print(f"{'Passage':<14}{'Condition':<14}{'CER':>8}{'WER':>8}{'Conf':>8}")
    cers: list[float] = []
    by_condition: dict[str, list[float]] = {}
    for name, text in PASSAGES.items():
        for label, (scale, degrade) in CONDITIONS.items():
            document = extract_document(render(text, scale, degrade), filename=f"{name}.png")
            hypothesis = document.pages[0].text
            cer = character_error_rate(text, hypothesis)
            wer = word_error_rate(text, hypothesis)
            cers.append(cer)
            by_condition.setdefault(label, []).append(cer)
            confidence = document.pages[0].confidence
            print(f"{name:<14}{label:<14}{cer:>8.2%}{wer:>8.2%}{confidence or 0:>8.1f}")
    print("\nMean CER by condition:")
    for label, values in by_condition.items():
        print(f"  {label:<14}{statistics.mean(values):>8.2%}")
    print(f"Overall mean CER: {statistics.mean(cers):.2%}")


if __name__ == "__main__":
    main()
