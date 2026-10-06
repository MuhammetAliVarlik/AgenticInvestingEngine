"""Builders for realistic test documents (text-layer PDFs and scanned pages)."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

SCANNED_TEXT = (
    "MATERIAL EVENT DISCLOSURE\n"
    "The Board of Directors resolved to distribute\n"
    "a gross cash dividend of TRY 2.50 per share.\n"
    "Payment date is 15 May 2026."
)


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


def text_pdf(body: str) -> bytes:
    from weasyprint import HTML

    paragraphs = "".join(f"<p>{line}</p>" for line in body.splitlines())
    return HTML(string=f"<html><body>{paragraphs}</body></html>").write_pdf()


def scanned_page(text: str = SCANNED_TEXT, *, size: tuple[int, int] = (1700, 900)) -> Image.Image:
    """A clean 'scan': black text on white at roughly 200 DPI."""
    image = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(image)
    font = unicode_font(44)
    y = 60
    for line in text.splitlines():
        draw.text((80, y), line, fill=0, font=font)
        y += 80
    return image


def image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def scanned_pdf(pages: int = 1) -> bytes:
    """An image-only PDF (no text layer), like a scanned KAP attachment."""
    images = [scanned_page() for _ in range(pages)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:], resolution=200)
    return buffer.getvalue()
