import io

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from investing_engine.ingestion.documents import (
    DocumentError,
    ExtractionLimits,
    ExtractionMethod,
    extract_document,
    ocr_languages,
)
from investing_engine.ingestion.evaluation import character_error_rate, word_error_rate
from tests.documents import SCANNED_TEXT, image_bytes, scanned_page, scanned_pdf, text_pdf

requires_ocr = pytest.mark.skipif(not ocr_languages(), reason="Tesseract is not installed")


def test_text_layer_pdf_is_read_without_ocr():
    body = "Dividend resolution. The Board approved a gross dividend of TRY 2.50 per share."
    document = extract_document(text_pdf(body), filename="kap.pdf")

    assert document.media_type == "application/pdf"
    assert document.pages[0].method is ExtractionMethod.TEXT_LAYER
    assert "dividend of TRY 2.50 per" in document.text
    assert document.summary()["ocr_pages"] == 0


@requires_ocr
def test_scanned_pdf_falls_back_to_ocr_with_low_error_rate():
    document = extract_document(scanned_pdf(), filename="scan.pdf")
    page = document.pages[0]

    assert page.method is ExtractionMethod.OCR
    assert page.confidence is not None
    assert page.confidence > 70
    assert character_error_rate(SCANNED_TEXT, page.text) < 0.05
    assert "2.50" in page.text


@requires_ocr
def test_png_upload_is_ocrd():
    document = extract_document(image_bytes(scanned_page()), filename="scan.png")
    assert document.media_type == "image/png"
    assert word_error_rate(SCANNED_TEXT, document.text.split("\n", 1)[1]) < 0.15


@requires_ocr
def test_page_limit_is_enforced_with_a_warning():
    document = extract_document(
        scanned_pdf(pages=3), filename="long.pdf", limits=ExtractionLimits(max_pages=2)
    )
    assert len(document.pages) == 2
    assert "first 2 of 3 pages" in document.warnings[0]


def test_file_type_is_detected_from_content_not_name():
    with pytest.raises(DocumentError, match="Unsupported file type"):
        extract_document(b"MZ\x90\x00 executable", filename="report.pdf")
    with pytest.raises(DocumentError, match="Unsupported file type"):
        extract_document(b"<html><script>alert(1)</script>", filename="x.png")


def test_oversized_upload_is_rejected():
    with pytest.raises(DocumentError, match="MiB limit"):
        extract_document(
            b"%PDF-" + b"0" * 2048, filename="x.pdf", limits=ExtractionLimits(max_bytes=1024)
        )


def test_encrypted_pdf_is_rejected():
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(text_pdf("secret content"))))
    writer.encrypt("password")
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(DocumentError, match="Encrypted"):
        extract_document(buffer.getvalue(), filename="locked.pdf")


def test_corrupt_pdf_is_rejected():
    with pytest.raises(DocumentError, match="could not be read"):
        extract_document(b"%PDF-1.7\n garbage without structure", filename="bad.pdf")


@requires_ocr
def test_decompression_bomb_is_rejected():
    bomb = Image.new("1", (10_000, 9_000), color=1)  # 90 MP, a few KB as PNG
    with pytest.raises(DocumentError, match="too large"):
        extract_document(image_bytes(bomb), filename="bomb.png")


def test_error_rates():
    assert character_error_rate("dividend", "dividend") == 0
    assert character_error_rate("dividend", "dlvidend") == pytest.approx(1 / 8)
    assert word_error_rate("a b c d", "a x c d") == pytest.approx(0.25)
    assert character_error_rate("", "") == 0
