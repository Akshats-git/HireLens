"""
PDF text extraction.

pdfplumber is tried first for its better layout handling, with PyMuPDF as a
fallback — the two fail on different malformed files, so trying both recovers
documents neither would handle alone.
"""

import io

from loguru import logger

# pdfplumber output shorter than this suggests it recovered headers only, so the
# fallback extractor is worth attempting.
_MIN_USABLE_CHARS = 100


def extract_text(file_bytes: bytes, filename: str = "") -> str:
    """
    Extract plain text from PDF bytes.

    Args:
        file_bytes: Raw PDF content.
        filename: Original filename, used only in log messages.

    Returns:
        Extracted text, empty for scanned documents with no text layer.
    """
    label = filename or "<unnamed>"

    text = _extract_with_pdfplumber(file_bytes, label)
    if len(text) >= _MIN_USABLE_CHARS:
        return text

    fallback = _extract_with_pymupdf(file_bytes, label)
    if len(fallback) > len(text):
        logger.debug(f"PyMuPDF recovered more text than pdfplumber for '{label}'.")
        return fallback

    if not text:
        logger.warning(f"No text extracted from '{label}' — likely a scanned PDF.")
    return text


def _extract_with_pdfplumber(file_bytes: bytes, label: str) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        logger.debug(f"pdfplumber failed on '{label}': {exc}")
        return ""


def _extract_with_pymupdf(file_bytes: bytes, label: str) -> str:
    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            pages = [page.get_text() for page in document]
        return "\n".join(pages).strip()
    except Exception as exc:
        logger.debug(f"PyMuPDF failed on '{label}': {exc}")
        return ""
