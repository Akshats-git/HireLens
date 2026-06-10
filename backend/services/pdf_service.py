"""
PDF text extraction.

Tries pdfplumber first (better layout preservation), falls back to PyMuPDF
(handles corrupt/scanned PDFs more gracefully).
"""

import io


def extract_text(file_bytes: bytes, filename: str = "") -> str:
    """
    Extract plain text from PDF bytes.

    Args:
        file_bytes: Raw PDF content.
        filename:   Original filename (used only for logging).

    Returns:
        Extracted text string (may be empty for scanned-only PDFs).
    """
    text = _try_pdfplumber(file_bytes)
    if len(text) > 100:
        return text

    text = _try_pymupdf(file_bytes)
    return text


def _try_pdfplumber(file_bytes: bytes) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""


def _try_pymupdf(file_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [doc[i].get_text() for i in range(doc.page_count)]
        doc.close()
        return "\n".join(pages).strip()
    except Exception:
        return ""
