"""Rasterize-based PDF normalization.

Renders a page at a fixed target DPI and re-embeds it, centered and scaled to fit, on
a new A4 page. Intended for scanned pages that have no existing text layer: it
flattens the page to a single compressed image, so any existing vector/text content
is lost. See `normalizer.normalize_to_a4` for a hybrid mode that only rasterizes
pages without a text layer worth preserving.

Defaults (200 DPI / JPEG quality 85) are chosen as an OCR-safe compromise: OCRmyPDF's
underlying Tesseract engine works best around 300 DPI and degrades noticeably below
~150 DPI, while the JPEG quality can stay moderate since OCRmyPDF's own PDF/A output
stage recompresses the image data again anyway.
"""

from __future__ import annotations

import io

import pymupdf
from PIL import Image

from ._geometry import a4_page_size, fit_rect

_POINTS_PER_INCH = 72.0


def normalize_to_a4_raster(
    pdf_bytes: bytes,
    *,
    target_dpi: int = 200,
    jpeg_quality: int = 85,
) -> bytes:
    """Rasterize every page at ``target_dpi`` and re-embed it centered on an A4 page.

    Each source page is rendered to a raster image at ``target_dpi``, JPEG-compressed
    at ``jpeg_quality``, and placed centered on a new A4 page (aspect ratio preserved,
    scaled to fit within the page). Page count is preserved; page geometry, any
    existing text layer, and vector content are not.
    """
    _validate_params(target_dpi, jpeg_quality)

    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()

    for src_page in src:
        _rasterize_page_onto(out, src_page, target_dpi=target_dpi, jpeg_quality=jpeg_quality)

    result = out.tobytes()
    src.close()
    out.close()
    return result


def _validate_params(target_dpi: int, jpeg_quality: int) -> None:
    if target_dpi <= 0:
        raise ValueError("target_dpi must be > 0")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")


def _rasterize_page_onto(
    out_doc: pymupdf.Document,
    src_page: pymupdf.Page,
    *,
    target_dpi: int,
    jpeg_quality: int,
) -> None:
    """Render ``src_page`` at ``target_dpi``, JPEG-compress it, and append it as a new,
    centered A4 page (portrait or landscape, matching the rendered content's
    orientation) to ``out_doc``."""
    zoom = target_dpi / _POINTS_PER_INCH
    matrix = pymupdf.Matrix(zoom, zoom)

    pixmap = src_page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    jpeg_buffer = io.BytesIO()
    image.save(jpeg_buffer, format="JPEG", quality=jpeg_quality)

    page_w, page_h = a4_page_size(pixmap.width, pixmap.height)
    rect = fit_rect(pixmap.width, pixmap.height, page_w, page_h)
    out_page = out_doc.new_page(width=page_w, height=page_h)
    out_page.insert_image(rect, stream=jpeg_buffer.getvalue())
