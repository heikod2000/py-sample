"""Auto-hybrid PDF-to-A4 normalization.

Combines the raster and structural strategies per page: pages without meaningful text
(pure scans / "nur-Bilder") are rasterized and recompressed wholesale (see
`raster.normalize_to_a4_raster`) — the only case where oversized images are shrunk.
Pages with a genuine text layer are normalized structurally instead: geometry only,
text/vectors/images all kept completely unchanged.

Additionally, in "auto" mode only, a second, much narrower pass (`safe_shrink`) also
shrinks oversized images left untouched by the structural path — but only ones that
pass a strict set of safety checks (used exactly once, on a page with no annotations,
see `safe_shrink.py` for the full list) making it implausible that shrinking them
could damage anything else on or around them. Anything not meeting every check is
left alone. ``mode="structural"`` never applies this pass — it stays a pure,
predictable geometry-only normalization.

Rasterizing a page that already has a text layer would also destroy it and force an
unnecessary, error-prone re-OCR — a second, independent reason pages with text take
the structural path in the first place.
"""

from __future__ import annotations

from typing import Literal

import pymupdf

from .raster import _rasterize_page_onto, _validate_params
from .safe_shrink import shrink_safe_oversized_images
from .structural import _embed_page_structurally_onto

Mode = Literal["auto", "raster", "structural"]

_DEFAULT_MIN_TEXT_CHARS = 50


def normalize_to_a4(
    pdf_bytes: bytes,
    *,
    target_dpi: int = 200,
    jpeg_quality: int = 85,
    mode: Mode = "auto",
    min_text_chars: int = _DEFAULT_MIN_TEXT_CHARS,
) -> bytes:
    """Normalize every page of a PDF onto A4, choosing a per-page strategy.

    - ``mode="raster"``: every page is rasterized and its image recompressed at
      ``target_dpi``/``jpeg_quality`` — destroys any existing text layer (equivalent
      to ``raster.normalize_to_a4_raster``).
    - ``mode="structural"``: every page keeps its content completely unchanged (text,
      vectors, images byte-for-byte); only page geometry is normalized to A4.
    - ``mode="auto"`` (default): per page, uses "raster" only when the page has at
      most ``min_text_chars`` characters of extractable text — i.e. is, for practical
      purposes, just an image ("nur-Bilder") and safe to flatten/shrink — otherwise
      "structural" (real text/layout present: resized, not recompressed). Afterwards,
      a narrow, strictly-gated pass additionally shrinks any *remaining* oversized
      image that can be shrunk with no plausible risk to anything else on the page
      (see `safe_shrink.py`); anything not clearly safe is left untouched.
    """
    if mode not in ("auto", "raster", "structural"):
        raise ValueError(f"unknown mode: {mode!r}")
    _validate_params(target_dpi, jpeg_quality)

    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()
    eligible_for_safe_shrink: set[int] = set()

    for src_page in src:
        if mode == "raster" or (
            mode == "auto" and len(src_page.get_text().strip()) <= min_text_chars
        ):
            _rasterize_page_onto(out, src_page, target_dpi=target_dpi, jpeg_quality=jpeg_quality)
            # already rasterized+compressed as a whole; re-touching its image here
            # would only cost a redundant JPEG generation for no benefit
        else:
            _embed_page_structurally_onto(out, src, src_page.number)
            if src_page.first_annot is None:
                eligible_for_safe_shrink.add(src_page.number)

    if mode == "auto":
        shrink_safe_oversized_images(
            out,
            target_dpi=target_dpi,
            jpeg_quality=jpeg_quality,
            eligible_pages=eligible_for_safe_shrink,
        )

    result = out.tobytes()
    src.close()
    out.close()
    return result
