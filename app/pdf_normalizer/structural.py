"""Structural (non-rasterizing) PDF page normalization.

Transcludes a source page's original content (text, vector graphics, embedded images,
annotations aside — see limitations below) into a new, centered A4 page via a
page-level transformation matrix, instead of rendering it to a raster image. Text
stays selectable/searchable, and every image is carried over completely untouched
(same bytes, same resolution) — geometry is the only thing that changes.

This is deliberately conservative: images are *not* downscaled/recompressed here, even
if oversized, because a page with real text/layout can have images that are load-
bearing in ways a generic heuristic can't safely judge (overlapping annotations,
watermarks, logos with soft-mask transparency, etc.) — shrinking them risks damaging
content we can't easily verify is safe to touch. Oversized-image shrinking is instead
limited to pages classified as pure scans (see `raster.py`), where flattening the
whole page to a single recompressed image is the norm anyway and carries no such risk
to unrelated content.

Known limitation: PDF annotations (highlights, sticky notes, form fields, links) are
not part of a page's content stream and are therefore not transcluded — they are
silently dropped, both visually and as interactive objects. Not a concern for raw
scans (which typically have none), but relevant if this is ever run on documents that
already carry such elements.
"""

from __future__ import annotations

import pymupdf

from ._geometry import a4_page_size, fit_rect


def _embed_page_structurally_onto(
    out_doc: pymupdf.Document, src_doc: pymupdf.Document, page_index: int
) -> None:
    """Append a new, centered A4 page (portrait or landscape, matching the source
    page's orientation) to ``out_doc`` that transcludes source page ``page_index``
    from ``src_doc`` completely unchanged — text, vectors, and images alike."""
    src_rect = src_doc[page_index].rect
    page_w, page_h = a4_page_size(src_rect.width, src_rect.height)
    rect = fit_rect(src_rect.width, src_rect.height, page_w, page_h)
    out_page = out_doc.new_page(width=page_w, height=page_h)
    out_page.show_pdf_page(rect, src_doc, page_index)
