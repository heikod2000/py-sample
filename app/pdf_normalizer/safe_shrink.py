"""Conservative, opt-in shrinking of oversized images on otherwise text-bearing pages.

Applied only by `normalizer.normalize_to_a4`'s "auto" mode, as a document-wide
post-processing pass after all pages have already been assembled. An image is only
ever touched when ALL of the following hold, so that shrinking it cannot plausibly
damage layout or content elsewhere:

- its page went through the *structural* path — raster-derived pages are already
  rasterized and compressed as a whole; re-touching their single page-image here
  would only cost a redundant JPEG generation for no benefit;
- it is used on exactly one page of the assembled output document — a shared
  resource (e.g. a repeated letterhead logo used on every page) is left alone, since
  sizing it for one page's placement could be wrong for another's;
- it is placed exactly once on that page — an image used more than once at different
  sizes can't be shrunk to a single target resolution without over- or under-sizing
  one of its placements;
- that page carries no annotations — avoids interfering with anything that might
  visually or semantically relate to the image (highlights, stamps, form fields);
- it actually has a colorspace (not a stencil/mask image used for some other effect);
- its effective on-page resolution exceeds `target_dpi`.

Any soft mask (transparency) belonging to a shrunk image is downscaled and rewritten
alongside it at the same new size, so transparency is preserved rather than silently
dropped (the bug an earlier, less careful version of this had).
"""

from __future__ import annotations

from collections import defaultdict

import pymupdf

_POINTS_PER_INCH = 72.0


def shrink_safe_oversized_images(
    out_doc: pymupdf.Document,
    *,
    target_dpi: int,
    jpeg_quality: int,
    eligible_pages: set[int] | None = None,
) -> None:
    """``eligible_pages``: output page numbers that (a) went through the structural
    path and (b) carried no annotations on their *source* page. Must be computed and
    passed by the caller — by the time a structurally-embedded page reaches this
    function, its own annotations (if any) have already been dropped by
    ``show_pdf_page``, so checking the output page itself can never detect them, and
    there is no way to tell a structural page from a raster-derived one here either.
    ``None`` means "assume every page qualifies" (for callers, e.g. tests, that
    already know every page is eligible)."""
    placements: dict[int, list[tuple[int, pymupdf.Rect, int]]] = defaultdict(list)
    for page in out_doc:
        for xref, smask_xref, *_ in page.get_images(full=True):
            for rect in page.get_image_rects(xref):
                placements[xref].append((page.number, rect, smask_xref))

    for xref, uses in placements.items():
        if len(uses) != 1 or len({page_number for page_number, _, _ in uses}) != 1:
            continue  # used more than once, and/or shared across pages

        page_number, rect, smask_xref = uses[0]
        if eligible_pages is not None and page_number not in eligible_pages:
            continue

        pixmap = pymupdf.Pixmap(out_doc, xref)
        if pixmap.colorspace is None:
            continue  # stencil/mask image, not a shrinkable photo

        effective_dpi = pixmap.width * _POINTS_PER_INCH / rect.width
        if effective_dpi <= target_dpi:
            continue

        scale = target_dpi / effective_dpi
        new_width = max(1, round(pixmap.width * scale))
        new_height = max(1, round(pixmap.height * scale))

        _shrink_image_object(out_doc, xref, pixmap, new_width, new_height, jpeg_quality)
        if smask_xref:
            mask_pixmap = pymupdf.Pixmap(out_doc, smask_xref)
            _shrink_image_object(out_doc, smask_xref, mask_pixmap, new_width, new_height, jpeg_quality)


def _shrink_image_object(
    out_doc: pymupdf.Document,
    xref: int,
    pixmap: pymupdf.Pixmap,
    new_width: int,
    new_height: int,
    jpeg_quality: int,
) -> None:
    if pixmap.colorspace.n not in (1, 3):
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    scaled = pymupdf.Pixmap(pixmap, new_width, new_height)

    out_doc.update_stream(
        xref, scaled.tobytes("jpeg", jpg_quality=jpeg_quality), new=False, compress=False
    )
    out_doc.xref_set_key(xref, "Width", str(scaled.width))
    out_doc.xref_set_key(xref, "Height", str(scaled.height))
    out_doc.xref_set_key(xref, "BitsPerComponent", "8")
    out_doc.xref_set_key(xref, "ColorSpace", "/DeviceGray" if scaled.colorspace.n == 1 else "/DeviceRGB")
    out_doc.xref_set_key(xref, "Filter", "/DCTDecode")
    out_doc.xref_set_key(xref, "Decode", "null")
