"""Shared A4 page geometry helpers for the raster and structural normalization paths."""

from __future__ import annotations

import pymupdf

A4_WIDTH, A4_HEIGHT = pymupdf.paper_size("a4")


def a4_page_size(content_w: float, content_h: float) -> tuple[float, float]:
    """A4 page size (portrait or landscape) matching the content's orientation.

    Fitting a landscape scan into a fixed portrait A4 page wastes most of the page
    (only ~half the height gets used) and effectively halves the achievable
    resolution. Emitting a landscape A4 page for landscape content instead uses the
    full page and keeps content at (close to) its natural size. This intentionally
    does not try to guess and correct the *reading* orientation (e.g. a portrait page
    fed sideways into a scanner) — that requires real content analysis (OCRmyPDF's
    own ``--rotate-pages``/OSD already does this downstream) and any guess we make
    here without it risks silently rotating pages the wrong way.
    """
    if content_w > content_h:
        return A4_HEIGHT, A4_WIDTH
    return A4_WIDTH, A4_HEIGHT


def fit_rect(content_w: float, content_h: float, page_w: float, page_h: float) -> pymupdf.Rect:
    """Centered rectangle that fits a content_w x content_h box into page_w x page_h,
    preserving aspect ratio."""
    scale = min(page_w / content_w, page_h / content_h)
    draw_w, draw_h = content_w * scale, content_h * scale
    x0 = (page_w - draw_w) / 2
    y0 = (page_h - draw_h) / 2
    return pymupdf.Rect(x0, y0, x0 + draw_w, y0 + draw_h)
