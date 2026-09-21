"""Rendering of PDF page content (text, images, mixed) via reportlab."""

from __future__ import annotations

import io
import random
from enum import Enum

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 50.0

_LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
    "velit esse cillum dolore eu fugiat nulla pariatur."
)


class ContentType(str, Enum):
    """Which kind of content unit test PDFs should contain."""

    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_header(
    c: canvas.Canvas,
    page_num: int,
    page_count: int,
    title: str,
    *,
    width: float = PAGE_WIDTH,
    height: float = PAGE_HEIGHT,
) -> float:
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, height - MARGIN, title)
    c.setFont("Helvetica", 9)
    c.drawRightString(width - MARGIN, height - MARGIN, f"Seite {page_num} / {page_count}")
    c.line(MARGIN, height - MARGIN - 8, width - MARGIN, height - MARGIN - 8)
    return height - MARGIN - 30


def _draw_text_block(
    c: canvas.Canvas, top_y: float, bottom_y: float, page_num: int
) -> None:
    c.setFont("Helvetica", 11)
    lines = _wrap_text(f"[Absatz {page_num}] {_LOREM}", 90)
    y = top_y
    for line in lines:
        if y < bottom_y:
            break
        c.drawString(MARGIN, y, line)
        y -= 14


def _generate_test_image(page_num: int, width_px: int = 480, height_px: int = 300) -> Image.Image:
    """Deterministically synthesize a small gradient+noise test image (no external assets).

    The noise component makes the image behave like a scanned photo rather than a
    perfectly smooth (and thus pathologically Flate-compressible) gradient, which
    matters for fixtures simulating oversized scans.
    """
    seed = page_num * 97 + 13
    top_color = ((seed * 37) % 256, (seed * 59) % 256, (seed * 83) % 256)
    bottom_color = ((seed * 71) % 256, (seed * 43) % 256, (seed * 19) % 256)

    image = Image.new("RGB", (width_px, height_px))
    draw = ImageDraw.Draw(image)
    for y in range(height_px):
        t = y / max(height_px - 1, 1)
        row_color = tuple(
            int(top_color[i] * (1 - t) + bottom_color[i] * t) for i in range(3)
        )
        draw.line([(0, y), (width_px, y)], fill=row_color)

    rng = random.Random(seed)
    noise_tile = Image.new("RGB", (64, 48))
    noise_tile.putdata(
        [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(64 * 48)]
    )
    noise = noise_tile.resize((width_px, height_px), Image.BILINEAR)
    image = Image.blend(image, noise, alpha=0.25)

    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width_px - 1, height_px - 1], outline=(0, 0, 0), width=2)
    draw.text((10, 10), f"Testbild Seite {page_num}", fill=(255, 255, 255))
    return image


def _draw_scaled_image(
    c: canvas.Canvas,
    image: Image.Image,
    top_y: float,
    bottom_y: float,
    *,
    width: float = PAGE_WIDTH,
) -> None:
    img_w, img_h = image.size
    available_w = width - 2 * MARGIN
    available_h = top_y - bottom_y
    scale = min(available_w / img_w, available_h / img_h)
    draw_w, draw_h = img_w * scale, img_h * scale
    x = (width - draw_w) / 2
    y = top_y - draw_h
    c.drawImage(ImageReader(image), x, y, width=draw_w, height=draw_h)


def render_pages(
    page_count: int,
    content_type: ContentType,
    title: str = "Testdokument",
    *,
    page_size: tuple[float, float] = A4,
    image_pixel_size: tuple[int, int] = (480, 300),
) -> bytes:
    """Render a plain (unencrypted, unsigned) multi-page PDF with the given content type.

    ``page_size`` (width, height in PDF points) and ``image_pixel_size`` (width, height
    in pixels of the synthesized test image) default to A4 / the original test-image
    size, but can be overridden to build fixtures that simulate non-A4 pages or
    oversized scan images (e.g. for testing PDF normalization tooling).
    """
    if page_count < 1:
        raise ValueError("page_count must be >= 1")

    width, height = page_size
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=page_size)
    for page_num in range(1, page_count + 1):
        content_top = _draw_header(c, page_num, page_count, title, width=width, height=height)
        if content_type is ContentType.TEXT:
            _draw_text_block(c, content_top, MARGIN, page_num)
        elif content_type is ContentType.IMAGE:
            image = _generate_test_image(page_num, *image_pixel_size)
            _draw_scaled_image(c, image, content_top, MARGIN, width=width)
        else:  # MIXED: text block on top, image below
            text_bottom = content_top - 130
            _draw_text_block(c, content_top, text_bottom, page_num)
            image = _generate_test_image(page_num, *image_pixel_size)
            _draw_scaled_image(c, image, text_bottom - 10, MARGIN, width=width)
        c.showPage()
    c.save()
    return buffer.getvalue()
