import io
import os

import pillow_heif
import pytest
from PIL import Image, ImageFile

from app.image_preprocessing.pillow_tools import ImageLoadError, load_image_with_pillow


def _png_bytes(size=200) -> bytes:
    img = Image.frombytes("RGB", (size, size), os.urandom(size * size * 3))  # Rauschen, nicht komprimierbar
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image_loads_without_truncated_flag():
    assert "truncated" not in load_image_with_pillow(_png_bytes()).info


def test_truncated_image_loads_via_fallback_and_is_flagged():
    data = _png_bytes()
    img = load_image_with_pillow(data[: int(len(data) * 0.6)])
    assert img.info["truncated"] is True


def test_flag_is_restored_after_fallback():
    data = _png_bytes()
    load_image_with_pillow(data[: int(len(data) * 0.6)])
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_garbage_raises_image_load_error():
    with pytest.raises(ImageLoadError):
        load_image_with_pillow(b"kein bild")


def test_heic_image_loads():
    """Regression: HEIC braucht pillow-heif, siehe register_heif_opener() in pillow_tools.py."""
    src = Image.new("RGB", (64, 48), "orange")
    buf = io.BytesIO()
    pillow_heif.from_pillow(src).save(buf, format="HEIF")

    img = load_image_with_pillow(buf.getvalue())

    assert img.size == (64, 48)
