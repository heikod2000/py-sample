import io
import os

import pillow_heif
from PIL import Image, ImageCms

from app.image_preprocessing.image_preprocessing_operator import preprocess_image_for_ocr


def _encode(img: Image.Image, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, **kwargs)
    return buf.getvalue()


def _png_bytes(size=200) -> bytes:
    img = Image.frombytes("RGB", (size, size), os.urandom(size * size * 3))
    return _encode(img, format="PNG")


def _out(data: bytes, **kwargs) -> Image.Image:
    return Image.open(io.BytesIO(preprocess_image_for_ocr(data, **kwargs).content))


# --- Laden/Truncated ----------------------------------------------------------
def test_truncated_image_is_tolerated_and_reencoded():
    data = _png_bytes()
    truncated = data[: int(len(data) * 0.6)]

    result = preprocess_image_for_ocr(truncated)

    assert result.content != truncated
    Image.open(io.BytesIO(result.content)).load()  # laedt jetzt ohne Toleranz-Modus


def test_unreadable_input_is_returned_unchanged():
    assert preprocess_image_for_ocr(b"kein bild").content == b"kein bild"


# --- Farbe/Alpha ----------------------------------------------------------------
def test_cmyk_becomes_rgb_jpeg():
    data = _encode(Image.new("CMYK", (20, 20)), format="TIFF")
    out = _out(data)
    assert out.format == "JPEG" and out.mode == "RGB"


def test_gif_with_transparency_is_flattened_to_white():
    src = Image.new("P", (10, 10), 0)
    src.info["transparency"] = 0
    data = _encode(src, format="GIF", transparency=0)
    out = _out(data)
    assert out.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_bilevel_scan_keeps_mode_and_requested_dpi():
    src = Image.new("1", (100, 100))
    data = _encode(src, format="TIFF", compression="group4", dpi=(300, 300))

    out = _out(data, dpi=300)

    assert out.format == "PNG" and out.mode == "1"
    assert round(out.info["dpi"][0]) == 300


def test_icc_profile_survives_when_color_space_matches():
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    data = _encode(Image.new("RGBA", (10, 10)), format="PNG", icc_profile=srgb)
    out = _out(data)
    assert out.info.get("icc_profile")


def test_icc_profile_is_dropped_when_color_space_no_longer_matches():
    """CMYK-Profil passt nach der Konvertierung nach RGB nicht mehr - wird verworfen, nicht mitkonvertiert."""
    fake_cmyk_icc = b"\x00" * 16 + b"CMYK" + b"\x00" * 4
    data = _encode(Image.new("CMYK", (10, 10)), format="TIFF", icc_profile=fake_cmyk_icc)
    out = _out(data)
    assert out.info.get("icc_profile") is None


# --- EXIF + A4-Resize -------------------------------------------------------
def test_rotated_photo_gets_portrait_canvas():
    """Regression: Canvas muss nach der EXIF-Drehung berechnet werden, nicht davor."""
    exif = Image.Exif()
    exif[0x0112] = 6  # gespeichert quer, angezeigt hochkant
    data = _encode(Image.new("RGB", (400, 200)), format="JPEG", exif=exif)

    out = _out(data, dpi=200)

    assert out.size == (1654, 2339)  # A4 hochkant bei 200 dpi


def test_large_rgb_becomes_a4_jpeg():
    data = _encode(Image.new("RGB", (3308, 4678), "white"), format="PNG")
    out = _out(data, dpi=200)
    assert out.format == "JPEG" and out.size == (1654, 2339)
    assert round(out.info["dpi"][0]) == 200


def test_grayscale_scan_stays_lossless_png():
    data = _encode(Image.new("L", (2000, 3000), 255), format="PNG")
    out = _out(data, dpi=200)
    assert out.format == "PNG" and out.mode == "L"


def test_dpi_none_falls_back_to_default():
    data = _encode(Image.new("RGB", (100, 100)), format="PNG")
    out = _out(data, dpi=None)
    assert out.size == (1654, 2339)


def test_heic_input_is_preprocessed_like_any_other_format():
    src = Image.new("RGB", (400, 300), "teal")
    buf = io.BytesIO()
    pillow_heif.from_pillow(src).save(buf, format="HEIF")

    out = _out(buf.getvalue(), dpi=200)

    assert out.format == "JPEG" and out.size == (2339, 1654)


# --- Process-Log --------------------------------------------------------------
def test_process_log_mentions_truncation_and_final_summary():
    data = _png_bytes()
    truncated = data[: int(len(data) * 0.6)]

    result = preprocess_image_for_ocr(truncated)

    assert any("abgeschnitten" in line for line in result.log)
    assert any("Bild vorverarbeitet" in line for line in result.log)


def test_process_log_mentions_exif_rotation_and_dropped_icc():
    exif = Image.Exif()
    exif[0x0112] = 6
    fake_cmyk_icc = b"\x00" * 16 + b"CMYK" + b"\x00" * 4
    data = _encode(Image.new("RGB", (400, 200)), format="JPEG", exif=exif)

    result = preprocess_image_for_ocr(data)
    assert any("EXIF-Orientation 6" in line for line in result.log)

    result_cmyk = preprocess_image_for_ocr(
        _encode(Image.new("CMYK", (10, 10)), format="TIFF", icc_profile=fake_cmyk_icc)
    )
    assert any("ICC-Profil" in line and "verworfen" in line for line in result_cmyk.log)


def test_process_log_is_empty_reasoning_for_untouched_bilevel_page():
    """Ein bereits A4-grosses Bilevel-Bild ohne Drehung/Alpha loest keine Schritt-Meldungen aus."""
    data = _encode(Image.new("1", (1654, 2339)), format="PNG")
    result = preprocess_image_for_ocr(data, dpi=200)
    assert len(result.log) == 1  # nur die Abschluss-Zusammenfassung
    assert "Bild vorverarbeitet" in result.log[0]
