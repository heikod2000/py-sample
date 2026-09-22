"""Demo-Einstieg für die Bildvorverarbeitung (OCRmyPDF-Preprocessing).

Erzeugt ein paar typische Beispielbilder (gedrehtes Foto, Bilevel-Scan,
transparenter Screenshot, unvollständig übertragener Scan, HEIC-Foto) und lässt
sie durch `preprocess_image_for_ocr` laufen, um Ergebnis und Logausgaben zu zeigen.
"""

import logging
from io import BytesIO
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw

from app.image_preprocessing.image_preprocessing_operator import preprocess_image_for_ocr


def _rotated_photo_jpeg() -> bytes:
    """Querformat-Foto, physisch quer gespeichert wie von einer Kamera (EXIF orientation 6)."""
    img = Image.new("RGB", (1200, 800), "skyblue")
    ImageDraw.Draw(img).rectangle((100, 100, 500, 400), fill="darkgreen")
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = BytesIO()
    img.save(buf, format="JPEG", exif=exif, dpi=(300, 300))
    return buf.getvalue()


def _bilevel_scan_tiff() -> bytes:
    """Bilevel-Scan mit group4-Kompression - die TIFF-Form des ursprünglichen Bugs."""
    img = Image.new("1", (2480, 3508), 1)  # A4 @ 300dpi, weiß
    ImageDraw.Draw(img).rectangle((300, 300, 2000, 500), fill=0)
    buf = BytesIO()
    img.save(buf, format="TIFF", compression="group4", dpi=(300, 300))
    return buf.getvalue()


def _transparent_screenshot_png() -> bytes:
    """Screenshot-artiges RGBA-Bild mit transparentem Hintergrund."""
    img = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((100, 100, 700, 500), fill=(30, 80, 200, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _truncated_scan_png() -> bytes:
    """Simuliert eine unvollständig übertragene/gespeicherte Bilddatei."""
    full = _transparent_screenshot_png()
    return full[: int(len(full) * 0.6)]


def _heic_photo() -> bytes:
    """iPhone-typisches HEIC-Foto (benötigt pillow-heif, siehe pillow_tools.py)."""
    img = Image.new("RGB", (1200, 1600), "peachpuff")
    ImageDraw.Draw(img).ellipse((300, 400, 900, 1200), fill="indianred")
    buf = BytesIO()
    pillow_heif.from_pillow(img).save(buf, format="HEIF")
    return buf.getvalue()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

    output_dir = Path(__file__).parent.parent / "output" / "image_preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "rotated_photo": _rotated_photo_jpeg(),
        "bilevel_scan": _bilevel_scan_tiff(),
        "transparent_screenshot": _transparent_screenshot_png(),
        "truncated_scan": _truncated_scan_png(),
        "heic_photo": _heic_photo(),
    }

    for name, source_bytes in samples.items():
        print(f"\n=== {name} ===")
        result_bytes = preprocess_image_for_ocr(source_bytes, dpi=200)

        result_format = Image.open(BytesIO(result_bytes)).format.lower()
        extension = "jpg" if result_format == "jpeg" else result_format
        out_path = output_dir / f"{name}.{extension}"
        out_path.write_bytes(result_bytes)
        print(f"Ergebnis: {out_path} ({len(source_bytes):,} -> {len(result_bytes):,} Bytes)")


if __name__ == "__main__":
    main()
