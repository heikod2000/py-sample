"""Bildvorverarbeitung für die OCR-Strecke (OCRmyPDF).

Einfache Funktion statt Operator-Klasse. Eine Anbindung an eine echte
Pipeline-Architektur (Basisklasse, Prozessdaten-Objekt fürs Protokoll,
Metrik-Events) würde `preprocess_image_for_ocr` nur noch aufrufen und
die `logger.info`/`logger.warning`-Zeilen unten gegen die dortige
Protokollierung tauschen - fachlich ändert sich nichts.
"""

import logging
from io import BytesIO

from PIL import Image, ImageOps
from PIL.Image import Resampling

from app.image_preprocessing.pillow_tools import ImageLoadError, load_image_with_pillow

logger = logging.getLogger(__name__)

_ORIENTATION_TAG = 0x0112

# A4-Maße in mm
_A4_W_MM = 210.0
_A4_H_MM = 297.0
_MM_PER_INCH = 25.4
_WHITE = (255, 255, 255)
_DEFAULT_DPI = 200
_JPEG_QUALITY = 85
_PNG_COMPRESS_LEVEL = 3
_RESAMPLING = Resampling.BICUBIC  # TODO: gegen LANCZOS vergleichen (OCR-Ergebnis, Laufzeit)


# --- EXIF --------------------------------------------------------------------
def _exif_orientation(img: Image.Image) -> int:
    """Liefert die EXIF-Orientierung (1 = normal). Defekte EXIF-Daten gelten als 1."""
    try:
        return int(img.getexif().get(_ORIENTATION_TAG, 1))
    except Exception:  # noqa: BLE001 - kaputtes EXIF darf die Verarbeitung nicht stoppen
        return 1


def _apply_exif_orientation(img: Image.Image, orientation: int) -> Image.Image:
    """Dreht/spiegelt die Pixel gemäß EXIF. Bei 90°-Drehungen werden die DPI-Achsen getauscht."""
    if orientation not in range(2, 9):
        return img
    dpi = img.info.get("dpi")
    result = ImageOps.exif_transpose(img)  # Kopie ohne Orientation-Tag
    if dpi and orientation >= 5:
        result.info["dpi"] = (dpi[1], dpi[0])
    return result


# --- Farbe/Alpha ---------------------------------------------------------------
def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info


def _normalize_color(img: Image.Image) -> Image.Image:
    """Bringt ein Bild in 8 Bit, ohne Alpha, Modus 1/L/RGB (OCR-tauglich).

    Modus 1 (Bilevel) und L (Graustufen) bleiben erhalten, kein Aufblasen auf RGB.
    """
    if _has_alpha(img):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, _WHITE)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    if img.mode.startswith("I"):  # I, I;16, I;16L, I;16B
        return img.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    if img.mode in ("1", "L", "RGB"):
        return img
    return img.convert("RGB")


def _matching_icc(icc: bytes | None, mode: str) -> bytes | None:
    """Liefert das ICC-Profil nur, wenn sein Farbraum zum Bildmodus passt (RGB/GRAY)."""
    if not icc or len(icc) < 20:
        return None
    space = icc[16:20]  # Farbraum-Signatur im ICC-Header
    if (mode == "RGB" and space == b"RGB ") or (mode == "L" and space == b"GRAY"):
        return icc
    return None


# --- A4-Resize -----------------------------------------------------------------
def _a4_canvas_size(width: int, height: int, dpi: int) -> tuple[int, int]:
    """A4-Zielgröße in Pixeln, Ausrichtung passend zum Bild."""
    w = round(_A4_W_MM / _MM_PER_INCH * dpi)
    h = round(_A4_H_MM / _MM_PER_INCH * dpi)
    return (w, h) if width <= height else (h, w)


def _scale_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Verkleinert proportional (nie vergrößern). Modus 1 wird vorher zu L, da Pillow dort NEAREST erzwingt."""
    scale = min(1.0, max_w / img.width, max_h / img.height)
    if scale == 1.0:
        return img
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    src = img.convert("L") if img.mode == "1" else img
    return src.resize(size, _RESAMPLING)


def _place_on_white_canvas(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Zentriert das Bild auf weißem Canvas im selben Modus."""
    fill = 255 if img.mode in ("1", "L") else _WHITE
    canvas = Image.new(img.mode, size, fill)
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def _encode(canvas: Image.Image, dpi: int, icc_profile: bytes | None) -> bytes:
    """RGB -> JPEG (Fotos); 1/L -> PNG verlustfrei (Text-Scans)."""
    kwargs: dict = {"dpi": (dpi, dpi)}
    if canvas.mode == "RGB":
        kwargs.update(format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    else:
        kwargs.update(format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
    if icc := _matching_icc(icc_profile, canvas.mode):
        kwargs["icc_profile"] = icc
    buf = BytesIO()
    canvas.save(buf, **kwargs)
    return buf.getvalue()


# --- Öffentliche Funktion --------------------------------------------------
def preprocess_image_for_ocr(content: bytes, dpi: int | None = _DEFAULT_DPI) -> bytes:
    """Bereitet ein Bild für die OCR-Strecke (OCRmyPDF) vor:

    1. Laden, abgeschnittene Bilder werden toleriert (`pillow_tools.load_image_with_pillow`)
    2. EXIF-Ausrichtung in die Pixel einbrennen
    3. Farbmodus normalisieren: Alpha entfernen, auf 1/L/RGB bringen, ICC-Profil nur
       übernehmen, wenn sein Farbraum zum Ergebnis-Modus passt
    4. Auf A4 verkleinern (nie vergrößern) und zentriert auf weißen Canvas setzen

    RGB wird als JPEG, Graustufen/Bilevel als PNG ausgegeben. Bei nicht behebbaren
    Fehlern wird das Original unverändert zurückgegeben statt eine Exception zu werfen -
    ein einzelnes kaputtes Bild soll nicht die gesamte Dokumentverarbeitung stoppen.

    `dpi=None` (z. B. weil der Aufrufer keinen Wert kennt) fällt auf den Standard zurück,
    statt fehlzuschlagen.
    """
    dpi = dpi or _DEFAULT_DPI
    try:
        img = load_image_with_pillow(content)
    except ImageLoadError as e:
        logger.warning("Preprocessing übersprungen, Bild nicht lesbar: %s", e)
        return content

    try:
        with img:
            src_size = img.size
            if img.info.get("truncated"):
                logger.info("Bildquelle war abgeschnitten, wurde im Toleranz-Modus geladen")

            orientation = _exif_orientation(img)
            if orientation != 1:
                logger.info("EXIF-Orientation %s angewendet", orientation)
            oriented = _apply_exif_orientation(img, orientation)
            icc = oriented.info.get("icc_profile")

            normalized = _normalize_color(oriented)

            a4_size = _a4_canvas_size(normalized.width, normalized.height, dpi)
            scaled = _scale_to_fit(normalized, *a4_size)
            canvas = _place_on_white_canvas(scaled, a4_size)
            result = _encode(canvas, dpi, icc)

        logger.info(
            "Bild vorverarbeitet: %dx%dpx -> %dx%dpx (%d dpi), %d -> %d Bytes",
            *src_size, *a4_size, dpi, len(content), len(result),
        )
        return result
    except Exception as e:  # noqa: BLE001 - ein defektes Einzelbild darf die Verarbeitung nicht stoppen
        logger.warning("Preprocessing übersprungen, unerwarteter Fehler: %s", e)
        return content
