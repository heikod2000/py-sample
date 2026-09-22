"""Bildvorverarbeitung für die OCR-Strecke (OCRmyPDF).

Einfache Funktion statt Operator-Klasse. Eine Anbindung an eine echte
Pipeline-Architektur (Basisklasse, Prozessdaten-Objekt fürs Protokoll,
Metrik-Events) würde `preprocess_image_for_ocr` nur noch aufrufen und die
`add_log()`-Funktion unten gegen die dortige Protokollierung tauschen -
fachlich ändert sich nichts.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
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

AddLog = Callable[[str], None]


@dataclass
class PreprocessResult:
    """Ergebnis einer Vorverarbeitung: das Bild plus das Process-Log dieser einen Konvertierung.

    `log` ist bewusst eine einfache Liste aus lesbaren Sätzen - ein Platzhalter für die
    spätere Anbindung an das echte Process-Log der Zielanwendung (z. B. ein `ProcessData`-
    Objekt mit `add_logline`). Dafür müsste nur `add_log()` in `preprocess_image_for_ocr`
    dorthin schreiben statt in die lokale Liste; der Rest der Funktion bliebe unverändert.
    """

    content: bytes
    log: list[str] = field(default_factory=list)


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


def _normalize_color(img: Image.Image, add_log: AddLog) -> Image.Image:
    """Bringt ein Bild in 8 Bit, ohne Alpha, Modus 1/L/RGB (OCR-tauglich).

    Modus 1 (Bilevel) und L (Graustufen) bleiben erhalten, kein Aufblasen auf RGB.
    """
    if _has_alpha(img):
        add_log("Farbe: Alpha-Kanal/Transparenz gefunden, auf Weiß geflacht")
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, _WHITE)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    if img.mode.startswith("I"):  # I, I;16, I;16L, I;16B
        add_log(f"Farbe: 16-Bit-Modus [{img.mode}] auf 8-Bit-Graustufen skaliert")
        return img.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    if img.mode in ("1", "L", "RGB"):
        logger.debug("Farbe: Modus [%s] unverändert", img.mode)
        return img
    add_log(f"Farbe: Modus [{img.mode}] zu [RGB] konvertiert")
    return img.convert("RGB")


def _matching_icc(icc: bytes | None, mode: str, add_log: AddLog) -> bytes | None:
    """Liefert das ICC-Profil nur, wenn sein Farbraum zum Bildmodus passt (RGB/GRAY)."""
    if not icc:
        return None
    if len(icc) < 20:
        logger.debug("ICC-Profil zu kurz/beschädigt (%d Bytes), wird ignoriert", len(icc))
        return None
    space = icc[16:20]  # Farbraum-Signatur im ICC-Header
    if (mode == "RGB" and space == b"RGB ") or (mode == "L" and space == b"GRAY"):
        return icc
    add_log(f"ICC-Profil (Farbraum {space!r}) passt nicht mehr zum Ergebnis-Modus [{mode}], wird verworfen")
    return None


# --- A4-Resize -----------------------------------------------------------------
def _a4_canvas_size(width: int, height: int, dpi: int) -> tuple[int, int]:
    """A4-Zielgröße in Pixeln, Ausrichtung passend zum Bild."""
    w = round(_A4_W_MM / _MM_PER_INCH * dpi)
    h = round(_A4_H_MM / _MM_PER_INCH * dpi)
    return (w, h) if width <= height else (h, w)


def _scale_to_fit(img: Image.Image, max_w: int, max_h: int, add_log: AddLog) -> Image.Image:
    """Verkleinert proportional (nie vergrößern). Modus 1 wird vorher zu L, da Pillow dort NEAREST erzwingt."""
    scale = min(1.0, max_w / img.width, max_h / img.height)
    if scale == 1.0:
        logger.debug("Resize: Bild passt bereits in %dx%dpx, keine Verkleinerung nötig", max_w, max_h)
        return img
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    add_log(f"Resize: {img.width}x{img.height}px auf {size[0]}x{size[1]}px verkleinert (Faktor {scale:.2f})")
    src = img.convert("L") if img.mode == "1" else img
    return src.resize(size, _RESAMPLING)


def _place_on_white_canvas(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Zentriert das Bild auf weißem Canvas im selben Modus."""
    fill = 255 if img.mode in ("1", "L") else _WHITE
    canvas = Image.new(img.mode, size, fill)
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def _encode(canvas: Image.Image, dpi: int, icc_profile: bytes | None, add_log: AddLog) -> bytes:
    """RGB -> JPEG (Fotos); 1/L -> PNG verlustfrei (Text-Scans)."""
    kwargs: dict = {"dpi": (dpi, dpi)}
    if canvas.mode == "RGB":
        kwargs.update(format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    else:
        kwargs.update(format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
    if icc := _matching_icc(icc_profile, canvas.mode, add_log):
        kwargs["icc_profile"] = icc
        add_log(f"ICC-Profil eingebettet ({len(icc)} Bytes)")
    logger.debug("Kodierung: %s, %dx%dpx, %d dpi", kwargs["format"], *canvas.size, dpi)
    buf = BytesIO()
    canvas.save(buf, **kwargs)
    return buf.getvalue()


# --- Öffentliche Funktion --------------------------------------------------
def preprocess_image_for_ocr(content: bytes, dpi: int | None = _DEFAULT_DPI) -> PreprocessResult:
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

    Das Ergebnis enthält neben den Bilddaten ein `log`: die für diese eine Konvertierung
    gesammelten Schritte in lesbarer Form, zur genauen Analyse durch den Aufrufer -
    unabhängig vom Standard-Logger, der dieselben Meldungen zusätzlich ausgibt.
    """
    dpi = dpi or _DEFAULT_DPI
    process_log: list[str] = []

    def add_log(message: str, level: int = logging.INFO) -> None:
        """Sammelt eine Zeile im Process-Log dieser Konvertierung und spiegelt sie im Standard-Logger."""
        process_log.append(message)
        logger.log(level, message)

    try:
        img = load_image_with_pillow(content)
    except ImageLoadError as e:
        add_log(f"Preprocessing übersprungen, Bild nicht lesbar: {e}", level=logging.WARNING)
        return PreprocessResult(content=content, log=process_log)

    try:
        with img:
            src_format = img.format
            src_mode = img.mode
            src_size = img.size
            logger.debug("Geladen: %s, Modus [%s], %dx%dpx", src_format, src_mode, *src_size)

            if img.info.get("truncated"):
                add_log("Bildquelle war abgeschnitten, wurde im Toleranz-Modus geladen")

            orientation = _exif_orientation(img)
            if orientation != 1:
                add_log(f"EXIF-Orientation {orientation} angewendet")
            else:
                logger.debug("EXIF-Orientation 1 (normal), keine Drehung nötig")
            oriented = _apply_exif_orientation(img, orientation)
            icc = oriented.info.get("icc_profile")

            normalized = _normalize_color(oriented, add_log)

            a4_size = _a4_canvas_size(normalized.width, normalized.height, dpi)
            scaled = _scale_to_fit(normalized, *a4_size, add_log)
            canvas = _place_on_white_canvas(scaled, a4_size)
            result = _encode(canvas, dpi, icc, add_log)

        result_format = "JPEG" if canvas.mode == "RGB" else "PNG"
        add_log(
            f"Bild vorverarbeitet: [{src_format} {src_mode} {src_size[0]}x{src_size[1]}px] -> "
            f"[{result_format} {canvas.mode} {a4_size[0]}x{a4_size[1]}px] ({dpi} dpi), "
            f"{len(content)} -> {len(result)} Bytes"
        )
        return PreprocessResult(content=result, log=process_log)
    except Exception as e:  # noqa: BLE001 - ein defektes Einzelbild darf die Verarbeitung nicht stoppen
        add_log(f"Preprocessing übersprungen, unerwarteter Fehler: {e}", level=logging.WARNING)
        return PreprocessResult(content=content, log=process_log)
