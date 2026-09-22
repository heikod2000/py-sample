"""Fehlertolerantes Laden von Bildern mit Pillow.

Eigenständige Funktion ohne Abhängigkeit auf den Preprocessing-Operator, weil
`load_image_with_pillow` an mehreren Stellen der Anwendung gebraucht wird,
überall dort, wo Bilddaten aus Bytes geöffnet werden.
"""

import logging
import threading
from contextlib import contextmanager
from io import BytesIO

from PIL import Image, ImageFile, UnidentifiedImageError
from pillow_heif import register_heif_opener

logger = logging.getLogger(__name__)

# Pillow bringt für HEIC/HEIF (typisch bei iPhone-Fotos) standardmäßig keinen
# Decoder mit (Lizenzgründe, libheif ist nicht gebündelt). register_heif_opener()
# meldet den Decoder aus pillow-heif bei Pillow an, danach erkennt Image.open()
# HEIC/HEIF-Bytes wie jedes andere Format. Muss nur einmal beim Modul-Import laufen.
register_heif_opener()

# Pillows LOAD_TRUNCATED_IMAGES-Flag ist ein globaler Modulzustand (nicht pro Bild),
# daher Lock, damit parallele Aufrufe sich nicht gegenseitig das Flag umschalten.
_TRUNCATED_LOCK = threading.Lock()


class ImageLoadError(Exception):
    """Bild konnte auch mit Toleranz-Modus nicht gelesen werden."""


@contextmanager
def _allow_truncated_images():
    """Aktiviert LOAD_TRUNCATED_IMAGES temporär und stellt den vorherigen Wert wieder her."""
    with _TRUNCATED_LOCK:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            yield
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _open_and_load(content: bytes) -> Image.Image:
    img = Image.open(BytesIO(content))
    img.load()  # erzwingt vollständiges Decodieren jetzt statt lazy beim ersten Zugriff
    return img


def load_image_with_pillow(content: bytes) -> Image.Image:
    """Lädt ein Bild vollständig in den Speicher.

    Schlägt das Laden fehl (z. B. weil die Datei abgeschnitten ist), folgt ein
    zweiter Versuch mit `ImageFile.LOAD_TRUNCATED_IMAGES = True`. Bilder, die nur
    so lesbar waren, tragen danach `img.info["truncated"] = True` - Aufrufer
    können das protokollieren oder anders behandeln.

    Wirft `ImageLoadError`, wenn auch der Toleranz-Modus scheitert bzw. die
    Daten von Anfang an kein Bild sind.
    """
    try:
        return _open_and_load(content)
    except (UnidentifiedImageError, Image.DecompressionBombError) as e:
        # Kein Bild bzw. verdächtig riesig: ein zweiter Versuch wäre sinnlos
        logger.warning("Bild nicht ladbar: %s", e)
        raise ImageLoadError(str(e)) from e
    except (OSError, SyntaxError, ValueError, EOFError) as e:
        logger.warning("Fehler beim Laden, versuche erneut im Toleranz-Modus: %s", e)

    try:
        with _allow_truncated_images():
            img = _open_and_load(content)
    except Exception as exc:
        logger.warning("Bild auch im Toleranz-Modus nicht ladbar: %s", exc)
        raise ImageLoadError(str(exc)) from exc

    img.info["truncated"] = True
    return img
