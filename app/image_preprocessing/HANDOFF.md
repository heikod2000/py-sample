# Handoff: Bild-Vorverarbeitung für die OCRmyPDF-Strecke (keep-ngx)

Stand: 21.09.2026. Übergabe aus einem claude.ai-Chat an Claude Code.
Arbeitsweise: kleine, iterative Schritte; nach jeder Änderung Tests laufen lassen; bei unklarem Kontext
kurz nachfragen, bevor etwas geändert wird; keine Zugangsdaten oder Keys committen.

## Ausgangsproblem

`OSError: encoder error -2 when writing image file` in `PIL/TiffImagePlugin.py` (`_save`, Zeile 2001), Python 3.13.
Aufrufkette: `router/web_ui.py::convert` → `router/webui/conversion.py::convert_document` → `Pipeline.run`
→ `ImgPreprocessingStep.execute` → `ImageColorOperator.run` (Zeile 46: `img.save(buf, format=output_format)`).

**Ursache ist eine Hypothese und nicht verifiziert:** `save()` ohne `compression` erbt `img.info["compression"]`
des Quellbilds (z. B. `group4` bei Scans). Nach `convert("RGB")` passt das nicht mehr zum Modus.
Prüfen: vor der Stelle `print(img.mode, img.info.get("compression"))`. Mit dem neuen Code ist das Problem
umgangen, weil intern PNG geschrieben wird.

## Entscheidungen

- Zwischenformat nach der Farbbehandlung ist **PNG**. Verlustfrei, kein Modus-Ärger, OCRmyPDF/img2pdf verarbeitet es.
- Es wird immer nur **ein Bild** verarbeitet (keine Multi-Page-TIFFs).
- Modus `1` (Bilevel) und `L` (Graustufen) bleiben erhalten, kein Aufblasen auf RGB.
- **Abgeschnittene Bilder werden akzeptiert** (Fallback `LOAD_TRUNCATED_IMAGES`), das Ergebnis wird immer neu kodiert
  und in `process_data` als Logzeile vermerkt.
- **EXIF-Orientierung wird in die Pixel eingebrannt.** Bei 90°-Drehungen werden die DPI-Achsen getauscht.
- Intakte JPEG/PNG in Modus 1/L/RGB ohne Drehung werden unverändert durchgereicht (Early-Return).
- Resize: RGB → JPEG (Qualität 85), `1`/`L` → PNG verlustfrei. Modus `1` wird vor dem Skalieren zu `L`
  (Pillow erzwingt bei `1` sonst NEAREST). ICC-Profile werden nur eingebettet, wenn der Farbraum zum Modus passt.
- `_RESAMPLING` im Resize ist `BICUBIC` wie im Original (dort mit TODO). In einem Zwischenstand stand `LANCZOS`. Nicht entschieden.

## Dateien (Ziel im Repo)

| Datei hier | Ziel |
|---|---|
| `pillow_tools_additions.py` | in `app/pipeline/utils/pillow_tools.py` einarbeiten; `load_image_with_pillow` ersetzen, Rest ist neu |
| `image_color_operator.py` | `app/pipeline/steps/img_preprocessing/image_color_operator.py` (ersetzt bestehende Datei) |
| `image_resize_operator.py` | ersetzt `image_resize_operator.py` (Pfad im Repo prüfen) |
| `test_pillow_tools.py`, `test_image_color_operator.py`, `test_image_resize_operator.py` | Testverzeichnis |

Achtung: Von `ImageColorOperator` kannte ich nur Kopf, Docstring, `BACKGROUND` und die `run()`-Methode aus den Chat-Ausschnitten.
Imports und Basisklasse habe ich aus der Resize-Datei übernommen. Andere Methoden der Klasse (falls vorhanden) prüfen.

## Teststand

44 Tests grün. Sie liefen in einer Sandbox mit **Stubs** für `app.metrics.event_tracker`, `PipelineOperator`, `ProcessData`,
`ProcessDataLogContext` und `format_bytes_with_delta`, mit Pillow 12.1.1 und nicht mit deiner Python-3.13-Umgebung und nicht gegen die echten Klassen.
Erster Schritt: Tests im echten Repo laufen lassen (`ProcessData` ist in den Tests ein `MagicMock`).

## Offene Punkte

1. `ImgPreprocessingStep.execute`: Nach `process_data.update_content(new_content)` prüfen, ob Dateiname, Mime-Type oder
   `image_metadata` (im Original auskommentiert) noch das alte Format tragen. Ohne Resize kommt jetzt PNG statt Quellformat heraus.
2. Woher kommt `ctx.ocr_options.image_dpi`? Kann der Wert `None` sein? Der Resize fällt jetzt auf 200 zurück, vorher schlug er dann still fehl (EVT04).
3. `ImageColorOperator` hat laut IDE 5 Usages. Alle Aufrufer darauf prüfen, dass sie nicht das Quellformat als Ausgabeformat erwarten.
4. Ist A4-Canvas für kleine Bilder gewollt? Ein kleines Bild landet ungeskaliert auf voller A4-Seite und wird dadurch physisch größer.
   Alternative: `image_dpi` aus dem Bild (`pHYs`) statt globalem Wert nutzen (maßtreu). Entscheidung steht aus.
5. JPEG-Qualität: 85 (vorher 80). Für farbigen Text eher 90. Konstanten `_JPEG_QUALITY`, `_PNG_COMPRESS_LEVEL` liegen im Resize-Modul.
6. Beim Aufruf von `load_image_with_pillow` in anderen Modulen prüfen, ob dort ein `with` möglich ist (Ressourcen).
7. Restrisiko: `LOAD_TRUNCATED_IMAGES` ist prozessweit. Der Lock serialisiert nur die Fallbacks untereinander. Bei parallelen Requests kann ein erster Versuch währenddessen tolerant laden (ein Event weniger, kein Datenrisiko).
8. Performance: Bei aktiviertem Resize wird das Bild zweimal decodiert und dazwischen als PNG kodiert. Nur angehen, wenn es messbar stört
   (Alternative: `Image`-Objekte zwischen den Operatoren durchreichen).

## Nächste Schritte in Claude Code

1. Dateien ins Repo übernehmen, Tests ausführen.
2. Die Hypothese zum Encoder-Fehler an einer echten Problemdatei bestätigen.
3. Offene Punkte 1 bis 3 klären (Lesen von `ProcessData.update_content`, `image_preprocessing_step.py`, Aufrufer).
4. Danach 4 und 5 entscheiden.
