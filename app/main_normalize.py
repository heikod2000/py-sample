from pathlib import Path

from app.pdf_generator import ContentType, PdfTestDocumentBuilder
from app.pdf_normalizer import normalize_to_a4, normalize_to_a4_raster


def main() -> None:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_sample(
        output_dir,
        name="scan_sample",
        content_type=ContentType.IMAGE,
        title="Scan-Beispiel (reine Bildseiten)",
    )
    _write_sample(
        output_dir,
        name="mixed_sample",
        content_type=ContentType.MIXED,
        title="Mischdokument-Beispiel (Text + Bild)",
    )
    _write_sample(
        output_dir,
        name="landscape_sample",
        content_type=ContentType.MIXED,
        title="Querformat-Beispiel (Text + Bild)",
        page_size=(950, 700),
        image_resolution=(4000, 3000),
    )


def _write_sample(
    output_dir: Path,
    *,
    name: str,
    content_type: ContentType,
    title: str,
    page_size: tuple[float, float] = (700, 950),
    image_resolution: tuple[int, int] = (3000, 4000),
) -> None:
    source_path = output_dir / f"{name}.pdf"
    raster_path = output_dir / f"{name}_normalized_raster.pdf"
    auto_path = output_dir / f"{name}_normalized_auto.pdf"

    # Simulates a batch-scanned document: non-A4 page size, oversized scan images.
    source_bytes = (
        PdfTestDocumentBuilder()
        .pages(3)
        .content(content_type)
        .title(title)
        .page_size(*page_size)
        .image_resolution(*image_resolution)
        .build()
    )
    source_path.write_bytes(source_bytes)

    raster_bytes = normalize_to_a4_raster(source_bytes, target_dpi=200, jpeg_quality=85)
    raster_path.write_bytes(raster_bytes)

    auto_bytes = normalize_to_a4(source_bytes, target_dpi=200, jpeg_quality=85, mode="auto")
    auto_path.write_bytes(auto_bytes)

    print(f"\n=== {name} ===")
    print(f"Quelle:                {source_path} ({len(source_bytes):,} Bytes)")
    print(f"Normalisiert (raster): {raster_path} ({len(raster_bytes):,} Bytes)")
    print(f"Normalisiert (auto):   {auto_path} ({len(auto_bytes):,} Bytes)")


if __name__ == "__main__":
    main()
