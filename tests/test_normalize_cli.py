import pymupdf
import pytest

from app.normalize_cli import main
from app.pdf_generator import ContentType, PdfTestDocumentBuilder

_A4_WIDTH, _A4_HEIGHT = pymupdf.paper_size("a4")
_POINT_TOLERANCE = 1.0


def _oversized_scan_pdf(tmp_path, pages: int = 2):
    source_bytes = (
        PdfTestDocumentBuilder()
        .pages(pages)
        .content(ContentType.IMAGE)
        .page_size(700, 950)
        .image_resolution(3000, 4000)
        .build()
    )
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(source_bytes)
    return source_path


def test_cli_writes_normalized_a4_pdf(tmp_path, capsys):
    source_path = _oversized_scan_pdf(tmp_path, pages=2)
    output_path = tmp_path / "normalized.pdf"

    main([str(source_path), str(output_path)])

    assert output_path.exists()
    assert output_path.stat().st_size < source_path.stat().st_size
    doc = pymupdf.open(str(output_path))
    assert doc.page_count == 2
    for page in doc:
        assert page.rect.width == pytest.approx(_A4_WIDTH, abs=_POINT_TOLERANCE)
        assert page.rect.height == pytest.approx(_A4_HEIGHT, abs=_POINT_TOLERANCE)

    out = capsys.readouterr().out
    assert str(source_path) in out
    assert str(output_path) in out


def test_cli_creates_missing_output_directory(tmp_path):
    source_path = _oversized_scan_pdf(tmp_path, pages=1)
    output_path = tmp_path / "nested" / "dir" / "normalized.pdf"

    main([str(source_path), str(output_path)])

    assert output_path.exists()


def test_cli_accepts_mode_and_quality_options(tmp_path):
    source_path = _oversized_scan_pdf(tmp_path, pages=1)
    output_path = tmp_path / "normalized.pdf"

    main([str(source_path), str(output_path), "--mode", "raster", "--dpi", "150", "--quality", "70"])

    assert output_path.exists()


def test_cli_rejects_invalid_mode(tmp_path):
    source_path = _oversized_scan_pdf(tmp_path, pages=1)
    output_path = tmp_path / "normalized.pdf"

    with pytest.raises(SystemExit):
        main([str(source_path), str(output_path), "--mode", "bogus"])
