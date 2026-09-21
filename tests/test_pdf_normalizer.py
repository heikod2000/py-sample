import pymupdf
import pytest

from app.pdf_generator import ContentType, PdfTestDocumentBuilder
from app.pdf_normalizer import normalize_to_a4, normalize_to_a4_raster

_A4_WIDTH, _A4_HEIGHT = pymupdf.paper_size("a4")
_POINT_TOLERANCE = 1.0
_LONG_ENOUGH_TEXT = "Some real text on this page, well above the auto-mode min_text_chars threshold."


def _oversized_scan_pdf(pages: int = 2) -> bytes:
    """Simulates a batch-scanned document: non-A4 page size, oversized scan image."""
    return (
        PdfTestDocumentBuilder()
        .pages(pages)
        .content(ContentType.IMAGE)
        .page_size(700, 950)
        .image_resolution(3000, 4000)
        .build()
    )


def _text_pdf(pages: int = 1) -> bytes:
    """Simulates a born-digital page with a real, extractable text layer."""
    return (
        PdfTestDocumentBuilder()
        .pages(pages)
        .content(ContentType.TEXT)
        .page_size(700, 950)
        .build()
    )


def _mixed_pdf(pages: int = 1) -> bytes:
    """Simulates a page with both real text and an embedded (oversized) image."""
    return (
        PdfTestDocumentBuilder()
        .pages(pages)
        .content(ContentType.MIXED)
        .page_size(700, 950)
        .image_resolution(3000, 2000)
        .build()
    )


def test_normalized_pages_are_a4_sized():
    source = _oversized_scan_pdf(pages=3)

    normalized = normalize_to_a4_raster(source)

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    assert doc.page_count == 3
    for page in doc:
        rect = page.rect
        assert rect.width == pytest.approx(_A4_WIDTH, abs=_POINT_TOLERANCE)
        assert rect.height == pytest.approx(_A4_HEIGHT, abs=_POINT_TOLERANCE)


def test_normalization_reduces_file_size():
    source = _oversized_scan_pdf(pages=2)

    normalized = normalize_to_a4_raster(source, target_dpi=200, jpeg_quality=85)

    assert len(normalized) < len(source)


def test_each_page_has_exactly_one_embedded_image():
    source = _oversized_scan_pdf(pages=2)

    normalized = normalize_to_a4_raster(source)

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    for page in doc:
        assert len(page.get_images(full=True)) == 1


def test_higher_target_dpi_yields_higher_resolution_output_image():
    source = _oversized_scan_pdf(pages=1)

    low_dpi = normalize_to_a4_raster(source, target_dpi=100)
    high_dpi = normalize_to_a4_raster(source, target_dpi=300)

    def _embedded_image_pixel_count(pdf_bytes: bytes) -> int:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        xref = doc[0].get_images(full=True)[0][0]
        pixmap = pymupdf.Pixmap(doc, xref)
        return pixmap.width * pixmap.height

    assert _embedded_image_pixel_count(high_dpi) > _embedded_image_pixel_count(low_dpi)


def test_invalid_parameters_raise():
    source = _oversized_scan_pdf(pages=1)

    with pytest.raises(ValueError):
        normalize_to_a4_raster(source, target_dpi=0)

    with pytest.raises(ValueError):
        normalize_to_a4_raster(source, jpeg_quality=0)

    with pytest.raises(ValueError):
        normalize_to_a4_raster(source, jpeg_quality=101)


def test_auto_mode_preserves_text_on_text_pages():
    source = _text_pdf(pages=2)

    normalized = normalize_to_a4(source, mode="auto")

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    assert doc.page_count == 2
    for page in doc:
        assert page.rect.width == pytest.approx(_A4_WIDTH, abs=_POINT_TOLERANCE)
        assert page.rect.height == pytest.approx(_A4_HEIGHT, abs=_POINT_TOLERANCE)
        assert "Absatz" in page.get_text()


def test_auto_mode_rasterizes_pure_scan_pages():
    source = _oversized_scan_pdf(pages=1)

    normalized = normalize_to_a4(source, mode="auto")

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    assert doc[0].get_text().strip() == ""
    assert len(doc[0].get_images(full=True)) == 1
    assert len(normalized) < len(source)


def test_auto_mode_does_not_double_compress_raster_pages():
    # the safe_shrink pass must not re-touch images already produced by the raster
    # path — that would cost a second, pointless JPEG generation
    source = _oversized_scan_pdf(pages=3)

    auto_normalized = normalize_to_a4(source, mode="auto")
    raster_normalized = normalize_to_a4_raster(source)

    assert len(auto_normalized) == len(raster_normalized)


def test_auto_mode_preserves_text_and_still_shrinks_a_safe_oversized_image():
    source = _mixed_pdf(pages=1)

    auto_normalized = normalize_to_a4(source, mode="auto")
    raster_normalized = normalize_to_a4_raster(source)

    auto_doc = pymupdf.open(stream=auto_normalized, filetype="pdf")
    raster_doc = pymupdf.open(stream=raster_normalized, filetype="pdf")
    assert "Absatz" in auto_doc[0].get_text()
    assert raster_doc[0].get_text().strip() == ""

    # the image is used exactly once, on an unannotated page: it passes every
    # safe_shrink check, so auto mode shrinks it despite keeping the text layer
    _, _, img_w, img_h = auto_doc[0].get_images(full=True)[0][:4]
    assert (img_w, img_h) < (3000, 2000)
    assert len(auto_normalized) < len(source)


def test_auto_mode_leaves_image_untouched_on_annotated_page():
    source = _mixed_pdf(pages=1)
    src = pymupdf.open(stream=source, filetype="pdf")
    src[0].add_highlight_annot(pymupdf.Rect(50, 90, 300, 110))
    annotated_source = src.tobytes()

    normalized = normalize_to_a4(annotated_source, mode="auto")

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    _, _, img_w, img_h = doc[0].get_images(full=True)[0][:4]
    assert (img_w, img_h) == (3000, 2000)  # left alone: page has an annotation


def test_auto_mode_leaves_image_shared_across_pages_untouched():
    # simulates a repeated letterhead logo: the same oversized image used on two
    # different pages must not be shrunk based on just one page's placement
    src = pymupdf.open()
    logo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 3000, 2000), False)
    logo.set_rect(logo.irect, (0, 0, 255))
    logo_bytes = logo.tobytes("png")

    for _ in range(2):
        page = src.new_page(width=700, height=950)
        page.insert_text((50, 100), _LONG_ENOUGH_TEXT, fontsize=12)
        page.insert_image(pymupdf.Rect(50, 200, 650, 500), stream=logo_bytes)
    source = src.tobytes()

    normalized = normalize_to_a4(source, mode="auto")

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    for page in doc:
        _, _, img_w, img_h = page.get_images(full=True)[0][:4]
        assert (img_w, img_h) == (3000, 2000)  # left alone: shared across pages


def test_auto_mode_preserves_transparency_of_shrunk_image():
    src = pymupdf.open()
    page = src.new_page(width=700, height=950)
    page.insert_text((50, 100), _LONG_ENOUGH_TEXT, fontsize=12)
    transparent_img = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 3000, 2000), True)
    transparent_img.set_rect(transparent_img.irect, (255, 0, 0, 128))
    page.insert_image(pymupdf.Rect(50, 200, 650, 500), stream=transparent_img.tobytes("png"))
    source = src.tobytes()

    normalized = normalize_to_a4(source, mode="auto")

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    xref, smask_xref, img_w, img_h, *_ = doc[0].get_images(full=True)[0]
    assert (img_w, img_h) < (3000, 2000)  # shrunk
    assert smask_xref != 0  # transparency mask still referenced
    mask_pixmap = pymupdf.Pixmap(doc, smask_xref)
    assert (mask_pixmap.width, mask_pixmap.height) == (img_w, img_h)  # mask resized to match


def test_structural_mode_leaves_embedded_images_byte_for_byte_unchanged():
    source = _mixed_pdf(pages=1)
    src_doc = pymupdf.open(stream=source, filetype="pdf")
    src_xref = src_doc[0].get_images(full=True)[0][0]
    src_image_bytes = src_doc.xref_stream_raw(src_xref)

    normalized = normalize_to_a4(source, mode="structural")

    out_doc = pymupdf.open(stream=normalized, filetype="pdf")
    assert out_doc[0].rect.width == pytest.approx(_A4_WIDTH, abs=_POINT_TOLERANCE)
    assert "Absatz" in out_doc[0].get_text()  # text layer preserved, unlike raster mode

    images = out_doc[0].get_images(full=True)
    assert len(images) == 1
    _, _, img_w, img_h = images[0][:4]
    assert (img_w, img_h) == (3000, 2000)  # oversized source image left untouched

    out_xref = images[0][0]
    assert out_doc.xref_stream_raw(out_xref) == src_image_bytes


def test_unknown_mode_raises():
    source = _text_pdf(pages=1)

    with pytest.raises(ValueError):
        normalize_to_a4(source, mode="bogus")


def _landscape_scan_pdf(pages: int = 1) -> bytes:
    """Simulates a landscape-fed scan: page and image both wider than tall."""
    return (
        PdfTestDocumentBuilder()
        .pages(pages)
        .content(ContentType.IMAGE)
        .page_size(950, 700)
        .image_resolution(4000, 3000)
        .build()
    )


def test_landscape_scan_becomes_landscape_a4_via_raster():
    source = _landscape_scan_pdf()

    normalized = normalize_to_a4_raster(source)

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    page = doc[0]
    assert page.rect.width == pytest.approx(_A4_HEIGHT, abs=_POINT_TOLERANCE)
    assert page.rect.height == pytest.approx(_A4_WIDTH, abs=_POINT_TOLERANCE)


def test_landscape_text_page_becomes_landscape_a4_via_structural():
    source = (
        PdfTestDocumentBuilder()
        .pages(1)
        .content(ContentType.TEXT)
        .page_size(950, 700)
        .build()
    )

    normalized = normalize_to_a4(source, mode="structural")

    doc = pymupdf.open(stream=normalized, filetype="pdf")
    page = doc[0]
    assert page.rect.width == pytest.approx(_A4_HEIGHT, abs=_POINT_TOLERANCE)
    assert page.rect.height == pytest.approx(_A4_WIDTH, abs=_POINT_TOLERANCE)
    assert "Absatz" in page.get_text()


def test_landscape_content_uses_page_more_fully_than_forced_portrait_would():
    """A landscape page fit into a landscape A4 page should end up noticeably larger
    (closer to its natural size) than the same content forced into a portrait A4
    page would be — that's the whole point of choosing the matching orientation."""
    from app.pdf_normalizer._geometry import A4_HEIGHT, A4_WIDTH, fit_rect

    content_w, content_h = 4000, 3000
    landscape_rect = fit_rect(content_w, content_h, A4_HEIGHT, A4_WIDTH)
    portrait_rect = fit_rect(content_w, content_h, A4_WIDTH, A4_HEIGHT)

    landscape_area = landscape_rect.width * landscape_rect.height
    portrait_area = portrait_rect.width * portrait_rect.height
    assert landscape_area > portrait_area * 1.5


def test_portrait_and_mixed_orientation_pages_each_get_matching_a4_orientation():
    portrait_bytes = _text_pdf(pages=1)
    landscape_bytes = _landscape_scan_pdf(pages=1)

    portrait_doc = pymupdf.open(stream=normalize_to_a4(portrait_bytes), filetype="pdf")
    landscape_doc = pymupdf.open(stream=normalize_to_a4(landscape_bytes), filetype="pdf")

    assert portrait_doc[0].rect.width == pytest.approx(_A4_WIDTH, abs=_POINT_TOLERANCE)
    assert landscape_doc[0].rect.width == pytest.approx(_A4_HEIGHT, abs=_POINT_TOLERANCE)
