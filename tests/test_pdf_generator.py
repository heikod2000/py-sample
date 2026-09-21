import io

import pikepdf
import pytest
from pypdf import PdfReader
from pyhanko.pdf_utils.reader import PdfFileReader
from pyhanko.sign.validation import validate_pdf_signature

from app.pdf_generator import ContentType, PdfTestDocumentBuilder


def _page_count(pdf_bytes: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


def test_default_document_has_one_text_page():
    pdf_bytes = PdfTestDocumentBuilder().build()

    assert _page_count(pdf_bytes) == 1
    text = PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
    assert "Testdokument" in text


@pytest.mark.parametrize("count", [1, 3, 10])
def test_page_count_is_configurable(count):
    pdf_bytes = PdfTestDocumentBuilder().pages(count).build()

    assert _page_count(pdf_bytes) == count


@pytest.mark.parametrize(
    "content_type", [ContentType.TEXT, ContentType.IMAGE, ContentType.MIXED]
)
def test_content_types_render_without_error(content_type):
    pdf_bytes = PdfTestDocumentBuilder().pages(2).content(content_type).build()

    assert _page_count(pdf_bytes) == 2


def test_mixed_content_page_has_text_and_embedded_image():
    pdf_bytes = PdfTestDocumentBuilder().pages(1).content(ContentType.MIXED).build()

    page = PdfReader(io.BytesIO(pdf_bytes)).pages[0]
    assert "Absatz 1" in page.extract_text()
    assert len(page.images) >= 1


def test_attachments_are_embedded():
    pdf_bytes = (
        PdfTestDocumentBuilder()
        .attach("notes.txt", b"hello world")
        .attach("data.json", b'{"a": 1}')
        .build()
    )

    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        names = set(pdf.attachments.keys())
        assert names == {"notes.txt", "data.json"}
        assert pdf.attachments["notes.txt"].get_file().read_bytes() == b"hello world"


def test_encryption_requires_password_to_open():
    pdf_bytes = (
        PdfTestDocumentBuilder()
        .encrypt(user_password="user-pw", owner_password="owner-pw")
        .build()
    )

    with pytest.raises(pikepdf.PasswordError):
        pikepdf.open(io.BytesIO(pdf_bytes))

    with pikepdf.open(io.BytesIO(pdf_bytes), password="user-pw") as pdf:
        assert len(pdf.pages) == 1
        assert pdf.encryption.user_password == b"user-pw"


def test_encrypt_without_any_password_raises():
    with pytest.raises(ValueError):
        PdfTestDocumentBuilder().encrypt()


def test_signature_is_present_and_valid():
    pdf_bytes = (
        PdfTestDocumentBuilder()
        .pages(2)
        .sign(reason="Unit test", location="Test Suite")
        .build()
    )

    reader = PdfFileReader(io.BytesIO(pdf_bytes))
    assert len(reader.embedded_signatures) == 1

    status = validate_pdf_signature(reader.embedded_signatures[0])
    assert status.intact
    assert status.valid


def test_invisible_signature_has_empty_appearance_rect():
    pdf_bytes = PdfTestDocumentBuilder().pages(1).sign().build_to_bytes()

    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        annot = pdf.pages[0].Annots[0]
        assert [float(v) for v in annot.Rect] == [0, 0, 0, 0]


def test_visible_signature_renders_appearance_on_last_page():
    pdf_bytes = (
        PdfTestDocumentBuilder()
        .pages(3)
        .sign(reason="Sichtbarer Test", signer_name="Max Mustermann", visible=True)
        .build_to_bytes()
    )

    reader = PdfFileReader(io.BytesIO(pdf_bytes))
    status = validate_pdf_signature(reader.embedded_signatures[0])
    assert status.intact
    assert status.valid

    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        assert "/Annots" not in pdf.pages[0]
        assert "/Annots" not in pdf.pages[1]
        annot = pdf.pages[2].Annots[0]
        assert len(annot.AP.N.read_bytes()) > 0


def test_visible_signature_with_custom_box_and_page():
    box = (50.0, 700.0, 300.0, 760.0)
    pdf_bytes = (
        PdfTestDocumentBuilder()
        .pages(2)
        .sign(page=0, box=box)
        .build_to_bytes()
    )

    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        annot = pdf.pages[0].Annots[0]
        assert [float(v) for v in annot.Rect] == list(box)


def test_attachments_signature_and_encryption_combined():
    pdf_bytes = (
        PdfTestDocumentBuilder()
        .pages(3)
        .content(ContentType.MIXED)
        .attach("report.txt", b"synthetic test data")
        .sign(field_name="Signature1", reason="Combined test")
        .encrypt(user_password="secret")
        .build()
    )

    with pikepdf.open(io.BytesIO(pdf_bytes), password="secret") as pdf:
        assert len(pdf.pages) == 3
        assert "report.txt" in pdf.attachments.keys()

    reader = PdfFileReader(io.BytesIO(pdf_bytes))
    reader.decrypt("secret")
    assert len(reader.embedded_signatures) == 1


def test_build_to_file_writes_pdf(tmp_path):
    out_path = tmp_path / "generated.pdf"

    result_path = PdfTestDocumentBuilder().pages(2).build_to_file(out_path)

    assert result_path == out_path
    assert out_path.exists()
    assert _page_count(out_path.read_bytes()) == 2
