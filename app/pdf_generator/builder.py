"""Fluent builder for generating synthetic PDF documents for unit tests."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pikepdf

from .content import MARGIN, PAGE_HEIGHT, PAGE_WIDTH, ContentType, render_pages
from .signing import SelfSignedCertificate, sign_pdf_bytes

_DEFAULT_SIGNATURE_BOX_WIDTH = 200.0
_DEFAULT_SIGNATURE_BOX_HEIGHT = 40.0


@dataclass
class _Attachment:
    filename: str
    data: bytes


@dataclass
class _EncryptionSpec:
    user_password: str = ""
    owner_password: str = ""


@dataclass
class _SignatureSpec:
    field_name: str = "Signature1"
    reason: str | None = None
    location: str | None = None
    signer_name: str | None = None
    certify: bool = False
    certificate: SelfSignedCertificate | None = None
    visible: bool = False
    page: int | None = None
    box: tuple[float, float, float, float] | None = None


class PdfTestDocumentBuilder:
    """Fluent builder that generates PDF test fixtures for unit tests.

    Example:
        pdf_bytes = (
            PdfTestDocumentBuilder()
            .pages(5)
            .content(ContentType.MIXED)
            .attach("notes.txt", b"hello")
            .encrypt(user_password="user", owner_password="owner")
            .sign(reason="Unit test")
            .build_to_bytes()
        )
    """

    def __init__(self) -> None:
        self._page_count: int = 1
        self._content_type: ContentType = ContentType.TEXT
        self._title: str = "Testdokument"
        self._page_size: tuple[float, float] = (PAGE_WIDTH, PAGE_HEIGHT)
        self._image_pixel_size: tuple[int, int] = (480, 300)
        self._attachments: list[_Attachment] = []
        self._encryption: _EncryptionSpec | None = None
        self._signature: _SignatureSpec | None = None

    def pages(self, count: int) -> "PdfTestDocumentBuilder":
        if count < 1:
            raise ValueError("count must be >= 1")
        self._page_count = count
        return self

    def content(self, content_type: ContentType | str) -> "PdfTestDocumentBuilder":
        self._content_type = ContentType(content_type)
        return self

    def page_size(self, width: float, height: float) -> "PdfTestDocumentBuilder":
        """Override the page size (PDF points), e.g. to simulate non-A4 scans."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        self._page_size = (width, height)
        return self

    def image_resolution(self, width_px: int, height_px: int) -> "PdfTestDocumentBuilder":
        """Override the pixel size of the synthesized test image, e.g. to simulate an
        oversized scan (only relevant for ContentType.IMAGE / ContentType.MIXED)."""
        if width_px <= 0 or height_px <= 0:
            raise ValueError("width_px and height_px must be > 0")
        self._image_pixel_size = (width_px, height_px)
        return self

    def title(self, title: str) -> "PdfTestDocumentBuilder":
        self._title = title
        return self

    def attach(self, filename: str, data: bytes) -> "PdfTestDocumentBuilder":
        """Add an embedded file attachment ("Anlage") to the document."""
        self._attachments.append(_Attachment(filename=filename, data=data))
        return self

    def encrypt(
        self, user_password: str = "", owner_password: str = ""
    ) -> "PdfTestDocumentBuilder":
        """Encrypt the document (AES-256 by default via pikepdf)."""
        if not user_password and not owner_password:
            raise ValueError("at least one of user_password/owner_password is required")
        self._encryption = _EncryptionSpec(
            user_password=user_password, owner_password=owner_password
        )
        return self

    def sign(
        self,
        *,
        field_name: str = "Signature1",
        reason: str | None = None,
        location: str | None = None,
        signer_name: str | None = None,
        certify: bool = False,
        certificate: SelfSignedCertificate | None = None,
        visible: bool = False,
        page: int | None = None,
        box: tuple[float, float, float, float] | None = None,
    ) -> "PdfTestDocumentBuilder":
        """Digitally sign the document using an on-the-fly self-signed test certificate.

        By default the signature is invisible (metadata only). Pass
        ``visible=True`` for a visible signature field showing pyHanko's
        default appearance ("Digitally signed by <signer_name>. Timestamp:
        ..."), rendered as a footer box on the given ``page`` (0-indexed,
        defaults to the last page). Pass ``box`` (``(x1, y1, x2, y2)`` in PDF
        points) for a custom position/size instead of the default footer box;
        supplying ``box`` implies a visible signature.
        """
        self._signature = _SignatureSpec(
            field_name=field_name,
            reason=reason,
            location=location,
            signer_name=signer_name,
            certify=certify,
            certificate=certificate,
            visible=visible or box is not None,
            page=page,
            box=box,
        )
        return self

    def build_to_bytes(self) -> bytes:
        """Render the configured document and return it as PDF bytes.

        Build order: content -> attachments -> signature -> encryption.
        Signing must happen on the unencrypted document (the signature covers
        the file's byte range), so encryption is always applied last.
        """
        pdf_bytes = render_pages(
            self._page_count,
            self._content_type,
            self._title,
            page_size=self._page_size,
            image_pixel_size=self._image_pixel_size,
        )

        if self._attachments:
            pdf_bytes = self._apply_attachments(pdf_bytes)

        if self._signature is not None:
            sig = self._signature
            on_page = sig.page if sig.page is not None else self._page_count - 1
            box = sig.box or (self._default_signature_box() if sig.visible else None)
            pdf_bytes = sign_pdf_bytes(
                pdf_bytes,
                field_name=sig.field_name,
                reason=sig.reason,
                location=sig.location,
                signer_name=sig.signer_name,
                certify=sig.certify,
                certificate=sig.certificate,
                on_page=on_page,
                box=box,
            )

        if self._encryption is not None:
            pdf_bytes = self._apply_encryption(pdf_bytes)

        return pdf_bytes

    def build(self) -> bytes:
        """Alias for :meth:`build_to_bytes`."""
        return self.build_to_bytes()

    def build_to_file(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.build_to_bytes())
        return path

    @staticmethod
    def _default_signature_box() -> tuple[float, float, float, float]:
        """A footer box in the page's bottom-right corner, clear of page content."""
        x2 = PAGE_WIDTH - MARGIN
        x1 = x2 - _DEFAULT_SIGNATURE_BOX_WIDTH
        y1 = 8.0
        y2 = y1 + _DEFAULT_SIGNATURE_BOX_HEIGHT
        return (x1, y1, x2, y2)

    def _apply_attachments(self, pdf_bytes: bytes) -> bytes:
        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            for attachment in self._attachments:
                pdf.attachments[attachment.filename] = attachment.data
            out = io.BytesIO()
            pdf.save(out)
            return out.getvalue()

    def _apply_encryption(self, pdf_bytes: bytes) -> bytes:
        assert self._encryption is not None
        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            out = io.BytesIO()
            pdf.save(
                out,
                encryption=pikepdf.Encryption(
                    user=self._encryption.user_password,
                    owner=self._encryption.owner_password,
                ),
            )
            return out.getvalue()
