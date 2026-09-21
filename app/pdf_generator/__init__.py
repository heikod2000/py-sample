"""Synthetic PDF generator for building unit test fixtures.

Supports configurable page count, embedded attachments, AES-256 encryption,
digital signatures (self-signed test certificates), and text/image/mixed
page content.
"""

from .builder import PdfTestDocumentBuilder
from .content import ContentType
from .signing import SelfSignedCertificate, generate_self_signed_certificate

__all__ = [
    "PdfTestDocumentBuilder",
    "ContentType",
    "SelfSignedCertificate",
    "generate_self_signed_certificate",
]
