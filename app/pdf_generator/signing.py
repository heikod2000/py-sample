"""Self-signed test certificates and PDF digital signing via pyHanko."""

from __future__ import annotations

import datetime
import io
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import NameOID
from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
from pyhanko.sign import PdfSignatureMetadata, sign_pdf
from pyhanko.sign.fields import SigFieldSpec
from pyhanko.sign.signers import SimpleSigner


@dataclass(frozen=True)
class SelfSignedCertificate:
    """An in-memory self-signed cert/key pair, only meant for test documents."""

    common_name: str
    certificate: x509.Certificate
    private_key: rsa.RSAPrivateKey

    def as_pkcs12(self) -> bytes:
        return pkcs12.serialize_key_and_certificates(
            name=self.common_name.encode(),
            key=self.private_key,
            cert=self.certificate,
            cas=None,
            encryption_algorithm=serialization.NoEncryption(),
        )


def generate_self_signed_certificate(
    common_name: str = "PDF Test Generator", valid_days: int = 3650
) -> SelfSignedCertificate:
    """Generate a throwaway self-signed RSA cert/key pair for signing test PDFs."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, common_name)]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=valid_days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return SelfSignedCertificate(common_name=common_name, certificate=cert, private_key=key)


def sign_pdf_bytes(
    pdf_bytes: bytes,
    *,
    field_name: str = "Signature1",
    reason: str | None = None,
    location: str | None = None,
    signer_name: str | None = None,
    certify: bool = False,
    certificate: SelfSignedCertificate | None = None,
    on_page: int = 0,
    box: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Sign a PDF (given as bytes) with a self-signed test certificate.

    If no certificate is supplied, a fresh one is generated on the fly.
    Passing ``box`` (a page-relative ``(x1, y1, x2, y2)`` rectangle) makes
    the signature field visible: pyHanko then renders its default appearance
    stamp ("Digitally signed by <signer_name>. Timestamp: ...") into that
    box. Leave ``box`` as ``None`` for an invisible signature (default).
    """
    cert = certificate or generate_self_signed_certificate()
    signer = SimpleSigner.load_pkcs12_data(cert.as_pkcs12(), other_certs=set())

    writer = IncrementalPdfFileWriter(io.BytesIO(pdf_bytes))
    metadata = PdfSignatureMetadata(
        field_name=field_name,
        reason=reason,
        location=location,
        name=signer_name,
        certify=certify,
    )
    output = sign_pdf(
        writer,
        metadata,
        signer=signer,
        new_field_spec=SigFieldSpec(
            sig_field_name=field_name, on_page=on_page, box=box
        ),
    )
    return output.getvalue()
