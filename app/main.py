from pathlib import Path

from app.pdf_generator import ContentType, PdfTestDocumentBuilder


def main() -> None:
    output_path = Path(__file__).parent.parent / "output" / "testdokument.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        PdfTestDocumentBuilder()
        .pages(3)
        .content(ContentType.MIXED)
        .title("Beispiel-Testdokument")
        #.attach("anlage.txt", b"Testinhalt der Anlage")
        .sign(
            reason="Beispiel-Signatur",
            location="uvsample04",
            signer_name="Max Mustermann",
            visible=True,
        )
        #.encrypt(user_password="user", owner_password="owner")
        .build_to_file(output_path)
    )

    print(f"PDF erzeugt: {output_path} ({output_path.stat().st_size} Bytes)")


if __name__ == "__main__":
    main()
