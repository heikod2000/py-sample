"""Command-line entry point for PDF-to-A4 normalization.

Intended as a preprocessing step before OCRmyPDF: normalizes every page onto A4, and
shrinks oversized images on pages that are, for practical purposes, just a scan (no
real text/layout) — the only case where recompressing images is safe by construction.
Pages with real text/layout are only resized, never recompressed.

Usage:
    uv run python -m app.normalize_cli input.pdf output.pdf [--dpi 200] [--quality 85] [--mode auto]

Typical pipeline usage:
    uv run python -m app.normalize_cli scan.pdf normalized.pdf
    ocrmypdf --skip-text normalized.pdf final.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.pdf_normalizer import normalize_to_a4


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a PDF onto uniform A4 pages and shrink oversized images "
            "(e.g. as a preprocessing step before OCRmyPDF)."
        )
    )
    parser.add_argument("input", type=Path, help="source PDF")
    parser.add_argument("output", type=Path, help="destination PDF")
    parser.add_argument(
        "--dpi", type=int, default=200, help="target DPI for shrunk images (default: 200)"
    )
    parser.add_argument(
        "--quality", type=int, default=85, help="JPEG quality, 1-100 (default: 85)"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "raster", "structural"],
        default="auto",
        help=(
            "auto (default): rasterize+shrink pure-scan ('nur-Bilder') pages, only "
            "resize (never recompress) pages with real text/layout; "
            "raster: flatten every page to an image and shrink it; "
            "structural: never rasterize, only resize page geometry, images untouched"
        ),
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=50,
        help="extractable-text threshold for a page to count as 'has text' in auto mode (default: 50)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    input_bytes = args.input.read_bytes()
    output_bytes = normalize_to_a4(
        input_bytes,
        target_dpi=args.dpi,
        jpeg_quality=args.quality,
        mode=args.mode,
        min_text_chars=args.min_text_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)

    ratio = len(output_bytes) / len(input_bytes) if input_bytes else 0.0
    print(
        f"{args.input} ({len(input_bytes):,} Bytes) -> "
        f"{args.output} ({len(output_bytes):,} Bytes, {ratio:.1%})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
