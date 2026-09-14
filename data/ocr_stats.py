"""Report RVL-CDIP OCR quality, the numbers the README quotes.

python -m data.ocr_stats
"""

from __future__ import annotations

from data.download import RAW_DIR
from data.map_rvlcdip import RVLCDIP_REGIONS, load_all, ocr_quality


def main() -> int:
    directory = RAW_DIR / "rvlcdip"
    if not directory.exists():
        raise SystemExit(f"{directory} not found - run `python -m data.download rvlcdip` first")

    quality = ocr_quality(directory)
    print(f"{quality.documents} documents, {quality.words} OCR words")
    print(f"  median per-word confidence : {quality.median:.4f}")
    print(f"  mean   per-word confidence : {quality.mean:.4f}")
    print(f"  words at zero confidence   : {quality.zero_confidence_words}")

    references = load_all(directory)
    print("region text recovered:")
    for region in RVLCDIP_REGIONS:
        found = sum(1 for r in references if r.regions[region])
        print(f"  {region:14s} {found:3d} / {len(references)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
