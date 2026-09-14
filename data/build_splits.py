"""Build and freeze the FATURA splits from the downloaded corpus.

    python -m data.build_splits

Writes `data/splits/*.json`, which ARE committed: the manifests plus their digests are
what make the freeze auditable in git history. Re-running is safe - it refuses to move a
frozen set.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from data.download import RAW_DIR
from data.sources import FATURA
from data.split import SPLIT_DIR, build_splits, manifest_digest, write_manifests

ANNOTATION_PREFIX = "invoices_dataset_final/Annotations/Original_Format/"


def fatura_doc_ids(archive: Path) -> list[str]:
    """Every FATURA document id, read straight from the archive without extracting it."""
    with zipfile.ZipFile(archive) as zf:
        return sorted(
            name[len(ANNOTATION_PREFIX) : -len(".json")]
            for name in zf.namelist()
            if name.startswith(ANNOTATION_PREFIX) and name.endswith(".json")
        )


def main() -> int:
    archive = RAW_DIR / FATURA.filename
    if not archive.exists():
        raise SystemExit(f"{archive} not found - run `python -m data.download fatura` first")

    doc_ids = fatura_doc_ids(archive)
    splits = build_splits(doc_ids)
    write_manifests(splits, SPLIT_DIR)

    print(f"{len(doc_ids)} documents, {len(splits)} splits written to {SPLIT_DIR}")
    for name, split in splits.items():
        mark = "frozen" if split.frozen else "open  "
        print(
            f"  {mark}  {name:12s} {len(split.doc_ids):5d} docs  "
            f"{len(split.layouts):2d} layouts  {manifest_digest(split)[:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
