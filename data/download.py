"""Fetch the datasets. No dataset file is ever committed - this script is the only way in.

    python -m data.download              # everything
    python -m data.download fatura       # just one
    python -m data.download --verify     # re-check what is already on disk

Downloads are checksum-verified against the Zenodo record before extraction, so a
truncated transfer fails loudly instead of quietly producing a short dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from data.sources import SOURCES, DatasetSource

RAW_DIR = Path(__file__).resolve().parent / "raw"
_CHUNK = 1 << 20


def md5sum(path: Path, chunk: int = _CHUNK) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check against a published sum, not security
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _report(done: int, total: int) -> None:
    if not sys.stderr.isatty() or not total:
        return
    pct = 100 * done / total
    print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)", end="", file=sys.stderr)


def fetch(source: DatasetSource, dest_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Download one archive and verify its checksum. Returns the archive path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / source.filename

    if archive.exists() and not force:
        if md5sum(archive) == source.md5:
            print(f"{source.key}: already present and verified")
            return archive
        print(f"{source.key}: checksum mismatch on disk, re-downloading")

    partial = archive.with_suffix(archive.suffix + ".part")
    print(f"{source.key}: downloading {source.size_bytes / 1e6:.0f} MB from Zenodo ({source.doi})")
    with urllib.request.urlopen(source.url) as response, partial.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length") or source.size_bytes)
        done = 0
        while block := response.read(_CHUNK):
            out.write(block)
            done += len(block)
            _report(done, total)
    print(file=sys.stderr)

    actual = md5sum(partial)
    if actual != source.md5:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{source.key}: checksum mismatch - expected {source.md5}, got {actual}. "
            "The download was truncated or the Zenodo record changed."
        )
    partial.replace(archive)
    return archive


def extract(archive: Path, target: Path, *, force: bool = False) -> Path:
    """Unzip into `target`, skipping if it already holds files."""
    if target.exists() and any(target.iterdir()) and not force:
        print(f"{target.name}: already extracted")
        return target
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    return target


def get(key: str, *, force: bool = False) -> Path:
    """Download and extract one dataset by key. Returns the extracted directory."""
    source = SOURCES[key]
    archive = fetch(source, force=force)
    return extract(archive, RAW_DIR / source.key, force=force)


def verify() -> int:
    """Re-check every archive already on disk. Returns a process exit code."""
    failures = 0
    for source in SOURCES.values():
        archive = RAW_DIR / source.filename
        if not archive.exists():
            print(f"{source.key}: missing")
            failures += 1
            continue
        actual = md5sum(archive)
        ok = actual == source.md5
        print(f"{source.key}: {'ok' if ok else f'CORRUPT ({actual})'}")
        failures += not ok
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=[*SOURCES, []], help="default: all")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument("--verify", action="store_true", help="checksum what is already on disk")
    args = parser.parse_args(argv)

    if args.verify:
        return verify()

    for key in args.datasets or list(SOURCES):
        path = get(key, force=args.force)
        print(f"{key}: ready at {path}")
        print(f"  licence: {SOURCES[key].licence} - cite: {SOURCES[key].citation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
