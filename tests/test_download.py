"""Download integrity.

A truncated transfer that quietly yields a short dataset is the failure mode this module
exists to prevent, so the checksum path is tested directly rather than trusted.
"""

import hashlib
import io
import zipfile
from dataclasses import replace

import pytest

from data import download
from data.sources import SOURCES, DatasetSource


def _zip_bytes(payload: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in payload.items():
            zf.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def archive_bytes():
    return _zip_bytes({"a.txt": "alpha", "nested/b.txt": "beta"})


@pytest.fixture
def source(archive_bytes):
    return DatasetSource(
        key="demo",
        name="demo",
        url="https://example.invalid/demo.zip",
        filename="demo.zip",
        md5=hashlib.md5(archive_bytes).hexdigest(),
        size_bytes=len(archive_bytes),
        doi="10.0000/demo",
        licence="CC-BY-4.0",
        citation="nobody, nowhere, never",
    )


@pytest.fixture
def serve(monkeypatch):
    """Serve fixed bytes in place of the network."""

    def _serve(payload: bytes):
        class _Response(io.BytesIO):
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()
                return False

        monkeypatch.setattr(download.urllib.request, "urlopen", lambda url: _Response(payload))

    return _serve


class TestMd5:
    def test_matches_hashlib(self, tmp_path, archive_bytes):
        path = tmp_path / "x.zip"
        path.write_bytes(archive_bytes)
        assert download.md5sum(path) == hashlib.md5(archive_bytes).hexdigest()

    def test_reads_files_larger_than_one_chunk(self, tmp_path):
        blob = b"0123456789" * 500_000
        path = tmp_path / "big.bin"
        path.write_bytes(blob)
        assert download.md5sum(path) == hashlib.md5(blob).hexdigest()


class TestFetch:
    def test_downloads_and_verifies(self, tmp_path, source, serve, archive_bytes):
        serve(archive_bytes)
        archive = download.fetch(source, tmp_path)
        assert archive.read_bytes() == archive_bytes

    def test_a_truncated_download_raises_and_leaves_no_archive(
        self, tmp_path, source, serve, archive_bytes
    ):
        serve(archive_bytes[: len(archive_bytes) // 2])
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            download.fetch(source, tmp_path)
        assert not (tmp_path / source.filename).exists()
        assert not list(tmp_path.glob("*.part")), "the partial file must not be left behind"

    def test_an_existing_verified_archive_is_not_refetched(
        self, tmp_path, source, serve, archive_bytes, monkeypatch
    ):
        (tmp_path / source.filename).write_bytes(archive_bytes)
        monkeypatch.setattr(
            download.urllib.request,
            "urlopen",
            lambda url: pytest.fail("should not have hit the network"),
        )
        assert download.fetch(source, tmp_path).exists()

    def test_a_corrupt_archive_on_disk_is_refetched(self, tmp_path, source, serve, archive_bytes):
        (tmp_path / source.filename).write_bytes(b"truncated junk")
        serve(archive_bytes)
        assert download.fetch(source, tmp_path).read_bytes() == archive_bytes


class TestExtract:
    def test_unzips(self, tmp_path, archive_bytes):
        archive = tmp_path / "a.zip"
        archive.write_bytes(archive_bytes)
        target = download.extract(archive, tmp_path / "out")
        assert (target / "a.txt").read_text() == "alpha"
        assert (target / "nested" / "b.txt").read_text() == "beta"

    def test_skips_a_populated_target(self, tmp_path, archive_bytes):
        archive = tmp_path / "a.zip"
        archive.write_bytes(archive_bytes)
        target = tmp_path / "out"
        target.mkdir()
        (target / "a.txt").write_text("edited by hand")
        download.extract(archive, target)
        assert (target / "a.txt").read_text() == "edited by hand"

    def test_force_replaces_the_target(self, tmp_path, archive_bytes):
        archive = tmp_path / "a.zip"
        archive.write_bytes(archive_bytes)
        target = tmp_path / "out"
        target.mkdir()
        (target / "stale.txt").write_text("old")
        download.extract(archive, target, force=True)
        assert not (target / "stale.txt").exists()
        assert (target / "a.txt").read_text() == "alpha"


class TestVerifyCli:
    def test_reports_failure_when_an_archive_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(download, "RAW_DIR", tmp_path)
        assert download.verify() == 1

    def test_reports_success_when_every_archive_matches(self, tmp_path, monkeypatch, source):
        blob = b"x" * 16
        (tmp_path / source.filename).write_bytes(blob)
        verified = replace(source, md5=hashlib.md5(blob).hexdigest())
        monkeypatch.setattr(download, "RAW_DIR", tmp_path)
        monkeypatch.setattr(download, "SOURCES", {verified.key: verified})
        assert download.verify() == 0

    def test_reports_failure_when_an_archive_is_corrupt(self, tmp_path, monkeypatch, source):
        (tmp_path / source.filename).write_bytes(b"not what was published")
        monkeypatch.setattr(download, "RAW_DIR", tmp_path)
        monkeypatch.setattr(download, "SOURCES", {source.key: source})
        assert download.verify() == 1

    def test_main_routes_verify(self, tmp_path, monkeypatch):
        monkeypatch.setattr(download, "RAW_DIR", tmp_path)
        assert download.main(["--verify"]) == 1


class TestSources:
    def test_every_source_is_keyed_by_its_own_key(self):
        assert all(key == src.key for key, src in SOURCES.items())

    def test_every_source_records_licence_and_citation(self):
        for src in SOURCES.values():
            assert src.licence and src.citation and src.doi
            assert len(src.md5) == 32
