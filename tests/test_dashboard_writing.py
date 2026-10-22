from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dashboard import writing


def setup_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    docs = repo / "docs" / "research"
    docs.mkdir(parents=True)
    (docs / "guide.md").write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(writing, "REPO_ROOT", repo)
    monkeypatch.setattr(writing, "BACKUP_ROOT", tmp_path / "private" / "backups")
    monkeypatch.setattr(writing.data, "WRITING_DOCS", {"guide": "guide.md"})
    return docs


def test_list_and_read_only_expose_allowlisted_doc(tmp_path, monkeypatch):
    setup_docs(tmp_path, monkeypatch)
    listing = writing.list_documents()
    result = writing.read_document("guide")
    assert listing == [{"id": "guide", "title": "Guide", "path": result["path"]}]
    assert result["markdown"] == "original\n"
    assert result["version"] == hashlib.sha256(b"original\n").hexdigest()


def test_save_uses_versioned_atomic_write_and_private_backup(tmp_path, monkeypatch):
    docs = setup_docs(tmp_path, monkeypatch)
    before = writing.read_document("guide")
    after = writing.save_document("guide", "updated\n", before["version"])
    assert after["markdown"] == "updated\n"
    assert after["version"] == hashlib.sha256(b"updated\n").hexdigest()
    assert (
        tmp_path / "private/backups" / f"guide-{before['version']}.md"
    ).read_bytes() == b"original\n"
    assert list(docs.glob("*.tmp")) == []


def test_save_rejects_stale_version(tmp_path, monkeypatch):
    setup_docs(tmp_path, monkeypatch)
    stale = writing.read_document("guide")["version"]
    writing.save_document("guide", "new\n", stale)
    with pytest.raises(writing.ConflictError):
        writing.save_document("guide", "lost update\n", stale)


def test_invalid_ids_versions_and_size_rejected(tmp_path, monkeypatch):
    setup_docs(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        writing.read_document("../README")
    with pytest.raises(ValueError):
        writing.save_document("guide", "ok", "bogus")
    with pytest.raises(ValueError):
        writing.save_document(
            "guide",
            "x" * (writing.MAX_DOCUMENT_BYTES + 1),
            writing.read_document("guide")["version"],
        )


def test_symlink_target_rejected(tmp_path, monkeypatch):
    docs = setup_docs(tmp_path, monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (docs / "guide.md").unlink()
    (docs / "guide.md").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        writing.read_document("guide")


def test_invalid_utf8_raises(tmp_path, monkeypatch):
    docs = setup_docs(tmp_path, monkeypatch)
    (docs / "guide.md").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        writing.read_document("guide")
