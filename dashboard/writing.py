"""Safe access to the dashboard's fixed set of local research documents."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from threading import Lock

from . import data

SAVE_LOCK = Lock()
MAX_DOCUMENT_BYTES = 1_048_576
REPO_ROOT = data.REPO_ROOT
BACKUP_ROOT = Path.home() / ".prime-agent" / "kernel-rlm" / "dashboard-writing-backups"


class ConflictError(RuntimeError):
    """The document changed since the caller last read it."""


def _document_path(document_id: str) -> Path:
    if not isinstance(document_id, str) or document_id not in data.WRITING_DOCS:
        raise ValueError("Unknown document id")
    docs_root = REPO_ROOT / "docs" / "research"
    mapped = data.WRITING_DOCS[document_id]
    if not isinstance(mapped, str) or Path(mapped).is_absolute() or len(Path(mapped).parts) != 1:
        raise ValueError("Invalid document allowlist entry")
    root = REPO_ROOT.resolve(strict=True)
    current = root
    for part in ("docs", "research", mapped):
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink documents are not allowed")
    canonical_root = docs_root.resolve(strict=True)
    target = (docs_root / mapped).resolve(strict=True)
    if not target.is_relative_to(canonical_root) or not target.is_relative_to(root):
        raise ValueError("Document path escapes the allowed directory")
    if not target.is_file():
        raise FileNotFoundError(document_id)
    return target


def _read(path: Path) -> bytes:
    content = path.read_bytes()
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document exceeds {MAX_DOCUMENT_BYTES} bytes")
    return content


def _version(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _title(document_id: str) -> str:
    return document_id.replace("-", " ").title()


def list_documents() -> list[dict[str, str]]:
    """Return only documents in the fixed data.WRITING_DOCS allowlist."""
    documents = []
    for document_id in data.WRITING_DOCS:
        path = _document_path(document_id)
        documents.append(
            {
                "id": document_id,
                "title": _title(document_id),
                "path": str(path.relative_to(REPO_ROOT.resolve())),
            }
        )
    return documents


def read_document(document_id: str) -> dict[str, str]:
    """Read one allowlisted document and return its SHA-256 optimistic version."""
    path = _document_path(document_id)
    content = _read(path)
    try:
        markdown = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Document is not valid UTF-8") from error
    return {
        "id": document_id,
        "title": _title(document_id),
        "path": str(path.relative_to(REPO_ROOT.resolve())),
        "markdown": markdown,
        "version": _version(content),
    }


def save_document(document_id: str, markdown: str, version: str) -> dict[str, str]:
    with SAVE_LOCK:
        return _save_document(document_id, markdown, version)


def _save_document(document_id: str, markdown: str, version: str) -> dict[str, str]:
    """Save if version matches; retain prior bytes in private backup storage."""
    if not isinstance(markdown, str):
        raise ValueError("markdown must be a string")
    if (
        not isinstance(version, str)
        or len(version) != 64
        or any(c not in "0123456789abcdef" for c in version)
    ):
        raise ValueError("version must be a SHA-256 hex digest")
    encoded = markdown.encode("utf-8")
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document exceeds {MAX_DOCUMENT_BYTES} bytes")
    path = _document_path(document_id)
    current = _read(path)
    if _version(current) != version:
        raise ConflictError("Document has changed since it was read")
    backup_dir = BACKUP_ROOT.resolve()
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(backup_dir, 0o700)
    backup = backup_dir / f"{document_id}-{version}.md"
    if not backup.exists():
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(current)
            stream.flush()
            os.fsync(stream.fileno())
    if path.is_symlink():
        raise ValueError("Symlink documents are not allowed")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise ValueError("Symlink documents are not allowed")
        if _version(_read(_document_path(document_id))) != version:
            raise ConflictError("Document changed during save")
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return read_document(document_id)
