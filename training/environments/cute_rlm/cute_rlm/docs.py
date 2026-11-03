"""Pinned local documentation loader and bounded search/read REPL bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_docs(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load only files named by a local manifest after validating path and SHA-256."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    import re

    if not isinstance(manifest.get("repo"), str) or not re.fullmatch(
        "[0-9a-f]{40}", str(manifest.get("revision", ""))
    ):
        raise ValueError("docs manifest requires repo and a full pinned commit revision")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("docs manifest requires nonempty files")
    docs = {}
    root = manifest_path.parent
    for entry in entries:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("documentation path escapes manifest directory")
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError("documentation path escapes manifest directory")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"documentation checksum mismatch: {rel}")
        text = payload.decode("utf-8")
        doc_id = entry.get("id", rel.as_posix())
        if not isinstance(doc_id, str) or not doc_id or doc_id in docs:
            raise ValueError("documentation IDs must be unique nonempty strings")
        docs[doc_id] = {
            "repo": manifest["repo"],
            "revision": manifest["revision"],
            "id": doc_id,
            "path": rel.as_posix(),
            "sha256": digest,
            "title": entry.get("title", rel.name),
            "text": text,
        }
    return docs


def docs_bootstrap(docs: dict[str, dict[str, Any]]) -> str:
    """Create source for worker REPL helpers; the model gets bounded search/read, not raw docs."""
    data = {
        doc_id: {"title": doc["title"], "text": doc["text"], "sha256": doc["sha256"]}
        for doc_id, doc in docs.items()
    }
    serialized = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return f"""import json as _cute_json
_CUTE_DOCS = _cute_json.loads({serialized!r})
import re as _cute_re
def search_docs(query, limit=4):
    if not isinstance(query, str) or not query.strip() or len(query) > 300 or type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("query must be 1..300 chars and limit 1..5")
    terms = set(_cute_re.findall(r"[a-zA-Z0-9_]+", query.casefold()))
    scored = []
    for doc_id, doc in _CUTE_DOCS.items():
        body = doc["text"].casefold()
        score = sum(body.count(term) for term in terms)
        if score:
            scored.append((score, doc_id, doc))
    return [{{"id": doc_id, "title": doc["title"], "sha256": doc["sha256"], "excerpt": doc["text"][:900]}} for _, doc_id, doc in sorted(scored, key=lambda row: (-row[0], row[1]))[:limit]]
def read_doc(doc_id, start=0, chars=1200):
    if doc_id not in _CUTE_DOCS or type(start) is not int or start < 0 or type(chars) is not int or not 1 <= chars <= 1600:
        raise ValueError("valid doc ID, nonnegative start, and chars 1..1600 required")
    doc = _CUTE_DOCS[doc_id]
    return {{"id": doc_id, "title": doc["title"], "sha256": doc["sha256"], "excerpt": doc["text"][start:start+chars], "more": start+chars < len(doc["text"])}}
"""


def prepare_docs(root: Path, paths: list[str], *, repo: str, revision: str, output: Path) -> dict:
    """Pin chosen existing documentation files; no network or generated solutions."""
    import re

    if not re.fullmatch("[0-9a-f]{40}", revision) or not repo or not paths:
        raise ValueError("A repository, full commit and explicit document paths are required")
    if output.exists():
        raise FileExistsError(output)
    root = root.resolve()
    if output.parent.resolve() != root:
        raise ValueError("Write the manifest in the documentation root")
    files = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Documentation path escapes root or is missing")
        files.append(
            {
                "id": relative,
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {"repo": repo, "revision": revision, "files": files}
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    load_docs(output)
    return manifest
