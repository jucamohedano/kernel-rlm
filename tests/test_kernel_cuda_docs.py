import asyncio
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
ENVIRONMENT = ROOT / "training/environments/kernel_rlm"


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ENVIRONMENT))
    from kernel_rlm.cuda_docs import ARCHIVE

    corpus = {
        "version": "13.3",
        "archive": ARCHIVE,
        "pages": [{"url": ARCHIVE}],
        "sections": [
            {
                "id": "cuda-00000",
                "title": "Synchronization",
                "url": ARCHIVE + "test.html",
                "text": "__syncthreads block synchronization " * 150,
            }
        ],
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_search_is_bounded_cited_and_immutable(collection):
    from kernel_rlm.cuda_docs import load_docs

    path, digest = collection
    metadata, search, read = load_docs(str(path), digest)
    hits = search("__syncthreads")
    assert len(hits) == 1 and len(hits[0]["excerpt"]) <= 600
    assert "/archive/13.3.0/" in hits[0]["url"]
    with pytest.raises(TypeError):
        metadata["version"] = "changed"
    hits[0]["excerpt"] = "mutated"
    assert "__syncthreads" in read("cuda-00000")["excerpt"]
    with pytest.raises(ValueError):
        read("cuda-00000", chars=100000)
    with pytest.raises(ValueError):
        search("x", limit=10)


def test_checksum_mismatch_fails_closed(collection):
    from kernel_rlm.cuda_docs import docs_manifest

    with pytest.raises(ValueError, match="checksum"):
        docs_manifest(collection[0], "wrong")


def test_archive_downloader_rejects_latest_or_external_urls(collection):
    from kernel_rlm.cuda_docs import fetch_archive

    with pytest.raises(ValueError, match="pinned"):
        fetch_archive("https://docs.nvidia.com/cuda/cuda-programming-guide/")


def test_archive_parser_retains_code_and_ignores_scripts(collection):
    from kernel_rlm.cuda_docs import ArchivePage

    parser = ArchivePage()
    parser.feed(
        "<nav>ignored</nav><main><h1>Indexing</h1><pre>int index = threadIdx.x;</pre><script>bad()</script></main>"
    )
    text = "".join(parser.text)
    assert "int index = threadIdx.x;" in text
    assert "ignored" not in text and "bad()" not in text


def test_document_search_through_real_repl_worker(collection, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "training/src"))
    monkeypatch.setenv("PYTHONPATH", f"{ROOT / 'training/src'}:{ENVIRONMENT}:{ROOT}")
    monkeypatch.setenv("RLM_TRAIN_READONLY_CONTEXT", "1")
    from rlm_train.repl.subprocess import SubprocessReplBackend

    path, digest = collection

    async def check():
        backend = SubprocessReplBackend(request_timeout=10)
        await backend.start("http://127.0.0.1:1", "docs-test")
        try:
            await backend.bootstrap(
                "from kernel_rlm.cuda_docs import load_docs\n"
                f"cuda_docs, search_cuda_docs, read_cuda_doc = load_docs({str(path)!r}, {digest!r})"
            )
            result = await backend.execute("hits = search_cuda_docs('__syncthreads'); print(hits)")
            assert "Synchronization" in result.stdout
            assert not result.stderr
            rejected = await backend.execute("cuda_docs = {}")
            assert "read-only" in rejected.stderr
            rejected = await backend.execute("search_cuda_docs = None")
            assert "read-only" in rejected.stderr
        finally:
            await backend.stop()

    asyncio.run(check())
