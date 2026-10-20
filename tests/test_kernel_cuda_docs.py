import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def targeted_collection(collection):
    from kernel_rlm.cuda_docs import TARGET_SEEDS

    path, _digest = collection
    corpus = json.loads(path.read_text())
    corpus["target_profile"] = "sm120"
    corpus["pages"].extend({"url": url} for url in TARGET_SEEDS["sm120"])
    corpus["sections"].append(
        {
            "id": "cuda-00001",
            "title": "SM120 occupancy",
            "url": TARGET_SEEDS["sm120"][0],
            "text": "Compute capability 12.0 occupancy reference; other Blackwell targets differ.",
        }
    )
    path.write_text(json.dumps(corpus))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_sm120_metadata_and_retrieval_keep_hardware_scope(targeted_collection):
    from kernel_rlm.cuda_docs import hardware_instruction, load_docs

    path, digest = targeted_collection
    metadata, search, read = load_docs(str(path), digest)
    assert metadata["target_profile"] == "sm120"
    assert metadata["compute_capability"] == "12.0"
    instruction = hardware_instruction(metadata, "12.0")
    assert "RTX PRO 6000 Blackwell" in instruction and "sm_120" in instruction
    assert "SM100/B200" in instruction and "Our L40S" not in instruction
    hit = search("occupancy")[0]
    assert hit["id"] == "cuda-00001" and hit["target_profile"] == "sm120"
    assert "CC12.0" in hit["hardware_scope"]
    assert read(hit["id"])["hardware_scope"] == hit["hardware_scope"]


@pytest.mark.parametrize("architecture", [None, "8.9", "10.0", "12.0;8.9"])
def test_hardware_profile_mismatch_fails_before_model_use(targeted_collection, architecture):
    from kernel_rlm.cuda_docs import docs_manifest, hardware_instruction

    with pytest.raises(ValueError, match="compiler architecture"):
        hardware_instruction(docs_manifest(*targeted_collection), architecture)


def test_sm120_requires_pinned_blackwell_guides_without_ada_target(targeted_collection):
    from kernel_rlm.cuda_docs import ARCHIVE, docs_manifest

    path, _digest = targeted_collection
    corpus = json.loads(path.read_text())
    corpus["pages"].append({"url": ARCHIVE + "ada-tuning-guide/index.html"})
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="Ada"):
        docs_manifest(path)
    corpus["pages"] = corpus["pages"][:1]
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="missing"):
        docs_manifest(path)


def test_generic_docs_instruction_does_not_assume_a_gpu(collection):
    from kernel_rlm.cuda_docs import DOCS_INSTRUCTION, SEEDS, docs_manifest, hardware_instruction

    assert "Our L40S" not in DOCS_INSTRUCTION
    assert not any("ada-tuning" in url for url in SEEDS)
    assert "does not identify the current GPU" in hardware_instruction(
        docs_manifest(*collection), None
    )


def test_rtx_driver_requires_the_targeted_profile(collection, targeted_collection):
    from kernel_trl.rtx_curriculum import require_rtx_documentation

    path, digest = targeted_collection
    assert (
        require_rtx_documentation(SimpleNamespace(cuda_docs=path, cuda_docs_sha256=digest))[
            "target_profile"
        ]
        == "sm120"
    )
    corpus = json.loads(path.read_text())
    corpus.pop("target_profile")
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="SM120"):
        require_rtx_documentation(
            SimpleNamespace(
                cuda_docs=path, cuda_docs_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        )


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


@pytest.mark.parametrize("collection_fixture", ["collection", "targeted_collection"])
def test_document_search_through_real_repl_worker(
    collection_fixture, request, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(ROOT / "training/src"))
    monkeypatch.setenv("PYTHONPATH", f"{ROOT / 'training/src'}:{ENVIRONMENT}:{ROOT}")
    monkeypatch.setenv("RLM_TRAIN_READONLY_CONTEXT", "1")
    from rlm_train.repl.subprocess import SubprocessReplBackend

    path, digest = request.getfixturevalue(collection_fixture)

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
