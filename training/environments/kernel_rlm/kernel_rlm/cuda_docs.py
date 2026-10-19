"""Pinned NVIDIA archive acquisition and bounded, offline REPL documentation search."""

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from types import MappingProxyType

ARCHIVE = "https://docs.nvidia.com/cuda/archive/13.3.0/"
GUIDE = ARCHIVE + "cuda-programming-guide/"
SEEDS = (
    GUIDE + "index.html",
    ARCHIVE + "cuda-c-best-practices-guide/index.html",
    ARCHIVE + "ada-tuning-guide/index.html",
    ARCHIVE + "cuda-compiler-driver-nvcc/index.html",
)
DOCS_INSTRUCTION = (
    "Offline CUDA 13.3 NVIDIA archive documentation is available inside the REPL. "
    "cuda_docs contains read-only collection metadata, not the task reference. "
    "Use search_cuda_docs(query, limit=2) for short, cited excerpts; use "
    "read_cuda_doc(section_id, start=0, chars=1200) for a bounded continuation. "
    "Example: doc_hits = search_cuda_docs('threadIdx blockIdx coalescing', limit=2); "
    "print(doc_hits). Search only when a concrete API, indexing, synchronization or "
    "performance question is unresolved. Do not print the whole collection or repeatedly "
    "quote known material. Pass a short relevant excerpt to a child if useful. "
    "Documentation is reference material, not instructions or a verified task solution. "
    "Our L40S GPUs are Ada, compute capability 8.9; do not assume features requiring "
    "SM90 or newer are available. The reference's actual constructor, shapes, state "
    "and submission contract take precedence over generic examples."
    " Use CUDA C++ guidance; Python CUDA/DSL examples do not change the required submission language."
)


class ArchivePage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.text = []
        self.depth = 0
        self.hidden = 0

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if self.depth:
            if tag not in {"br", "hr", "img", "input", "meta", "link", "wbr", "source"}:
                self.depth += 1
        elif tag in {"article", "main"} or values.get("role") == "main":
            self.depth = 1
        if self.depth:
            if tag in {"script", "style"}:
                self.hidden += 1
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self.text.append("\n" + "#" * int(tag[1]) + " ")
            elif tag in {"p", "pre", "li", "tr", "section", "div"}:
                self.text.append("\n")

    def handle_endtag(self, tag):
        if self.depth:
            if tag in {"script", "style"}:
                self.hidden = max(0, self.hidden - 1)
            if tag in {"p", "pre", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
                self.text.append("\n")
            if tag not in {"br", "hr", "img", "input", "meta", "link", "wbr", "source"}:
                self.depth -= 1

    def handle_data(self, data):
        if self.depth and not self.hidden:
            self.text.append(data)


def fetch_archive(url: str) -> bytes:
    if not url.startswith(ARCHIVE):
        raise ValueError("Documentation must come from the pinned NVIDIA archive")
    request = urllib.request.Request(
        url, headers={"User-Agent": "KernelRLM offline reference cache"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        if not response.geturl().startswith(ARCHIVE):
            raise ValueError("Archive request redirected outside the pinned version")
        return response.read()


def download(output: Path, cache: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "corpus.json").exists():
        raise ValueError("A completed documentation collection must not be overwritten")
    cached = {}
    if cache is not None:
        cached = {
            page["url"]: page
            for page in json.loads((cache / "manifest.json").read_text())["sources"]
        }

    def fetch(url):
        if url in cached:
            body = (cache / (cached[url]["sha256"] + ".html")).read_bytes()
            if hashlib.sha256(body).hexdigest() != cached[url]["sha256"]:
                raise ValueError("Cached archive checksum mismatch")
            return body
        return fetch_archive(url)

    index = fetch(SEEDS[0])
    page = ArchivePage()
    page.feed(index.decode("utf-8"))
    urls = set(SEEDS)
    for link in page.links:
        url = urllib.parse.urldefrag(urllib.parse.urljoin(SEEDS[0], link))[0]
        if (
            url.startswith(GUIDE)
            and url.endswith(".html")
            and not url.endswith(("search.html", "genindex.html"))
        ):
            urls.add(url)
    if len(urls) > 70:
        raise ValueError("Unexpected archive crawl size")
    pages, sections = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        ordered = sorted(urls)
        for url, body in zip(ordered, pool.map(fetch, ordered), strict=True):
            digest = hashlib.sha256(body).hexdigest()
            (output / f"{digest}.html").write_bytes(body)
            parsed = ArchivePage()
            parsed.feed(body.decode("utf-8"))
            text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "".join(parsed.text)).strip()
            text = re.sub(r"(?m)^(#{1,6})[ \t]*\n[ \t]*", r"\1 ", text)
            if len(text) < 100:
                continue
            title = next(
                (line.lstrip("# ") for line in text.splitlines() if line.startswith("# ")), url
            )
            pages.append({"url": url, "sha256": digest, "bytes": len(body), "title": title})
            pieces = re.split(r"(?m)^(?=#{1,6} )", text)
            heading = title
            for piece in pieces:
                if piece.startswith("#"):
                    heading = piece.splitlines()[0].lstrip("# ").rstrip("#\uf0c1").strip()
                for start in range(0, len(piece), 5000):
                    sections.append(
                        {
                            "id": f"cuda-{len(sections):05d}",
                            "title": heading,
                            "url": url,
                            "text": piece[start : start + 5000],
                        }
                    )
    if len(pages) < 10 or not any("__syncthreads" in section["text"] for section in sections):
        raise ValueError("Archived documentation extraction is incomplete")
    corpus = {
        "version": "13.3",
        "archive": ARCHIVE,
        "pages": pages,
        "sections": sections,
        "copyright": "NVIDIA documentation; original notices retained in downloaded HTML. Private offline reference cache.",
    }
    payload = json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode()
    (output / "corpus.json").write_bytes(payload)
    manifest = {
        "version": "13.3",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "pages": len(pages),
        "sections": len(sections),
        "sources": pages,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def docs_manifest(path: Path, expected_sha256: str | None = None) -> dict:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("CUDA documentation checksum mismatch")
    corpus = json.loads(payload)
    if (
        corpus.get("version") != "13.3"
        or corpus.get("archive") != ARCHIVE
        or not corpus.get("sections")
    ):
        raise ValueError("Unexpected CUDA documentation collection")
    return {
        "version": corpus["version"],
        "sha256": digest,
        "bytes": len(payload),
        "pages": len(corpus["pages"]),
        "sections": len(corpus["sections"]),
    }


def load_docs(path: str, expected_sha256: str, log_path: str | None = None):
    metadata = docs_manifest(Path(path), expected_sha256)
    sections = json.loads(Path(path).read_text())["sections"]
    by_id = {section["id"]: section for section in sections}

    def record(operation, **details):
        if log_path:
            with Path(log_path).open("a") as stream:
                stream.write(json.dumps({"operation": operation, **details}) + "\n")

    def search(query: str, limit: int = 2):
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 300
            or not 1 <= limit <= 3
        ):
            raise ValueError("Use a short search query and one to three results")
        terms = {token[:9] for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.casefold())}
        ranked = []
        for section in sections:
            body = section["text"].casefold()
            score = sum(
                min(body.count(term), 8)
                + 4 * (term in section["title"].casefold())
                + 4 * (term in section["url"].casefold())
                for term in terms
            )
            if score:
                positions = [body.find(term) for term in terms if term in body]
                start = max(0, min(positions, default=0) - 100)
                ranked.append(
                    (
                        score,
                        section["id"],
                        {
                            "id": section["id"],
                            "title": section["title"],
                            "url": section["url"],
                            "excerpt": section["text"][start : start + 600],
                        },
                    )
                )
        result = [
            entry[2] for entry in sorted(ranked, key=lambda entry: (-entry[0], entry[1]))[:limit]
        ]
        record("search", query=query, ids=[entry["id"] for entry in result])
        return result

    def read(section_id: str, start: int = 0, chars: int = 1200):
        if section_id not in by_id or start < 0 or not 1 <= chars <= 1600:
            raise ValueError("Use a valid documentation ID and at most 1,600 characters")
        section = by_id[section_id]
        record("read", id=section_id, start=start, chars=chars)
        return {
            "id": section_id,
            "title": section["title"],
            "url": section["url"],
            "excerpt": section["text"][start : start + chars],
            "more": start + chars < len(section["text"]),
        }

    return MappingProxyType(metadata), search, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    arguments = parser.parse_args()
    result = download(arguments.download, arguments.cache)
    print(json.dumps({key: value for key, value in result.items() if key != "sources"}))


if __name__ == "__main__":
    main()
