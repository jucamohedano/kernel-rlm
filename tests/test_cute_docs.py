"""Functional tests for the bounded documentation helpers used by worker REPLs."""

import hashlib

import pytest
from cute_rlm.docs import docs_bootstrap


def helpers(texts):
    docs = {
        doc_id: {
            "title": doc_id,
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "path": "/private/docs/source.txt",
        }
        for doc_id, text in texts.items()
    }
    namespace = {}
    exec(docs_bootstrap(docs), namespace)
    return namespace["search_docs"], namespace["read_doc"]


def test_deep_symbol_match_beats_common_headers_and_returns_readable_offsets():
    prefix = "// License notice\n" * 100 + "// CuTe documentation\n" * 100
    definition = "def make_layout(shape, stride):\n    return Layout(shape, stride)\n"
    text = prefix + definition + "// More documentation\n" * 100
    search, read = helpers({"a-common": "CuTe\n" * 5000, "z-definition": text})

    hit = search("CuTe make_layout")[0]

    assert hit["id"] == "z-definition"
    assert definition in hit["excerpt"]
    assert hit["start"] > 900
    assert hit["start"] <= len(prefix) < hit["end"]
    assert hit["excerpt"] == text[hit["start"] : hit["end"]]
    assert 0 < hit["end"] - hit["start"] <= 900
    page = read(hit["id"], start=hit["start"], chars=hit["end"] - hit["start"])
    assert page["excerpt"] == hit["excerpt"]
    assert page["sha256"] == hit["sha256"]
    assert set(hit) == {"id", "title", "sha256", "excerpt", "start", "end"}


def test_exact_symbol_tokens_do_not_match_longer_identifiers():
    search, _ = helpers(
        {
            "partial": "make_layout_like remake_layout make_layout2",
            "exact": "def make_layout(shape): pass",
        }
    )

    assert [hit["id"] for hit in search("MAKE_LAYOUT")] == ["exact"]
    assert search("absent_symbol") == []


def test_multi_term_coverage_then_same_line_coverage_beats_frequency():
    search, _ = helpers(
        {
            "a-common": "layout\n" * 5000,
            "b-separated": "layout\n" * 100 + "tensor\n",
            "z-together": "layout tensor\n",
        }
    )

    assert [hit["id"] for hit in search("layout tensor")] == [
        "z-together",
        "b-separated",
        "a-common",
    ]


def test_ties_are_stable_by_document_id_and_first_matching_position():
    text = "padding\n" * 200 + "symbol\n" + "padding\n" * 200 + "symbol\n"
    search, _ = helpers({"z": text, "a": text})
    reverse_search, _ = helpers({"a": text, "z": text})

    hits = search("symbol")

    assert hits == search("symbol") == reverse_search("symbol")
    assert [hit["id"] for hit in hits] == ["a", "z"]
    assert hits[0]["start"] <= text.index("symbol") < hits[0]["end"]
    assert text.rindex("symbol") >= hits[0]["end"]


@pytest.mark.parametrize("query", ["", " \t\n", "...", "x" * 301, None, 1])
def test_search_rejects_empty_nonlexical_or_out_of_bounds_query(query):
    search, _ = helpers({"doc": "symbol"})

    with pytest.raises(ValueError):
        search(query)


@pytest.mark.parametrize("limit", [0, 6, True, 1.5])
def test_search_rejects_out_of_bounds_or_noninteger_limit(limit):
    search, _ = helpers({"doc": "symbol"})

    with pytest.raises(ValueError):
        search("symbol", limit=limit)


def test_search_accepts_query_and_limit_boundaries():
    text = "x " + "y" * 300
    search, _ = helpers({str(index): text for index in range(6)})

    assert len(search("x", limit=1)) == 1
    assert len(search("y" * 300, limit=5)) == 5


@pytest.mark.parametrize(
    "text", ["symbol", "symbol\n" + "padding\n" * 200, "padding\n" * 200 + "symbol"]
)
def test_excerpt_offsets_stay_inside_document_at_edges(text):
    search, _ = helpers({"doc": text})

    hit = search("symbol")[0]

    assert 0 <= hit["start"] < hit["end"] <= len(text)
    assert hit["excerpt"] == text[hit["start"] : hit["end"]]
    assert len(hit["excerpt"]) <= 900
    assert "symbol" in hit["excerpt"]


def test_source_offsets_use_original_characters_and_find_matches_on_long_lines():
    text = "ß" * 1000 + " " * 1000 + "symbol" + " " * 1000
    search, read = helpers({"doc": text})

    hit = search("symbol")[0]

    assert hit["start"] <= text.index("symbol") < hit["end"]
    assert hit["excerpt"] == text[hit["start"] : hit["end"]]
    assert read("doc", start=hit["start"], chars=900)["excerpt"] == hit["excerpt"]


def test_read_doc_keeps_page_contract_and_bounds_without_exposing_paths():
    text = "symbol " * 400
    _, read = helpers({"doc": text})

    page = read("doc", chars=1600)
    assert page["excerpt"] == text[:1600]
    assert page["more"] is True
    assert set(page) == {"id", "title", "sha256", "excerpt", "more"}
    assert read("doc", start=len(text), chars=1)["more"] is False
    assert read("doc", start=len(text), chars=1)["excerpt"] == ""
    for kwargs in ({"chars": 0}, {"chars": 1601}, {"chars": True}, {"start": -1}):
        with pytest.raises(ValueError):
            read("doc", **kwargs)
    with pytest.raises(ValueError):
        read("/private/docs/source.txt")
