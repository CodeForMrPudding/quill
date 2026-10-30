import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quill.helpers.pdf_ingestion import (
    _strip_repeated_lines,
    _is_boilerplate_chunk,
    _is_legal_boilerplate_page,
    _split_oversized_text,
    _extract_page_and_clean,
    _attribute_pages_to_pieces,
    _should_fallback_to_page_blocks,
    _page_block_label,
    _page_marker,
    _retry_import,
    MIN_CHUNK_CHARS,
    PAGE_BLOCK_SIZE,
    MIN_PAGES_FOR_FALLBACK_CHECK,
)


def make_page(index, body_line):
    """Builds a page with >2*edge_lines (=6) total lines, so the top-3/
    bottom-3 edge slices used by _strip_repeated_lines never overlap and the
    body line in the middle is never mistaken for a running header/footer."""
    lines = [
        "Linear Algebra — Chapter 3",
        "Section 3.2",
        "",
        body_line,
        "",
        "",
        f"Page {index} of 500",
    ]
    return {"text": "\n".join(lines)}


def test_strip_repeated_lines_removes_running_header_and_footer():
    pages = [make_page(i, f"Unique content sentence for page {i}.") for i in range(1, 6)]

    cleaned = _strip_repeated_lines(pages)

    assert len(cleaned) == 5
    for i, page_text in enumerate(cleaned, start=1):
        assert "Linear Algebra — Chapter 3" not in page_text
        assert "Section 3.2" not in page_text
        assert "of 500" not in page_text
        assert f"Unique content sentence for page {i}." in page_text


def test_strip_repeated_lines_keeps_lines_below_repeat_threshold():
    pages = [make_page(1, "Rare line only on page 1")]
    pages += [make_page(i, f"middle body unique to page {i}") for i in range(2, 6)]

    cleaned = _strip_repeated_lines(pages)

    # "Rare line only on page 1" is the body of just 1/5 pages — below the
    # 40% repeat threshold — so it must be preserved, not treated as boilerplate.
    assert "Rare line only on page 1" in cleaned[0]


def test_strip_repeated_lines_empty_input():
    assert _strip_repeated_lines([]) == []


def test_is_boilerplate_chunk_flags_known_boilerplate_headers():
    long_text = "x" * (MIN_CHUNK_CHARS + 10)
    assert _is_boilerplate_chunk({"Header 1": "References"}, long_text) is True
    assert _is_boilerplate_chunk({"Header 1": "Bibliography"}, long_text) is True
    assert _is_boilerplate_chunk({"Header 1": "INDEX"}, long_text) is True
    assert _is_boilerplate_chunk({"Header 3": "Acknowledgements"}, long_text) is True
    # Bare "Contents" (not just "Table of Contents") and "Dedication" are
    # common front-matter headings in real books (found via live testing on
    # a real novel PDF) and must be filtered too.
    assert _is_boilerplate_chunk({"Header 2": "Contents"}, long_text) is True
    assert _is_boilerplate_chunk({"Header 1": "Dedication"}, long_text) is True


def test_is_boilerplate_chunk_keeps_real_content_headers():
    long_text = "x" * (MIN_CHUNK_CHARS + 10)
    assert _is_boilerplate_chunk({"Header 1": "Chapter 3: Linear Transformations"}, long_text) is False
    assert _is_boilerplate_chunk({"Header 2": "3.1 Eigenvalues"}, long_text) is False


def test_is_boilerplate_chunk_flags_short_fragments_even_with_real_header():
    short_text = "Fig. 3.2"
    assert len(short_text) < MIN_CHUNK_CHARS
    assert _is_boilerplate_chunk({"Header 1": "Chapter 3"}, short_text) is True


def test_is_boilerplate_chunk_boundary_length():
    exactly_min = "x" * MIN_CHUNK_CHARS
    one_under = "x" * (MIN_CHUNK_CHARS - 1)
    assert _is_boilerplate_chunk({}, exactly_min) is False
    assert _is_boilerplate_chunk({}, one_under) is True


def test_is_legal_boilerplate_page_flags_real_copyright_page():
    # Real copyright-page text found via live testing on an actual novel PDF.
    text = (
        "Project Hail Mary is a work of fiction. Names, characters, places, "
        "and incidents are the products of the author's imagination or are "
        "used fictitiously. Copyright © 2021 by Andy Weir. All rights reserved."
    )
    assert _is_legal_boilerplate_page(text) is True


def test_is_legal_boilerplate_page_flags_isbn_notice():
    assert _is_legal_boilerplate_page("ISBN: 978-0-593-13520-4") is True


def test_is_legal_boilerplate_page_flags_library_of_congress():
    assert _is_legal_boilerplate_page("Library of Congress Cataloging-in-Publication Data") is True


def test_is_legal_boilerplate_page_keeps_real_narrative_content():
    text = (
        "I wake up. How long was I out? It must have been a while because I "
        "feel rested. I open my eyes without any effort. That's progress."
    )
    assert _is_legal_boilerplate_page(text) is False


def test_is_legal_boilerplate_page_keeps_empty_page():
    assert _is_legal_boilerplate_page("") is False


def test_split_oversized_text_returns_single_piece_when_under_budget():
    text = "short text"
    assert _split_oversized_text(text, max_chars=1000) == [text]


def test_split_oversized_text_splits_on_paragraph_boundaries():
    # Simulates one giant header-delimited section (e.g. a chapter with no
    # sub-headers spanning many pages) that must be sub-split to stay within
    # a bounded chunk size for embedding/drafting safety.
    paragraphs = [f"Paragraph {i} " + ("word " * 20) for i in range(20)]
    text = "\n\n".join(paragraphs)

    pieces = _split_oversized_text(text, max_chars=300, overlap=20)

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 300
    # No paragraph's content is lost.
    reconstructed = " ".join(pieces)
    for i in range(20):
        assert f"Paragraph {i} " in reconstructed


def test_split_oversized_text_hard_cuts_a_single_oversized_paragraph():
    # One giant paragraph with no internal breaks at all -- must still be
    # bounded, via a hard cut with overlap, rather than left oversized.
    text = "word " * 500  # ~2500 chars, no "\n\n" breaks anywhere
    assert "\n\n" not in text

    pieces = _split_oversized_text(text, max_chars=300, overlap=20)

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 300


def test_split_oversized_text_no_piece_ever_exceeds_max_chars():
    # Mixed paragraph sizes, some far larger than the budget -- every
    # returned piece must still respect the cap, regardless of input shape.
    text = "\n\n".join([
        "short one",
        "word " * 200,  # oversized paragraph
        "another short paragraph here",
        "word " * 5,
    ])

    pieces = _split_oversized_text(text, max_chars=250, overlap=20)

    assert all(len(p) <= 250 for p in pieces)


def test_extract_page_and_clean_finds_marker_and_strips_it():
    text = f"{_page_marker(7)}\nSome heading\n\nBody content here."
    page, cleaned = _extract_page_and_clean(text)
    assert page == 7
    assert "QUILLPAGE" not in cleaned
    assert "Some heading" in cleaned
    assert "Body content here." in cleaned


def test_extract_page_and_clean_no_marker_returns_none():
    text = "Just plain content, no marker."
    page, cleaned = _extract_page_and_clean(text)
    assert page is None
    assert cleaned == text


def test_attribute_pages_to_pieces_assigns_each_piece_its_own_marker():
    pieces = [
        f"{_page_marker(0)}\nPage zero content here.",
        f"{_page_marker(1)}\nPage one content here.",
        f"{_page_marker(2)}\nPage two content here.",
    ]
    result = _attribute_pages_to_pieces(pieces)
    assert [p for p, _ in result] == [0, 1, 2]
    assert all("QUILLPAGE" not in text for _, text in result)


def test_attribute_pages_to_pieces_carries_forward_when_piece_has_no_marker():
    # Regression case: a giant header-delimited section spanning many pages
    # (the exact scenario that triggers page-block fallback) gets split into
    # size-capped pieces where only the first piece from each page's content
    # carries that page's marker -- later pieces from the *same* page, or
    # from a page whose marker landed in an earlier sibling piece, must
    # inherit the most recently seen page rather than losing it.
    pieces = [
        f"{_page_marker(5)}\nFirst chunk of page five.",
        "More page five content, no marker here.",
        f"{_page_marker(6)}\nPage six starts now.",
    ]
    result = _attribute_pages_to_pieces(pieces)
    assert [p for p, _ in result] == [5, 5, 6]


def test_attribute_pages_to_pieces_uses_carry_from_for_leading_unmarked_piece():
    pieces = ["No marker at all in this piece."]
    result = _attribute_pages_to_pieces(pieces, carry_from=12)
    assert result == [(12, "No marker at all in this piece.")]


def test_attribute_pages_to_pieces_empty_input():
    assert _attribute_pages_to_pieces([]) == []


def test_should_fallback_short_document_never_falls_back():
    # Even zero headers shouldn't trigger fallback for a short document --
    # a handful of "Part" blocks would be pointless there.
    header_counts = {"": 10}
    assert _should_fallback_to_page_blocks(header_counts, 10, MIN_PAGES_FOR_FALLBACK_CHECK - 1) is False


def test_should_fallback_no_headers_found_on_long_document():
    header_counts = {"": 50}
    assert _should_fallback_to_page_blocks(header_counts, 50, 100) is True


def test_should_fallback_one_dominant_header_on_long_document():
    # Real scenario from live testing: a false-positive header swallows
    # almost the entire document (e.g. chapter dividers rendered as images
    # with no extractable text, so no real headers are ever found again).
    header_counts = {"Misdetected Header": 56, "": 2}
    assert _should_fallback_to_page_blocks(header_counts, 58, 100) is True


def test_should_fallback_well_distributed_headers_do_not_fall_back():
    header_counts = {f"Chapter {i}": 5 for i in range(10)}
    assert _should_fallback_to_page_blocks(header_counts, 50, 100) is False


def test_should_fallback_zero_chunks_returns_false():
    assert _should_fallback_to_page_blocks({}, 0, 100) is False


def test_page_block_label_groups_within_same_block():
    total_pages = 100
    assert _page_block_label(0, total_pages, block_size=25) == "Part (pages 1-25)"
    assert _page_block_label(24, total_pages, block_size=25) == "Part (pages 1-25)"


def test_page_block_label_advances_to_next_block():
    total_pages = 100
    assert _page_block_label(25, total_pages, block_size=25) == "Part (pages 26-50)"


def test_page_block_label_truncates_final_block_at_total_pages():
    total_pages = 90
    assert _page_block_label(80, total_pages, block_size=25) == "Part (pages 76-90)"


def test_retry_import_succeeds_first_try():
    calls = []

    def import_fn():
        calls.append(1)
        return "ok"

    assert _retry_import(import_fn, attempts=3, delay=0) == "ok"
    assert len(calls) == 1


def test_retry_import_succeeds_after_transient_failures():
    calls = []

    def import_fn():
        calls.append(1)
        if len(calls) < 3:
            raise ImportError("DLL load failed while importing onnxruntime_pybind11_state")
        return "ok"

    assert _retry_import(import_fn, attempts=3, delay=0) == "ok"
    assert len(calls) == 3


def test_retry_import_raises_after_exhausting_attempts():
    def import_fn():
        raise ImportError("DLL load failed")

    with pytest.raises(ImportError):
        _retry_import(import_fn, attempts=3, delay=0)


def test_retry_import_retries_oserror_too():
    calls = []

    def import_fn():
        calls.append(1)
        if len(calls) < 2:
            raise OSError("transient")
        return "ok"

    assert _retry_import(import_fn, attempts=3, delay=0) == "ok"


def test_ingest_pdfs_picks_up_uppercase_extension(tmp_path, monkeypatch):
    import json
    import types
    from quill.helpers.pdf_ingestion import ingest_pdfs

    fake = types.ModuleType("pymupdf4llm")
    fake.to_markdown = lambda path, page_chunks=True: [
        {"text": "# Chapter 1\n\nA sufficiently long paragraph of real content for the corpus."}
    ]
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake)

    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "REPORT.PDF").write_bytes(b"%PDF fake")
    out = tmp_path / "corpus.json"

    ingest_pdfs(str(drop), str(out))

    corpus = json.loads(out.read_text(encoding="utf-8"))
    assert {c["source"] for c in corpus} == {"REPORT.PDF"}


def test_strip_repeated_lines_keeps_short_single_page_content():
    # Regression: overlapping top/bottom edge slices double-counted every
    # line of a short page, stripping a one-page document to nothing.
    pages = [{"text": "# Title\n\nOnly paragraph."}]
    assert _strip_repeated_lines(pages) == ["# Title\n\nOnly paragraph."]
