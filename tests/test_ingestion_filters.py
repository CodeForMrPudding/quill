import os
import sys
import json
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quill.helpers.pdf_ingestion import (
    MIN_CHUNK_CHARS,
    _is_boilerplate_chunk,
    _should_drop_legal_page,
    _strip_repeated_lines,
    ingest_pdfs,
    load_manifest,
)


def page(*lines):
    return {"text": "\n".join(lines)}


def test_repeated_line_lookalikes_in_page_body_are_kept():
    # "Step 1"/"Step 2" both normalize to "Step #", the running footer's
    # shape. Only the footer position (page edge) may be stripped.
    pages = [page("Header", "", "", f"Step {i} of the method", "", "", f"Step {i}") for i in range(1, 6)]
    cleaned = _strip_repeated_lines(pages)
    for i, text in enumerate(cleaned, start=1):
        assert f"Step {i} of the method" in text
        assert not text.rstrip().endswith(f"Step {i}")


def test_table_separator_rows_are_never_stripped():
    table = ["| a | b |", "|---|---|", "| 1 | 2 |"]
    pages = [page(*table, "", f"unique body {i}", "", *table) for i in range(6)]
    for text in _strip_repeated_lines(pages):
        assert "|---|---|" in text


def test_appendix_sections_are_kept():
    assert _is_boilerplate_chunk({"Header 1": "Appendix A: Proofs"}, "x" * (MIN_CHUNK_CHARS + 10)) is False
    assert _is_boilerplate_chunk({"Header 1": "Appendix B"}, "x" * (MIN_CHUNK_CHARS + 10)) is False


def test_legal_page_dropped_in_front_matter_only():
    copyright_page = "Copyright (c) 2021 Someone. All rights reserved. ISBN 978-0-00-000000-0"
    bibliography_line = "Smith, J. (2019). A Book. Publisher. ISBN 978-1-23-456789-0. " * 40
    assert _should_drop_legal_page(copyright_page, 1, 300) is True
    assert _should_drop_legal_page(copyright_page, 299, 300) is True  # back matter
    # Mid-book page mentioning an ISBN (bibliography, citation) is content.
    assert _should_drop_legal_page(bibliography_line, 150, 300) is False
    # A law chapter discussing copyright isn't dropped for one phrase.
    assert _should_drop_legal_page("Under copyright (c) law, fair use..." * 20, 120, 300) is False
    # A short mid-book page stacked with legal notices still is.
    assert _should_drop_legal_page(copyright_page, 150, 300) is True


def test_scanned_pdf_is_flagged_in_manifest(tmp_path, monkeypatch):
    fake = types.SimpleNamespace(to_markdown=lambda path, page_chunks=True: [{"text": ""}, {"text": " "}, {"text": ""}])
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake)
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "scan.pdf").write_bytes(b"%PDF image only")
    corpus = tmp_path / "corpus.json"

    ingest_pdfs(str(drop), str(corpus))

    entry = load_manifest(str(corpus))["scan.pdf"]
    assert entry["scanned"] is True and entry["chunks"] == 0 and entry["pages"] == 3


def test_load_manifest_upgrades_old_list_format(tmp_path):
    corpus = tmp_path / "corpus.json"
    (tmp_path / "corpus.manifest.json").write_text(json.dumps({"a.pdf": [10, 20]}), encoding="utf-8")
    assert load_manifest(str(corpus)) == {"a.pdf": {"sig": [10, 20]}}
