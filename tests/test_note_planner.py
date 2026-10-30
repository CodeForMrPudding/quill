import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quill.helpers.note_planner import (
    slugify,
    group_chunks_by_chapter,
    batch_chunks_by_char_budget,
    chunk_ids_for_source,
    note_page_name,
    build_index_page,
    find_unprocessed_pdfs,
    assign_page_names,
    split_leading_arg,
    has_notes,
)


def make_chunk(idx, text, header1="", header2="", source="book.pdf"):
    return {"title": f"chunk {idx}", "text": text, "idx": idx, "source": source, "header1": header1, "header2": header2}


def test_group_chunks_by_chapter_preserves_order_and_groups_by_header1():
    corpus = [
        make_chunk(0, "intro text", header1="Chapter 1"),
        make_chunk(1, "more ch1", header1="Chapter 1"),
        make_chunk(2, "ch2 text", header1="Chapter 2"),
        make_chunk(3, "back to ch1?", header1="Chapter 1"),
    ]

    groups = group_chunks_by_chapter(corpus)

    assert list(groups.keys()) == ["Chapter 1", "Chapter 2"]
    assert len(groups["Chapter 1"]) == 3
    assert len(groups["Chapter 2"]) == 1


def test_group_chunks_falls_back_to_source_when_no_header():
    corpus = [
        make_chunk(0, "text a", header1="", source="untitled.pdf"),
        make_chunk(1, "text b", header1="", source="untitled.pdf"),
    ]

    groups = group_chunks_by_chapter(corpus)

    assert list(groups.keys()) == ["untitled.pdf"]
    assert len(groups["untitled.pdf"]) == 2


def test_group_chunks_falls_back_to_untitled_when_no_header_or_source():
    corpus = [{"title": "x", "text": "y", "idx": 0}]
    groups = group_chunks_by_chapter(corpus)
    assert list(groups.keys()) == ["Untitled"]


def test_batch_chunks_respects_char_budget():
    chunks = [make_chunk(i, "x" * 100) for i in range(10)]

    batches = batch_chunks_by_char_budget(chunks, max_chars=250)

    for batch in batches:
        total = sum(len(c["text"]) for c in batch)
        assert total <= 250 or len(batch) == 1  # oversized single chunks are the only exception
    # All chunks accounted for, order preserved.
    flattened = [c["idx"] for batch in batches for c in batch]
    assert flattened == list(range(10))


def test_batch_chunks_oversized_single_chunk_gets_its_own_batch():
    chunks = [make_chunk(0, "x" * 500), make_chunk(1, "y" * 10)]

    batches = batch_chunks_by_char_budget(chunks, max_chars=100)

    assert len(batches) == 2
    assert batches[0] == [chunks[0]]
    assert batches[1] == [chunks[1]]


def test_batch_chunks_empty_input():
    assert batch_chunks_by_char_budget([], max_chars=100) == []


def test_batch_chunks_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        batch_chunks_by_char_budget([make_chunk(0, "x")], max_chars=0)


def test_chunk_ids_for_source_filters_by_source():
    corpus = [
        make_chunk(0, "a", source="book1.pdf"),
        make_chunk(1, "b", source="book2.pdf"),
        make_chunk(2, "c", source="book1.pdf"),
    ]

    result = chunk_ids_for_source(corpus, "book1.pdf")

    assert [c["idx"] for c in result] == [0, 2]


def test_chunk_ids_for_source_falls_back_to_whole_corpus_when_untagged():
    corpus = [{"title": "x", "text": "a", "idx": 0}, {"title": "y", "text": "b", "idx": 1}]
    result = chunk_ids_for_source(corpus, "anything.pdf")
    assert result == corpus


def test_chunk_ids_for_source_unknown_source_in_tagged_corpus_returns_empty():
    corpus = [make_chunk(0, "a", source="book1.pdf"), make_chunk(1, "b", source="book2.pdf")]
    assert chunk_ids_for_source(corpus, "Meeting") == []


def test_slugify_strips_unsafe_characters():
    assert slugify("What's 2+2? / A Study!!") == "Whats_22_A_Study"


def test_slugify_keeps_non_latin_letters():
    assert slugify("Über Vektorräume") == "Über_Vektorräume"
    assert slugify("Введение") == "Введение"
    assert slugify('a<b>c:d"e|f?g*h') == "abcdefgh"  # Windows-forbidden chars still removed


def test_assign_page_names_dedupes_slug_and_case_collisions():
    names = assign_page_names("book.pdf", ["Q&A", "QA", "qa", "Intro"])
    assert list(names.values()) == [
        "bookpdf__QA.md", "bookpdf__QA_2.md", "bookpdf__qa_3.md", "bookpdf__Intro.md",
    ]


def test_assign_page_names_never_reuses_index_page_name():
    names = assign_page_names("book.pdf", ["index"])
    assert names["index"] == "bookpdf__index_2.md"


def test_split_leading_arg_resolves_known_name_with_spaces():
    known = ["Meeting #2.pdf", "Project Mary Hail.pdf"]
    assert split_leading_arg("Meeting #2.pdf Financials", ".pdf", known) == ("Meeting #2.pdf", "Financials")
    assert split_leading_arg("project mary hail.pdf", ".pdf", known) == ("Project Mary Hail.pdf", "")


def test_split_leading_arg_quoted_and_suffix_fallbacks():
    assert split_leading_arg('"My Note.md" fix typo', ".md") == ("My Note.md", "fix typo")
    assert split_leading_arg("My Note.md fix typo", ".md") == ("My Note.md", "fix typo")
    assert split_leading_arg("folder/Some Note.md 12", ".md") == ("folder/Some Note.md", "12")


def test_split_leading_arg_falls_back_to_first_token():
    assert split_leading_arg("note 3", ".md") == ("note", "3")
    assert split_leading_arg("", ".md") == ("", "")


def test_split_leading_arg_known_name_must_end_at_word_boundary():
    # "book.pdf" must not match the start of "book.pdfx"
    assert split_leading_arg("book.pdfx rest", ".pdf", ["book.pdf"]) == ("book.pdfx", "rest")


def test_has_notes_matches_generated_prefix_only():
    assert has_notes("Meeting #2.pdf", ["Meeting_2pdf__index.md"])
    assert not has_notes("Meeting #2.pdf", ["Meeting_2pdf.md", "Other__index.md"])


def test_slugify_empty_falls_back_to_untitled():
    assert slugify("") == "untitled"
    assert slugify("???") == "untitled"


def test_note_page_name_combines_source_and_chapter_slugs():
    assert note_page_name("book.pdf", "Chapter 1: Vector Spaces") == "bookpdf__Chapter_1_Vector_Spaces.md"


def test_build_index_page_links_every_chapter():
    name, content = build_index_page("book.pdf", ["Chapter 1: Vector Spaces", "Chapter 2: Eigenvalues"])

    assert name == "bookpdf__index.md"
    assert "# book.pdf" in content
    assert "[[bookpdf__Chapter_1_Vector_Spaces|Chapter 1: Vector Spaces]]" in content
    assert "[[bookpdf__Chapter_2_Eigenvalues|Chapter 2: Eigenvalues]]" in content


def test_build_index_page_empty_chapter_list():
    name, content = build_index_page("book.pdf", [])
    assert name == "bookpdf__index.md"
    assert "# book.pdf" in content
    assert "## Chapters" in content


def test_find_unprocessed_pdfs_returns_pdfs_with_no_notes_yet(tmp_path):
    drop = tmp_path / "pdf_drop"
    drop.mkdir()
    (drop / "new_book.pdf").write_bytes(b"fake pdf")
    (drop / "already_drafted.pdf").write_bytes(b"fake pdf")
    (drop / "notes.txt").write_text("not a pdf")

    vault = tmp_path / "vault"
    vault.mkdir()
    # A real note exists for already_drafted.pdf (its index page).
    (vault / f"{slugify('already_drafted.pdf')}__index.md").write_text("# already_drafted.pdf")

    result = find_unprocessed_pdfs(str(drop), str(vault))

    assert result == ["new_book.pdf"]


def test_find_unprocessed_pdfs_ingested_but_undrafted_still_counts_as_unprocessed(tmp_path):
    # Regression case: a PDF can be sitting in corpus.json (auto-ingested by
    # NoteAgent's startup step) without the user ever having been asked --
    # corpus membership alone must not suppress the prompt.
    drop = tmp_path / "pdf_drop"
    drop.mkdir()
    (drop / "book.pdf").write_bytes(b"fake pdf")

    vault = tmp_path / "vault"
    vault.mkdir()  # no notes written yet, even though ingestion may have run

    assert find_unprocessed_pdfs(str(drop), str(vault)) == ["book.pdf"]


def test_find_unprocessed_pdfs_no_vault_dir_returns_all_pdfs(tmp_path):
    drop = tmp_path / "pdf_drop"
    drop.mkdir()
    (drop / "a.pdf").write_bytes(b"x")
    (drop / "b.pdf").write_bytes(b"x")

    result = find_unprocessed_pdfs(str(drop), str(tmp_path / "nonexistent_vault"))

    assert result == ["a.pdf", "b.pdf"]


def test_find_unprocessed_pdfs_missing_drop_dir_returns_empty(tmp_path):
    assert find_unprocessed_pdfs(str(tmp_path / "nonexistent"), str(tmp_path)) == []


def test_find_unprocessed_pdfs_all_already_drafted_returns_empty(tmp_path):
    drop = tmp_path / "pdf_drop"
    drop.mkdir()
    (drop / "book.pdf").write_bytes(b"x")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / f"{slugify('book.pdf')}__Chapter_1.md").write_text("# Chapter 1")

    assert find_unprocessed_pdfs(str(drop), str(vault)) == []
