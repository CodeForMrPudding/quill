import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quill.agents.note_agent import NoteAgent


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    """Records prompts it was called with and returns a short canned draft,
    so the chapter-filtering/index-skip glue logic can be tested fast and
    deterministically without live Ollama."""

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.calls.append(prompt)
        return FakeResponse(f"[draft #{len(self.calls)}]")


def make_agent(tmp_path, corpus):
    corpus_file = os.path.join(str(tmp_path), "corpus.json")
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f)

    agent = NoteAgent.__new__(NoteAgent)
    agent.model = FakeModel()
    agent.directory = str(tmp_path)
    agent.corpus_file = corpus_file
    agent.vector_store = None
    return agent


def make_chunk(idx, text, header1, source="book.pdf"):
    return {"title": f"chunk {idx}", "text": text, "idx": idx, "source": source, "header1": header1, "header2": ""}


def test_list_chapters_returns_title_counts_and_preview(tmp_path):
    corpus = [
        make_chunk(0, "Context: Ch1 > \n\nA vector space is a set of vectors.", "Chapter 1: Vector Spaces"),
        make_chunk(1, "Context: Ch1 > \n\nEvery vector space has a zero vector.", "Chapter 1: Vector Spaces"),
        make_chunk(2, "Context: Ch2 > \n\nAn eigenvalue scales an eigenvector.", "Chapter 2: Eigenvalues"),
    ]
    agent = make_agent(tmp_path, corpus)

    checkpoints = agent.list_chapters("book.pdf")

    assert [c["title"] for c in checkpoints] == ["Chapter 1: Vector Spaces", "Chapter 2: Eigenvalues"]
    assert checkpoints[0]["chunk_count"] == 2
    assert checkpoints[1]["chunk_count"] == 1
    assert "A vector space is a set of vectors." in checkpoints[0]["preview"]
    # No LLM calls for a cheap listing.
    assert agent.model.calls == []


def test_list_chapters_unknown_source_returns_empty(tmp_path):
    corpus = [make_chunk(0, "Context: Ch1 > \n\ntext", "Chapter 1", source="book.pdf")]
    agent = make_agent(tmp_path, corpus)

    # Regression: this used to fall back to the whole corpus, so a typo'd or
    # unparseable PDF would list (and draft) every other document's chapters.
    assert agent.list_chapters("other.pdf") == []


def test_list_sources_returns_distinct_ingested_filenames(tmp_path):
    corpus = [
        make_chunk(0, "a", "Ch1", source="Meeting #2.pdf"),
        make_chunk(1, "b", "Ch1", source="book.pdf"),
        make_chunk(2, "c", "Ch2", source="book.pdf"),
    ]
    agent = make_agent(tmp_path, corpus)
    assert agent.list_sources() == ["Meeting #2.pdf", "book.pdf"]


def test_draft_multi_page_notes_colliding_titles_get_distinct_pages(tmp_path):
    # "Q&A" and "QA" slugify identically; both chapters must survive, and the
    # index must link to the same names the pages were saved under.
    corpus = [
        make_chunk(0, "Context: \n\nfirst", "Q&A"),
        make_chunk(1, "Context: \n\nsecond", "QA"),
    ]
    agent = make_agent(tmp_path, corpus)

    pages = agent.draft_multi_page_notes("book.pdf")

    chapter_pages = sorted(name for name in pages if not name.endswith("__index.md"))
    assert chapter_pages == ["bookpdf__QA.md", "bookpdf__QA_2.md"]
    index = pages["bookpdf__index.md"]
    assert "[[bookpdf__QA|Q&A]]" in index
    assert "[[bookpdf__QA_2|QA]]" in index


def test_draft_single_chapter_uses_same_name_as_full_draft(tmp_path):
    corpus = [
        make_chunk(0, "Context: \n\nfirst", "Q&A"),
        make_chunk(1, "Context: \n\nsecond", "QA"),
    ]
    agent = make_agent(tmp_path, corpus)

    pages = agent.draft_multi_page_notes("book.pdf", chapters=["QA"])

    assert list(pages) == ["bookpdf__QA_2.md"]


def test_draft_multi_page_notes_all_chapters_includes_index(tmp_path):
    corpus = [
        make_chunk(0, "Context: Ch1 > \n\ntext one", "Chapter 1"),
        make_chunk(1, "Context: Ch2 > \n\ntext two", "Chapter 2"),
    ]
    agent = make_agent(tmp_path, corpus)

    pages = agent.draft_multi_page_notes("book.pdf")

    assert any(name.endswith("__index.md") for name in pages)
    assert len(pages) == 3  # 2 chapters + index
    assert len(agent.model.calls) == 2  # one draft call per chapter, no reduce needed


def test_draft_multi_page_notes_scoped_to_one_chapter_skips_index(tmp_path):
    corpus = [
        make_chunk(0, "Context: Ch1 > \n\ntext one", "Chapter 1: Vector Spaces"),
        make_chunk(1, "Context: Ch2 > \n\ntext two", "Chapter 2: Eigenvalues"),
    ]
    agent = make_agent(tmp_path, corpus)

    pages = agent.draft_multi_page_notes("book.pdf", chapters=["Chapter 1: Vector Spaces"])

    assert len(pages) == 1
    assert not any(name.endswith("__index.md") for name in pages)
    assert len(agent.model.calls) == 1  # only the requested chapter was drafted


def test_draft_multi_page_notes_unmatched_chapter_returns_empty(tmp_path):
    corpus = [make_chunk(0, "Context: Ch1 > \n\ntext", "Chapter 1")]
    agent = make_agent(tmp_path, corpus)

    pages = agent.draft_multi_page_notes("book.pdf", chapters=["Nonexistent Chapter"])

    assert pages == {}
    assert agent.model.calls == []
