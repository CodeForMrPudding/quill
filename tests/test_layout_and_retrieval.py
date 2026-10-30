import os
import sys
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from quill.agents.note_agent import NoteAgent, format_context
from quill.helpers.note_planner import build_index_page, group_chunks_by_chapter
from quill.helpers.paths import default_home, migrate_legacy_layout, resolve_paths
from quill.helpers.pending_drafts import PendingDraftStore


# --- Paths / layout ----------------------------------------------------------

def test_default_home_uses_cwd_only_when_it_has_a_quill_layout(tmp_path):
    (tmp_path / "proj" / "obsidian_vault").mkdir(parents=True)
    assert default_home(str(tmp_path / "proj")) == str(tmp_path / "proj")
    assert default_home(str(tmp_path)).endswith("Quill")


def test_resolve_paths_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("QUILL_VAULT", str(tmp_path / "env_vault"))
    paths = resolve_paths(home=str(tmp_path))
    assert paths.vault == str(tmp_path / "env_vault")
    assert paths.drop == str(tmp_path / "pdf_drop")
    assert resolve_paths(home=str(tmp_path), vault=str(tmp_path / "cli_vault")).vault == str(tmp_path / "cli_vault")
    assert paths.corpus == os.path.join(str(tmp_path / "env_vault"), ".quill", "corpus.json")


def test_migrate_legacy_layout_moves_files_without_overwriting(tmp_path):
    paths = resolve_paths(home=str(tmp_path))
    os.makedirs(paths.vault)
    open(os.path.join(paths.vault, "corpus.json"), "w").write("[]")
    os.makedirs(os.path.join(paths.vault, "chroma_index"))

    assert sorted(migrate_legacy_layout(paths)) == ["chroma_index", "corpus.json"]
    assert os.path.exists(paths.corpus) and os.path.isdir(paths.index)
    assert not os.path.exists(os.path.join(paths.vault, "corpus.json"))
    assert migrate_legacy_layout(paths) == []


# --- Pending drafts survive restarts -------------------------------------------

def test_pending_drafts_persist_across_restarts(tmp_path):
    store_dir = str(tmp_path / "pending")
    store = PendingDraftStore(str(tmp_path), store_dir=store_dir)
    store.add("book__Ch1.md", "# Ch1", source="book.pdf")
    store.add("book__Ch2.md", "# Ch2", source="book.pdf")

    reopened = PendingDraftStore(str(tmp_path), store_dir=store_dir)
    assert reopened.list() == [("book__Ch1.md", "book.pdf"), ("book__Ch2.md", "book.pdf")]
    assert reopened.take("book__Ch1.md") == "# Ch1"
    reopened.discard("book__Ch2.md")
    assert PendingDraftStore(str(tmp_path), store_dir=store_dir).list() == []


# --- Chapter structure ---------------------------------------------------------

def test_generic_section_titles_stay_with_their_chapter():
    chunks = [{"header1": h, "text": "x", "source": "b.pdf"} for h in
              ["Chapter 1: Sets", "Exercises", "Chapter 2: Maps", "Exercises", "Summary"]]
    groups = group_chunks_by_chapter(chunks)
    assert list(groups) == ["Chapter 1: Sets", "Chapter 1: Sets — Exercises", "Chapter 2: Maps",
                            "Chapter 2: Maps — Exercises", "Chapter 2: Maps — Summary"]


def test_index_page_includes_overview_when_given():
    _, content = build_index_page("b.pdf", ["Ch1"], overview="It covers sets.")
    assert content.index("## Overview") < content.index("## Chapters")
    assert "It covers sets." in content


class FakeResponse:
    def __init__(self, content):
        self.content = content


class CountingModel:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return FakeResponse(f"overview {len(self.prompts)}")


def test_draft_overview_reduces_many_chapters_hierarchically(tmp_path):
    agent = NoteAgent.__new__(NoteAgent)
    agent.model = CountingModel()
    result = agent.draft_overview("b.pdf", [f"# Ch{i}\n\nbody {i}" for i in range(20)])
    assert result.startswith("overview")
    assert len(agent.model.prompts) == 4  # 20 pages -> 3 groups of <=8 -> 1


def test_draft_multi_page_notes_streams_pages_as_they_finish(tmp_path):
    corpus = [{"text": f"Context: c\n\nt{i}", "source": "b.pdf", "header1": f"Ch{i}", "idx": i, "title": ""}
              for i in range(3)]
    (tmp_path / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    agent = NoteAgent.__new__(NoteAgent)
    agent.model, agent.corpus_file = CountingModel(), str(tmp_path / "corpus.json")
    seen = []
    agent.draft_multi_page_notes("b.pdf", on_page=lambda name, content: seen.append(name))
    assert seen == ["bpdf__Ch0.md", "bpdf__Ch1.md", "bpdf__Ch2.md", "bpdf__index.md"]


# --- Chat retrieval: notes indexed, citations, scoping -------------------------

@pytest.fixture
def indexed_agent(tmp_path, monkeypatch):
    import langchain_ollama
    monkeypatch.setattr(langchain_ollama, "OllamaEmbeddings", lambda model: DeterministicFakeEmbedding(size=16))
    vault = tmp_path / "vault"
    (vault / ".quill").mkdir(parents=True)
    corpus = [
        {"text": "apples are red", "source": "Fruit Guide.pdf", "header1": "Ch", "page": 11, "idx": 0, "title": ""},
        {"text": "bolts need torque", "source": "Car Manual.pdf", "header1": "Ch", "page": 3, "idx": 1, "title": ""},
    ]
    (vault / ".quill" / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    (vault / "My Thoughts.md").write_text("I prefer green apples.", encoding="utf-8")
    agent = NoteAgent.__new__(NoteAgent)
    agent.model, agent.directory = None, str(vault)
    agent.data_dir = str(vault / ".quill")
    agent.corpus_file = str(vault / ".quill" / "corpus.json")
    agent.index_dir = str(vault / ".quill" / "chroma_index")
    agent.pdf_source_dir = str(tmp_path / "drop")
    agent.vector_store, agent._setup_lock = None, threading.Lock()
    agent.setup_retrieval()
    return agent, vault


def test_notes_are_searchable_and_resynced_only_when_changed(indexed_agent):
    agent, vault = indexed_agent
    assert agent.sync_note_index() == 1
    assert agent.sync_note_index() == 0  # unchanged: nothing re-embedded
    notes = agent.search_docs("apples", k=10, kind="note")
    assert [m["path"] for _, m in notes] == ["My Thoughts.md"]

    (vault / "My Thoughts.md").write_text("Actually I prefer red apples.", encoding="utf-8")
    os.utime(vault / "My Thoughts.md", None)
    assert agent.sync_note_index() == 1
    assert [t for t, _ in agent.search_docs("apples", k=10, kind="note")] == ["Actually I prefer red apples."]

    (vault / "My Thoughts.md").unlink()
    agent.sync_note_index()
    assert agent.search_docs("apples", k=10, kind="note") == []


def test_setup_retrieval_does_not_delete_note_entries(indexed_agent):
    agent, vault = indexed_agent
    agent.sync_note_index()
    agent.setup_retrieval()
    assert len(agent.search_docs("x", k=10, kind="note")) == 1


def test_context_is_labeled_for_citation(indexed_agent):
    agent, vault = indexed_agent
    agent.sync_note_index()
    context = format_context(agent.search_docs("apples", k=10))
    assert "[Fruit Guide.pdf, p.12]" in context  # 0-based page 11 shown as p.12
    assert "[note: My Thoughts.md]" in context


def test_detect_source_and_scoped_search(indexed_agent):
    agent, _ = indexed_agent
    assert agent.detect_source("what does the car manual say about bolts?") == "Car Manual.pdf"
    assert agent.detect_source("hello there") is None
    assert [m["source"] for _, m in agent.search_docs("anything", k=10, source="Car Manual.pdf")] == ["Car Manual.pdf"]
