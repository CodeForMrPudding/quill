import os
import sys
import json
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from quill.helpers.pdf_ingestion import ingest_pdfs, manifest_path_for
from quill.agents.note_agent import NoteAgent, chunk_ids


# --- Ingestion: only new/changed PDFs are parsed ---------------------------

@pytest.fixture
def fake_pymupdf(monkeypatch):
    """Fake pymupdf4llm whose output is the PDF file's own bytes, recording
    which files were parsed."""
    parsed = []

    def to_markdown(path, page_chunks=True):
        parsed.append(os.path.basename(path))
        with open(path, "rb") as f:
            body = f.read().decode("utf-8")
        if body == "CORRUPT":
            raise RuntimeError("cannot parse")
        return [{"text": f"# Chapter\n\n{body} -- a sufficiently long paragraph of real content."}]

    monkeypatch.setitem(sys.modules, "pymupdf4llm", types.SimpleNamespace(to_markdown=to_markdown))
    return parsed


def drop(dir_path, name, body):
    (dir_path / name).write_text(body, encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_unchanged_pdfs_are_not_reparsed(tmp_path, fake_pymupdf):
    pdfs, corpus = tmp_path / "drop", tmp_path / "corpus.json"
    pdfs.mkdir()
    drop(pdfs, "a.pdf", "alpha")
    drop(pdfs, "b.pdf", "beta")

    assert ingest_pdfs(str(pdfs), str(corpus)) == ["a.pdf", "b.pdf"]
    first = load(corpus)

    drop(pdfs, "c.pdf", "gamma")
    fake_pymupdf.clear()
    assert ingest_pdfs(str(pdfs), str(corpus)) == ["c.pdf"]
    assert fake_pymupdf == ["c.pdf"]  # a and b reused from the stored corpus

    second = load(corpus)
    assert [c["source"] for c in second] == ["a.pdf", "b.pdf", "c.pdf"]
    assert second[:2] == first[:2]
    assert [c["idx"] for c in second] == [0, 1, 2]

    fake_pymupdf.clear()
    assert ingest_pdfs(str(pdfs), str(corpus)) == []
    assert fake_pymupdf == []


def test_changed_pdf_is_reparsed_and_removed_pdf_dropped(tmp_path, fake_pymupdf):
    pdfs, corpus = tmp_path / "drop", tmp_path / "corpus.json"
    pdfs.mkdir()
    drop(pdfs, "a.pdf", "alpha")
    drop(pdfs, "b.pdf", "beta")
    ingest_pdfs(str(pdfs), str(corpus))

    drop(pdfs, "a.pdf", "alpha, second edition with more text")
    (pdfs / "b.pdf").unlink()
    fake_pymupdf.clear()
    ingest_pdfs(str(pdfs), str(corpus))

    assert fake_pymupdf == ["a.pdf"]
    chunks = load(corpus)
    assert [c["source"] for c in chunks] == ["a.pdf"]
    assert "second edition" in chunks[0]["text"]
    assert set(load(tmp_path / "corpus.manifest.json")) == {"a.pdf"}


def test_failed_parse_is_retried_next_run(tmp_path, fake_pymupdf):
    pdfs, corpus = tmp_path / "drop", tmp_path / "corpus.json"
    pdfs.mkdir()
    drop(pdfs, "bad.pdf", "CORRUPT")
    ingest_pdfs(str(pdfs), str(corpus))
    assert "bad.pdf" not in load(tmp_path / "corpus.manifest.json")

    fake_pymupdf.clear()
    ingest_pdfs(str(pdfs), str(corpus))
    assert fake_pymupdf == ["bad.pdf"]


def test_ingest_leaves_no_temp_files(tmp_path, fake_pymupdf):
    pdfs, corpus = tmp_path / "drop", tmp_path / "corpus.json"
    pdfs.mkdir()
    drop(pdfs, "a.pdf", "alpha")
    ingest_pdfs(str(pdfs), str(corpus))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["corpus.json", "corpus.manifest.json", "drop"]
    assert manifest_path_for(str(corpus)).endswith("corpus.manifest.json")


# --- Vector index: only new chunks are embedded ------------------------------

class CountingEmbeddings(DeterministicFakeEmbedding):
    embedded: list = []

    def embed_documents(self, texts):
        CountingEmbeddings.embedded.extend(texts)
        return super().embed_documents(texts)


@pytest.fixture
def fake_embeddings(monkeypatch):
    import langchain_ollama
    CountingEmbeddings.embedded = []
    monkeypatch.setattr(langchain_ollama, "OllamaEmbeddings", lambda model: CountingEmbeddings(size=16))
    return CountingEmbeddings


def make_corpus(*items):
    return [{"text": text, "source": source, "header1": "Ch", "header2": "", "page": 0, "idx": i, "title": f"t{i}"}
            for i, (source, text) in enumerate(items)]


def make_agent(tmp_path, corpus):
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    (vault / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    agent = NoteAgent.__new__(NoteAgent)
    import threading
    agent.model, agent.directory = None, str(vault)
    agent.corpus_file = str(vault / "corpus.json")
    agent.index_dir = str(vault / ".quill" / "chroma_index")
    agent.vector_store, agent._setup_lock = None, threading.Lock()
    return agent


def test_chunk_ids_are_stable_across_renumbering_and_distinct_for_duplicates():
    corpus = make_corpus(("a.pdf", "same"), ("a.pdf", "same"), ("b.pdf", "same"))
    ids = chunk_ids(corpus)
    assert len(set(ids)) == 3
    renumbered = [{**c, "idx": c["idx"] + 100, "title": "other"} for c in corpus]
    assert chunk_ids(renumbered) == ids


def test_setup_retrieval_embeds_only_new_chunks(tmp_path, fake_embeddings):
    corpus = make_corpus(("a.pdf", "apples are red"), ("a.pdf", "bananas are yellow"))
    agent = make_agent(tmp_path, corpus)
    agent.setup_retrieval()
    assert len(fake_embeddings.embedded) == 2

    # Relaunch: a fresh agent on the same index embeds nothing.
    fake_embeddings.embedded.clear()
    make_agent(tmp_path, corpus).setup_retrieval()
    assert fake_embeddings.embedded == []

    # A new PDF's chunk is the only thing embedded; a removed PDF's chunks go.
    corpus = make_corpus(("b.pdf", "cherries are dark red"), ("a.pdf", "apples are red"))
    (tmp_path / "vault" / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    fake_embeddings.embedded.clear()
    agent.setup_retrieval()
    assert fake_embeddings.embedded == ["cherries are dark red"]
    assert sorted(agent.vector_store.get(include=[])["ids"]) == sorted(chunk_ids(corpus))


def test_search_can_be_scoped_to_one_source(tmp_path, fake_embeddings):
    agent = make_agent(tmp_path, make_corpus(("a.pdf", "apples"), ("b.pdf", "bananas")))
    agent.setup_retrieval()
    assert agent.search("fruit", k=5, source="b.pdf") == ["bananas"]
    assert len(agent.search("fruit", k=5)) == 2


def test_index_rebuilt_once_when_embedding_model_changes(tmp_path, fake_embeddings, monkeypatch):
    corpus = make_corpus(("a.pdf", "apples"))
    make_agent(tmp_path, corpus).setup_retrieval()
    import quill.agents.note_agent as na
    monkeypatch.setattr(na, "EMBEDDING_MODEL_NAME", "some-other-embedder")

    fake_embeddings.embedded.clear()
    make_agent(tmp_path, corpus).setup_retrieval()
    assert fake_embeddings.embedded == ["apples"]


def test_search_without_index_returns_empty(tmp_path):
    agent = make_agent(tmp_path, [])
    assert agent.search("anything") == []


def test_note_agent_startup_ingests_pdfs_added_while_quill_was_off(tmp_path, fake_pymupdf, fake_embeddings):
    vault, pdfs = tmp_path / "vault", tmp_path / "drop"
    pdfs.mkdir()
    drop(pdfs, "a.pdf", "alpha")
    NoteAgent(None, str(vault), pdf_source_dir=str(pdfs))
    drop(pdfs, "b.pdf", "beta")  # dropped while nothing was running

    agent = NoteAgent(None, str(vault), pdf_source_dir=str(pdfs))

    assert agent.list_sources() == ["a.pdf", "b.pdf"]
