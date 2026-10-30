import os
import sys
import json
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quill.helpers.dashboard import (
    human_size,
    dir_size,
    find_notes,
    find_pending_pdfs,
    corpus_stats,
    version_control_stats,
    session_stats,
    gather_dashboard_stats,
)


def test_human_size_formats_across_units():
    assert human_size(0) == "0 B"
    assert human_size(500) == "500 B"
    assert human_size(2048) == "2.0 KB"
    assert human_size(5 * 1024 * 1024) == "5.0 MB"
    assert human_size(3 * 1024 * 1024 * 1024) == "3.0 GB"


def test_dir_size_sums_files_recursively(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_bytes(b"y" * 250)

    assert dir_size(str(tmp_path)) == 350


def test_dir_size_missing_path_returns_zero(tmp_path):
    assert dir_size(str(tmp_path / "nonexistent")) == 0


def test_find_notes_excludes_bookkeeping_dirs(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text("hello")
    (vault / "note2.md").write_text("world!")
    quill_dir = vault / ".quill"
    quill_dir.mkdir()
    (quill_dir / "versions.db").write_bytes(b"not a note")
    chroma_dir = vault / "chroma_index"
    chroma_dir.mkdir()
    (chroma_dir / "index.bin").write_bytes(b"not a note either")

    notes = find_notes(str(vault))

    assert [n for n, _ in notes] == ["note1.md", "note2.md"]


def test_find_notes_missing_vault_returns_empty():
    assert find_notes("/nonexistent/path") == []


def test_find_pending_pdfs_lists_only_pdfs(tmp_path):
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "book.pdf").write_bytes(b"x" * 10)
    (drop / "notes.txt").write_text("not a pdf")

    pdfs = find_pending_pdfs(str(drop))

    assert pdfs == [("book.pdf", 10)]


def test_corpus_stats_counts_chunks_and_sources(tmp_path):
    vault = tmp_path
    corpus = [
        {"source": "book1.pdf", "text": "a"},
        {"source": "book1.pdf", "text": "b"},
        {"source": "book2.pdf", "text": "c"},
    ]
    with open(os.path.join(str(vault), "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f)

    stats = corpus_stats(str(vault))

    assert stats == {"chunk_count": 3, "source_count": 2}


def test_corpus_stats_missing_file_returns_zeros(tmp_path):
    assert corpus_stats(str(tmp_path)) == {"chunk_count": 0, "source_count": 0}


def test_version_control_stats_reads_real_schema(tmp_path):
    vault = tmp_path
    quill_dir = os.path.join(str(vault), ".quill")
    os.makedirs(quill_dir)
    db_path = os.path.join(quill_dir, "versions.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE notes (path TEXT PRIMARY KEY, head_commit_id INTEGER)")
    conn.execute("CREATE TABLE commits (id INTEGER PRIMARY KEY, note_path TEXT)")
    conn.execute("CREATE TABLE stashes (id INTEGER PRIMARY KEY, note_path TEXT)")
    conn.execute("INSERT INTO notes VALUES ('a.md', 1)")
    conn.execute("INSERT INTO commits (id, note_path) VALUES (1, 'a.md')")
    conn.execute("INSERT INTO commits (id, note_path) VALUES (2, 'a.md')")
    conn.commit()
    conn.close()

    stats = version_control_stats(str(vault))

    assert stats["tracked_notes"] == 1
    assert stats["commits"] == 2
    assert stats["stashes"] == 0
    assert stats["db_size"] > 0


def test_version_control_stats_missing_db_returns_zeros(tmp_path):
    stats = version_control_stats(str(tmp_path))
    assert stats == {"tracked_notes": 0, "commits": 0, "stashes": 0, "db_size": 0}


def test_session_stats_reads_real_schema(tmp_path):
    db_path = os.path.join(str(tmp_path), "chat_history.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('s1')")
    conn.execute("INSERT INTO messages (id, session_id) VALUES (1, 's1')")
    conn.commit()
    conn.close()

    stats = session_stats(db_path)

    assert stats["session_count"] == 1
    assert stats["message_count"] == 1
    assert stats["db_size"] > 0


def test_session_stats_missing_db_returns_zeros(tmp_path):
    stats = session_stats(os.path.join(str(tmp_path), "chat_history.db"))
    assert stats == {"session_count": 0, "message_count": 0, "db_size": 0}


def test_gather_dashboard_stats_assembles_everything(tmp_path):
    repo = tmp_path
    vault = os.path.join(str(repo), "obsidian_vault")
    os.makedirs(vault)
    with open(os.path.join(vault, "note.md"), "w") as f:
        f.write("# Note")

    stats = gather_dashboard_stats(str(repo))

    assert stats["repo_dir"] == str(repo)
    assert stats["vault_dir"] == vault
    assert stats["notes"] == [("note.md", 6)]
    assert stats["pending_pdfs"] == []
    assert stats["corpus"] == {"chunk_count": 0, "source_count": 0}
    assert stats["vector_index_size"] == 0
    assert stats["version_control"]["tracked_notes"] == 0
    assert stats["sessions"]["session_count"] == 0
    assert stats["vault_total_size"] >= 6
