import os
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

from quill.helpers.paths import QuillPaths, resolve_paths


def human_size(num_bytes: int) -> str:
    """Formats a byte count as a human-readable size (e.g. '4.2 MB')."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def dir_size(path: str) -> int:
    """Total size in bytes of all files under `path`, recursively. Missing path -> 0."""
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def find_notes(vault_dir: str) -> List[Tuple[str, int]]:
    """Lists (relative_path, size_bytes) for every .md note in the vault,
    excluding internal bookkeeping directories (.quill, chroma_index)."""
    if not os.path.isdir(vault_dir):
        return []
    notes = []
    for root, dirs, files in os.walk(vault_dir):
        dirs[:] = [d for d in dirs if d not in (".quill", "chroma_index", ".obsidian", ".trash")]
        for f in files:
            if f.endswith(".md"):
                fp = os.path.join(root, f)
                try:
                    size = os.path.getsize(fp)
                except OSError:
                    size = 0
                notes.append((os.path.relpath(fp, vault_dir), size))
    return sorted(notes)


def find_pending_pdfs(drop_dir: str) -> List[Tuple[str, int]]:
    """Lists (filename, size_bytes) for every PDF sitting in the drop folder."""
    if not os.path.isdir(drop_dir):
        return []
    result = []
    for f in os.listdir(drop_dir):
        if f.lower().endswith(".pdf"):
            fp = os.path.join(drop_dir, f)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            result.append((f, size))
    return sorted(result)


def corpus_stats(vault_dir: str) -> Dict[str, int]:
    """Returns chunk/source counts from corpus.json, or zeros if absent/unreadable."""
    corpus_path = os.path.join(vault_dir, "corpus.json")
    if not os.path.exists(corpus_path):
        return {"chunk_count": 0, "source_count": 0}
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"chunk_count": 0, "source_count": 0}
    sources = {c.get("source") for c in corpus if c.get("source")}
    return {"chunk_count": len(corpus), "source_count": len(sources)}


def version_control_stats(vault_dir: str) -> Dict[str, int]:
    """Returns tracked-note/commit/stash counts from .quill/versions.db, or zeros if absent."""
    db_path = os.path.join(vault_dir, ".quill", "versions.db")
    empty = {"tracked_notes": 0, "commits": 0, "stashes": 0, "db_size": 0}
    if not os.path.exists(db_path):
        return empty
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        tracked = cur.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        commits = cur.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        stashes = cur.execute("SELECT COUNT(*) FROM stashes").fetchone()[0]
        conn.close()
    except sqlite3.Error:
        return {**empty, "db_size": os.path.getsize(db_path)}
    return {"tracked_notes": tracked, "commits": commits, "stashes": stashes, "db_size": os.path.getsize(db_path)}


def session_stats(session_db_path: str) -> Dict[str, int]:
    """Returns session/message counts from chat_history.db, or zeros if absent."""
    empty = {"session_count": 0, "message_count": 0, "db_size": 0}
    if not os.path.exists(session_db_path):
        return empty
    try:
        conn = sqlite3.connect(session_db_path)
        cur = conn.cursor()
        sessions = cur.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()
    except sqlite3.Error:
        return {**empty, "db_size": os.path.getsize(session_db_path)}
    return {"session_count": sessions, "message_count": messages, "db_size": os.path.getsize(session_db_path)}


def gather_dashboard_stats(repo_dir: Optional[str] = None, paths: Optional[QuillPaths] = None) -> dict:
    """
    Collects everything `quill dashboard` displays: note locations, pending
    PDFs, corpus/index stats, version-control stats, session stats, and disk
    usage -- purely from the filesystem/sqlite, no LLM or Ollama connection
    required, so the dashboard works even when Ollama isn't running.
    """
    paths = paths or resolve_paths(home=repo_dir)
    # Older layouts kept the corpus and index directly in the vault.
    data_dir = paths.data if os.path.exists(paths.corpus) else paths.vault
    index_dir = paths.index if os.path.isdir(paths.index) else os.path.join(paths.vault, "chroma_index")

    return {
        "repo_dir": paths.home,
        "vault_dir": paths.vault,
        "drop_dir": paths.drop,
        "notes": find_notes(paths.vault),
        "pending_pdfs": find_pending_pdfs(paths.drop),
        "corpus": corpus_stats(data_dir),
        "vector_index_size": dir_size(index_dir),
        "version_control": version_control_stats(paths.vault),
        "sessions": session_stats(paths.sessions_db),
        "vault_total_size": dir_size(paths.vault),
    }
