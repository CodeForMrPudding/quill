import os
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from difflib import SequenceMatcher, unified_diff
from typing import Dict, List, Optional, Tuple

SNAPSHOT_INTERVAL = 10

# Vault subdirectories that never hold user notes: Quill's own bookkeeping,
# the vector index, and Obsidian's config/trash folders.
IGNORED_VAULT_DIRS = frozenset({".quill", "chroma_index", ".obsidian", ".trash"})


def is_ignored_vault_path(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    return any(p in IGNORED_VAULT_DIRS or p.startswith(".") for p in parts[:-1])


class NoteVersionControl:
    """
    Delta-based version control for individual Obsidian notes, independent of
    real git. Every commit after the first stores a diff (difflib opcodes)
    against its parent rather than a full copy, so disk usage grows with edit
    size, not file size. A full snapshot is stored whenever the delta chain
    back to the last full copy reaches SNAPSHOT_INTERVAL, so checkout never
    has to replay a long chain.

    On top of per-note commits, a *vault snapshot* records every note's HEAD
    at one moment, so the whole vault can be restored to that point at once.

    Safe to share between threads (one lock) and between processes (write
    transactions take SQLite's write lock up front, and re-read HEAD inside
    it), so the REPL and `quill watch` can both run auto-snapshots.
    """

    def __init__(self, vault_dir: str):
        self.vault_dir = vault_dir
        self.quill_dir = os.path.join(vault_dir, ".quill")
        os.makedirs(self.quill_dir, exist_ok=True)
        self.db_path = os.path.join(self.quill_dir, "versions.db")
        # Autocommit mode: transactions are opened explicitly by _txn().
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30, isolation_level=None)
        self._lock = threading.RLock()
        self._txn_depth = 0
        self._init_db()

    def _init_db(self):
        with self._txn():
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS notes (
                    path TEXT PRIMARY KEY,
                    head_commit_id INTEGER
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS commits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_path TEXT,
                    message TEXT,
                    timestamp TEXT,
                    parent_id INTEGER,
                    is_snapshot INTEGER,
                    content TEXT,
                    opcodes TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS stashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_path TEXT,
                    content TEXT,
                    timestamp TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS vault_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT,
                    timestamp TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS vault_snapshot_entries (
                    snapshot_id INTEGER,
                    note_path TEXT,
                    commit_id INTEGER,
                    PRIMARY KEY (snapshot_id, note_path)
                )
            ''')

    @contextmanager
    def _txn(self):
        """Thread-locked write transaction. Re-entrant: nested uses join the
        outermost transaction. BEGIN IMMEDIATE takes SQLite's write lock up
        front, so two processes can't both read the same HEAD and commit
        divergent children of it."""
        with self._lock:
            if self._txn_depth:
                self._txn_depth += 1
                try:
                    yield
                finally:
                    self._txn_depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._txn_depth = 1
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")
            finally:
                self._txn_depth = 0

    @staticmethod
    def _normalize(rel_path: str) -> str:
        rel_path = rel_path.replace("\\", "/").lstrip("/")
        if ".." in rel_path.split("/"):
            raise ValueError(f"Invalid note path: {rel_path}")
        return rel_path

    def _abs_path(self, rel_path: str) -> str:
        return os.path.join(self.vault_dir, *rel_path.split("/"))

    def _read_working(self, rel_path: str) -> Optional[str]:
        abs_path = self._abs_path(rel_path)
        if not os.path.exists(abs_path):
            return None
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_working(self, rel_path: str, content: str) -> None:
        abs_path = self._abs_path(rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        # newline="\n": Obsidian uses LF; plain text mode would rewrite every
        # restored note with CRLF on Windows.
        with open(abs_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

    def _get_note(self, rel_path: str) -> Optional[Tuple[str, Optional[int]]]:
        return self.conn.execute(
            "SELECT path, head_commit_id FROM notes WHERE path = ?", (rel_path,)
        ).fetchone()

    def _get_commit(self, commit_id: int):
        return self.conn.execute(
            "SELECT id, note_path, message, timestamp, parent_id, is_snapshot, content, opcodes "
            "FROM commits WHERE id = ?",
            (commit_id,)
        ).fetchone()

    @staticmethod
    def _apply_opcodes(old_lines: List[str], opcodes) -> List[str]:
        new_lines = []
        for tag, i1, i2, j1, j2, replacement in opcodes:
            if tag == "equal":
                new_lines.extend(old_lines[i1:i2])
            else:
                new_lines.extend(replacement)
        return new_lines

    def _reconstruct(self, rel_path: str, commit_id: int) -> str:
        chain = []
        current_id = commit_id
        while current_id is not None:
            row = self._get_commit(current_id)
            if row is None or row[1] != rel_path:
                raise ValueError(f"Commit {current_id} not found for note '{rel_path}'")
            chain.append(row)
            if row[5]:  # is_snapshot
                break
            current_id = row[4]  # parent_id
        chain.reverse()

        base_row = chain[0]
        lines = base_row[6].splitlines(keepends=True)  # content
        for row in chain[1:]:
            opcodes = json.loads(row[7])
            lines = self._apply_opcodes(lines, opcodes)
        return "".join(lines)

    def _deltas_since_full_copy(self, commit_id: int) -> int:
        """Number of delta commits between `commit_id` and its nearest
        full-copy ancestor. Measured along the actual parent chain (not a
        per-note commit count), so checking out an old version and
        committing on top of it can't grow an unbounded replay chain."""
        depth = 0
        row = self._get_commit(commit_id)
        while row is not None and not row[5]:
            depth += 1
            row = self._get_commit(row[4])
        return depth

    # --- Per-note commits -------------------------------------------------

    def commit(self, rel_path: str, message: str) -> int:
        rel_path = self._normalize(rel_path)
        working_content = self._read_working(rel_path)
        if working_content is None:
            raise FileNotFoundError(f"Note not found in vault: {rel_path}")

        with self._txn():
            note = self._get_note(rel_path)
            timestamp = datetime.now().isoformat()

            if note is None:
                cursor = self.conn.execute(
                    "INSERT INTO commits (note_path, message, timestamp, parent_id, is_snapshot, content, opcodes) "
                    "VALUES (?, ?, ?, ?, 1, ?, NULL)",
                    (rel_path, message, timestamp, None, working_content)
                )
                commit_id = cursor.lastrowid
                self.conn.execute(
                    "INSERT INTO notes (path, head_commit_id) VALUES (?, ?)", (rel_path, commit_id)
                )
                return commit_id

            head_commit_id = note[1]
            head_content = self._reconstruct(rel_path, head_commit_id)
            if head_content == working_content:
                raise ValueError(f"Nothing to commit — '{rel_path}' matches HEAD")

            is_snapshot = self._deltas_since_full_copy(head_commit_id) + 1 >= SNAPSHOT_INTERVAL

            if is_snapshot:
                cursor = self.conn.execute(
                    "INSERT INTO commits (note_path, message, timestamp, parent_id, is_snapshot, content, opcodes) "
                    "VALUES (?, ?, ?, ?, 1, ?, NULL)",
                    (rel_path, message, timestamp, head_commit_id, working_content)
                )
            else:
                old_lines = head_content.splitlines(keepends=True)
                new_lines = working_content.splitlines(keepends=True)
                matcher = SequenceMatcher(None, old_lines, new_lines)
                opcodes = [
                    [tag, i1, i2, j1, j2, None if tag == "equal" else new_lines[j1:j2]]
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                ]
                cursor = self.conn.execute(
                    "INSERT INTO commits (note_path, message, timestamp, parent_id, is_snapshot, content, opcodes) "
                    "VALUES (?, ?, ?, ?, 0, NULL, ?)",
                    (rel_path, message, timestamp, head_commit_id, json.dumps(opcodes))
                )

            commit_id = cursor.lastrowid
            self.conn.execute("UPDATE notes SET head_commit_id = ? WHERE path = ?", (commit_id, rel_path))
            return commit_id

    def commit_if_changed(self, rel_path: str, message: str) -> Optional[int]:
        """Commits `rel_path` if it's untracked or differs from HEAD; returns
        the new commit id, or None if there was nothing to commit (or the
        file is gone)."""
        try:
            return self.commit(rel_path, message)
        except (ValueError, FileNotFoundError):
            return None

    def commit_all(self, message: str) -> List[Tuple[str, int]]:
        """Commits every untracked or modified note in the vault. Returns
        [(path, commit_id), ...] for the notes that were committed."""
        committed = []
        for path, state in self.status():
            if state in ("untracked", "modified"):
                commit_id = self.commit_if_changed(path, message)
                if commit_id is not None:
                    committed.append((path, commit_id))
        return committed

    def save(self, rel_path: str, content: str, message: str) -> int:
        """
        Writes `content` to a note and commits it, without ever destroying
        what was there before: if the note already exists with edits that
        aren't committed yet (or exists but was never tracked), those edits
        are committed first as their own version, so overwriting a note the
        user changed in Obsidian is always recoverable via /log + /checkout.
        Use this -- not a bare open(..., "w") -- for any generated content
        that may land on an existing note. Returns the new HEAD commit id.
        """
        rel_path = self._normalize(rel_path)
        with self._txn():
            if self._status_one(rel_path) in ("untracked", "modified"):
                self.commit(rel_path, f"Snapshot of existing edits before: {message}")

            if self._read_working(rel_path) != content:
                self._write_working(rel_path, content)

            try:
                return self.commit(rel_path, message)
            except ValueError:
                # Content already matches HEAD -- nothing new to record.
                return self._get_note(rel_path)[1]

    def rename(self, old_path: str, new_path: str) -> bool:
        """Moves a note's whole history to its new path (e.g. after a rename
        in Obsidian), so history follows the note instead of the new name
        starting from scratch. Returns False if `old_path` wasn't tracked or
        `new_path` already has history of its own."""
        old_path, new_path = self._normalize(old_path), self._normalize(new_path)
        with self._txn():
            if self._get_note(old_path) is None or self._get_note(new_path) is not None:
                return False
            self.conn.execute("UPDATE notes SET path = ? WHERE path = ?", (new_path, old_path))
            self.conn.execute("UPDATE commits SET note_path = ? WHERE note_path = ?", (new_path, old_path))
            self.conn.execute("UPDATE stashes SET note_path = ? WHERE note_path = ?", (new_path, old_path))
            self.conn.execute(
                "UPDATE vault_snapshot_entries SET note_path = ? WHERE note_path = ?", (new_path, old_path)
            )
            return True

    # --- Inspection -------------------------------------------------------

    def _status_one(self, rel_path: str) -> str:
        working_content = self._read_working(rel_path)
        note = self._get_note(rel_path)

        if note is None:
            return "untracked" if working_content is not None else "unknown"
        if working_content is None:
            return "missing"

        head_content = self._reconstruct(rel_path, note[1])
        return "clean" if working_content == head_content else "modified"

    def status(self, rel_path: Optional[str] = None) -> List[Tuple[str, str]]:
        with self._lock:
            if rel_path is not None:
                rel_path = self._normalize(rel_path)
                return [(rel_path, self._status_one(rel_path))]

            results = {}
            for (path,) in self.conn.execute("SELECT path FROM notes").fetchall():
                results[path] = self._status_one(path)

            for root, dirs, files in os.walk(self.vault_dir):
                dirs[:] = [d for d in dirs if d not in IGNORED_VAULT_DIRS and not d.startswith(".")]
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    abs_path = os.path.join(root, filename)
                    rel = os.path.relpath(abs_path, self.vault_dir).replace(os.sep, "/")
                    if rel not in results:
                        results[rel] = "untracked"

            return sorted(results.items())

    def head(self, rel_path: str) -> Optional[int]:
        note = self._get_note(self._normalize(rel_path))
        return note[1] if note else None

    def log(self, rel_path: str) -> List[Tuple[int, str, str]]:
        """Every commit ever made to the note, newest first -- including
        versions on abandoned lines after a checkout. See ancestors() for
        just the line leading to the current HEAD."""
        rel_path = self._normalize(rel_path)
        return self.conn.execute(
            "SELECT id, message, timestamp FROM commits WHERE note_path = ? ORDER BY id DESC",
            (rel_path,)
        ).fetchall()

    def ancestors(self, rel_path: str) -> List[Tuple[int, str, str]]:
        """The note's current HEAD followed by its parent, grandparent, ...
        -- i.e. the history that led to the version on disk."""
        rel_path = self._normalize(rel_path)
        result = []
        current = self.head(rel_path)
        while current is not None:
            row = self._get_commit(current)
            result.append((row[0], row[2], row[3]))
            current = row[4]
        return result

    def latest(self, rel_path: str) -> Optional[int]:
        """The most recently created commit for the note, regardless of
        where HEAD currently points -- 'go back to the newest version'."""
        entries = self.log(rel_path)
        return entries[0][0] if entries else None

    def show(self, rel_path: str, commit_id: int) -> str:
        rel_path = self._normalize(rel_path)
        with self._lock:
            return self._reconstruct(rel_path, commit_id)

    def diff(self, rel_path: str, from_commit: Optional[int] = None, to_commit: Optional[int] = None) -> str:
        """Unified diff between two versions of a note. Defaults: from = HEAD,
        to = the working file (i.e. 'what changed since the last version')."""
        rel_path = self._normalize(rel_path)
        with self._lock:
            if self._get_note(rel_path) is None:
                raise ValueError(f"'{rel_path}' is not tracked")
            if from_commit is None:
                from_commit = self.head(rel_path)
            old = self._reconstruct(rel_path, from_commit)
            if to_commit is None:
                new = self._read_working(rel_path) or ""
                to_label = "working copy"
            else:
                new = self._reconstruct(rel_path, to_commit)
                to_label = f"#{to_commit}"
        return "".join(unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"{rel_path} (#{from_commit})", tofile=f"{rel_path} ({to_label})",
        ))

    # --- Restoring --------------------------------------------------------

    def checkout(self, rel_path: str, commit_id: int) -> None:
        rel_path = self._normalize(rel_path)
        with self._txn():
            note = self._get_note(rel_path)
            if note is None:
                raise ValueError(f"'{rel_path}' is not tracked")

            target_row = self._get_commit(commit_id)
            if target_row is None or target_row[1] != rel_path:
                raise ValueError(f"Commit {commit_id} not found for note '{rel_path}'")

            working_content = self._read_working(rel_path)
            head_content = self._reconstruct(rel_path, note[1])

            if working_content is not None and working_content != head_content:
                self.conn.execute(
                    "INSERT INTO stashes (note_path, content, timestamp) VALUES (?, ?, ?)",
                    (rel_path, working_content, datetime.now().isoformat())
                )

            self._write_working(rel_path, self._reconstruct(rel_path, commit_id))
            self.conn.execute("UPDATE notes SET head_commit_id = ? WHERE path = ?", (commit_id, rel_path))

    def stash_list(self, rel_path: Optional[str] = None) -> List[Tuple[int, str, str]]:
        if rel_path is not None:
            rel_path = self._normalize(rel_path)
            return self.conn.execute(
                "SELECT id, note_path, timestamp FROM stashes WHERE note_path = ? ORDER BY id DESC",
                (rel_path,)
            ).fetchall()
        return self.conn.execute("SELECT id, note_path, timestamp FROM stashes ORDER BY id DESC").fetchall()

    def stash_pop(self, rel_path: str, stash_id: Optional[int] = None) -> None:
        rel_path = self._normalize(rel_path)
        with self._txn():
            if stash_id is not None:
                row = self.conn.execute(
                    "SELECT id, content FROM stashes WHERE id = ? AND note_path = ?", (stash_id, rel_path)
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT id, content FROM stashes WHERE note_path = ? ORDER BY id DESC LIMIT 1", (rel_path,)
                ).fetchone()
            if row is None:
                raise ValueError(f"No stash found for '{rel_path}'")

            stash_row_id, content = row
            # Don't let popping a stash destroy *other* uncommitted edits:
            # stash those first, so the two swap places.
            if self._status_one(rel_path) == "modified":
                self.conn.execute(
                    "INSERT INTO stashes (note_path, content, timestamp) VALUES (?, ?, ?)",
                    (rel_path, self._read_working(rel_path), datetime.now().isoformat())
                )
            self._write_working(rel_path, content)
            self.conn.execute("DELETE FROM stashes WHERE id = ?", (stash_row_id,))

    # --- Vault-wide snapshots ---------------------------------------------

    def snapshot(self, message: str) -> int:
        """Commits every changed note, then records each existing note's HEAD
        under one vault snapshot id. Returns the snapshot id."""
        with self._txn():
            self.commit_all(f"Saved for vault snapshot: {message}")
            cursor = self.conn.execute(
                "INSERT INTO vault_snapshots (message, timestamp) VALUES (?, ?)",
                (message, datetime.now().isoformat())
            )
            snapshot_id = cursor.lastrowid
            for path, head_id in self.conn.execute("SELECT path, head_commit_id FROM notes").fetchall():
                if self._read_working(path) is not None:
                    self.conn.execute(
                        "INSERT INTO vault_snapshot_entries (snapshot_id, note_path, commit_id) VALUES (?, ?, ?)",
                        (snapshot_id, path, head_id)
                    )
            return snapshot_id

    def list_snapshots(self, limit: int = 50) -> List[Tuple[int, str, str, int]]:
        """[(id, message, timestamp, note_count), ...], newest first."""
        return self.conn.execute(
            "SELECT s.id, s.message, s.timestamp, COUNT(e.note_path) FROM vault_snapshots s "
            "LEFT JOIN vault_snapshot_entries e ON e.snapshot_id = s.id "
            "GROUP BY s.id ORDER BY s.id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    def ensure_recent_snapshot(self, max_age: timedelta = timedelta(hours=24),
                               message: str = "Automatic daily snapshot") -> Optional[int]:
        """Takes a vault snapshot if the newest one is older than `max_age`
        (or none exists). Returns the new snapshot id, or None."""
        with self._txn():
            row = self.conn.execute("SELECT MAX(timestamp) FROM vault_snapshots").fetchone()
            if row[0] and datetime.now() - datetime.fromisoformat(row[0]) < max_age:
                return None
            if not self.conn.execute("SELECT 1 FROM notes LIMIT 1").fetchone() and not any(
                state == "untracked" for _, state in self.status()
            ):
                return None  # empty vault -- nothing worth snapshotting
            return self.snapshot(message)

    def restore_snapshot(self, snapshot_id: int) -> Dict[str, list]:
        """
        Puts every note recorded in the snapshot back to the version it had
        then. The current state is snapshotted first, so a restore can
        itself be undone by restoring that safety snapshot. Notes created
        after the snapshot are left in place (reported under "kept"), never
        deleted. Returns {"safety_snapshot", "restored", "kept"}.
        """
        with self._txn():
            if self.conn.execute("SELECT 1 FROM vault_snapshots WHERE id = ?", (snapshot_id,)).fetchone() is None:
                raise ValueError(f"Vault snapshot #{snapshot_id} not found")
            safety_id = self.snapshot(f"Before restoring vault snapshot #{snapshot_id}")

            entries = dict(self.conn.execute(
                "SELECT note_path, commit_id FROM vault_snapshot_entries WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchall())
            restored = []
            for path, commit_id in sorted(entries.items()):
                if self.head(path) != commit_id or self._status_one(path) != "clean":
                    self.checkout(path, commit_id)
                    restored.append(path)

            kept = sorted(
                path for path, state in self.status()
                if path not in entries and state not in ("missing", "unknown")
            )
            return {"safety_snapshot": safety_id, "restored": restored, "kept": kept}
