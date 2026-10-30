import os
import time
import threading
from datetime import timedelta

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from quill.helpers.note_versioning import NoteVersionControl, is_ignored_vault_path

# A note is committed once it has gone this long without another change.
# Obsidian saves every couple of seconds while typing; committing on every
# save would bury real versions under hundreds of keystroke-level ones.
DEFAULT_QUIET_SECONDS = 30.0
DAILY_SNAPSHOT_AGE = timedelta(hours=24)


class _NoteChangeHandler(FileSystemEventHandler):
    def __init__(self, snapshotter: "NoteAutoSnapshotter"):
        self._s = snapshotter

    def on_created(self, event):
        if not event.is_directory:
            self._s._touch(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._s._touch(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._s._moved(event.src_path, event.dest_path)


class NoteAutoSnapshotter:
    """
    Keeps note history without the user ever running /commit:

    - on start, commits every note changed while Quill wasn't running;
    - while running, commits each edited note once it's been quiet for
      `quiet_seconds` (debounced, so one editing session = one version);
    - follows renames, so a note's history moves with it;
    - takes an automatic vault-wide snapshot if the last one is a day old.

    Commits only actual changes (NoteVersionControl compares against HEAD),
    so Quill's own writes -- which already commit -- don't create duplicates.
    """

    def __init__(self, vcs: NoteVersionControl, quiet_seconds: float = DEFAULT_QUIET_SECONDS,
                 snapshot_age: timedelta = DAILY_SNAPSHOT_AGE, poll_interval: float = 1.0):
        self.vcs = vcs
        self.vault_dir = vcs.vault_dir
        self.quiet_seconds = quiet_seconds
        self.snapshot_age = snapshot_age
        self.poll_interval = poll_interval

        self._pending = {}  # rel_path -> monotonic time of last change
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._observer = Observer()
        self._worker = None

    def _rel(self, abs_path: str):
        try:
            rel = os.path.relpath(abs_path, self.vault_dir).replace(os.sep, "/")
        except ValueError:  # different drive on Windows
            return None
        if rel.startswith("../") or not rel.endswith(".md") or is_ignored_vault_path(rel):
            return None
        return rel

    def _touch(self, abs_path: str) -> None:
        rel = self._rel(abs_path)
        if rel:
            with self._pending_lock:
                self._pending[rel] = time.monotonic()

    def _moved(self, src_abs: str, dest_abs: str) -> None:
        src, dest = self._rel(src_abs), self._rel(dest_abs)
        if src and dest:
            with self._pending_lock:
                self._pending.pop(src, None)
            try:
                self.vcs.rename(src, dest)
            except Exception as e:
                print(f"[Quill Snapshots] Couldn't move history of '{src}' to '{dest}': {e}")
        if dest:
            self._touch(dest_abs)

    def catch_up(self) -> int:
        """Commits notes changed while Quill wasn't running; returns how many."""
        return len(self.vcs.commit_all("Auto-snapshot of changes made while Quill wasn't running"))

    def flush(self, force: bool = False) -> int:
        """Commits pending notes that have been quiet long enough (or all of
        them, with force=True). Returns how many were committed."""
        now = time.monotonic()
        with self._pending_lock:
            due = [p for p, t in self._pending.items() if force or now - t >= self.quiet_seconds]
            for p in due:
                del self._pending[p]
        committed = 0
        for rel in due:
            try:
                if self.vcs.commit_if_changed(rel, "Auto-snapshot") is not None:
                    committed += 1
            except Exception as e:
                print(f"[Quill Snapshots] Couldn't snapshot '{rel}': {e}")
        return committed

    def start(self) -> None:
        os.makedirs(self.vault_dir, exist_ok=True)
        self.catch_up()
        self.vcs.ensure_recent_snapshot(self.snapshot_age)
        self._observer.schedule(_NoteChangeHandler(self), self.vault_dir, recursive=True)
        self._observer.start()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """Stops watching, committing any edits still waiting out their
        quiet period so nothing typed just before exit is left unversioned."""
        self._stop_event.set()
        self._observer.stop()
        self._observer.join(timeout=5)
        if self._worker:
            self._worker.join(timeout=5)
        self.flush(force=True)

    def _loop(self) -> None:
        last_snapshot_check = time.monotonic()
        while not self._stop_event.wait(self.poll_interval):
            self.flush()
            if time.monotonic() - last_snapshot_check >= 600:
                last_snapshot_check = time.monotonic()
                try:
                    self.vcs.ensure_recent_snapshot(self.snapshot_age)
                except Exception as e:
                    print(f"[Quill Snapshots] Daily snapshot failed: {e}")
