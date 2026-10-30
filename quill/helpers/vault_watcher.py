import os
import time
import threading
import queue
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class _PdfDropHandler(FileSystemEventHandler):
    def __init__(self, pending_queue: "queue.Queue[str]", seen: set, lock: threading.Lock):
        self._pending_queue = pending_queue
        self._seen = seen
        self._lock = lock

    def _maybe_enqueue(self, path: str) -> None:
        if not path.lower().endswith(".pdf"):
            return
        with self._lock:
            if path in self._seen:
                return
            self._seen.add(path)
        self._pending_queue.put(path)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.dest_path)

    def on_modified(self, event):
        # Also catches a file whose creation event was missed or that was
        # rewritten in place; the callback is expected to skip PDFs it has
        # already handled (ingestion is a no-op for unchanged files).
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)


def wait_until_stable(path: str, poll_interval: float = 0.5, stable_checks: int = 3,
                      empty_timeout: float = 600.0, stop_event: Optional[threading.Event] = None) -> bool:
    """
    Blocks until `path`'s size stops changing across `stable_checks`
    consecutive polls, so a PDF that's still being copied/downloaded isn't
    ingested mid-write. There's no overall time limit while the file keeps
    growing -- the old fixed 60s limit silently dropped any large file whose
    copy took longer. Returns False if the file disappears, stays empty for
    `empty_timeout` seconds, or `stop_event` is set.
    """
    empty_since = time.monotonic()
    last_size = -1
    stable_count = 0
    while not (stop_event and stop_event.is_set()):
        if not os.path.exists(path):
            return False
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size > 0:
            empty_since = time.monotonic()
        elif time.monotonic() - empty_since > empty_timeout:
            return False
        if size == last_size and size > 0:
            stable_count += 1
            if stable_count >= stable_checks:
                return True
        else:
            stable_count = 0
        last_size = size
        time.sleep(poll_interval)
    return False


class VaultWatcher:
    """
    Watches `drop_dir` for new PDFs and, once each file's write is stable,
    calls `on_pdf_ready(path)` from a background worker thread. This class is
    intentionally just a file-arrival detector — ingestion, drafting, and
    persistence all live in the callback, not here.
    """

    def __init__(self, drop_dir: str, on_pdf_ready, poll_interval: float = 0.5, stable_checks: int = 3):
        os.makedirs(drop_dir, exist_ok=True)
        self.drop_dir = drop_dir
        self.on_pdf_ready = on_pdf_ready
        self.poll_interval = poll_interval
        self.stable_checks = stable_checks

        self._pending: "queue.Queue[str]" = queue.Queue()
        self._seen_lock = threading.Lock()
        self._seen = set()
        self._observer = Observer()
        self._handler = _PdfDropHandler(self._pending, self._seen, self._seen_lock)
        self._worker_thread = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._observer.schedule(self._handler, self.drop_dir, recursive=False)
        self._observer.start()
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._observer.stop()
        self._observer.join(timeout=5)
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                path = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue

            if wait_until_stable(path, self.poll_interval, self.stable_checks, stop_event=self._stop_event):
                try:
                    self.on_pdf_ready(path)
                except Exception as e:
                    print(f"[Watcher] Error handling {path}: {e}")

            with self._seen_lock:
                self._seen.discard(path)
