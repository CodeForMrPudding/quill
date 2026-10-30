import os
import queue
import shutil
import threading
from typing import Callable, Optional

from quill.helpers.note_planner import assign_page_names, build_index_page, has_notes
from quill.helpers.pdf_ingestion import ingest_pdfs, _file_signature


class PdfWatchFlow:
    """
    What `quill watch` does with PDFs, independent of the CLI:

    - Asking ("Generate notes for X?") happens on its own short-lived
      thread per PDF, so an unanswered toast no longer blocks every other
      PDF behind it for minutes.
    - Drafting happens on one worker thread (one GPU; one book at a time),
      fed by a queue of approved PDFs.
    - Progress is recorded per chapter in a WatchJobStore, so a run that's
      interrupted resumes where it stopped, and the final notification
      reports what actually succeeded.
    - An optional inbox folder (e.g. Downloads) can be watched as well: new
      PDFs there are offered, and only approved ones are copied into the
      library folder and ingested. The library folder is what gets parsed
      in full, so pointing it at Downloads would ingest every unrelated PDF
      in there.
    """

    def __init__(self, note_agent, vcs, notifier, jobs, log: Optional[Callable[..., None]] = None,
                 ask_timeout: float = 300.0):
        self.note_agent = note_agent
        self.vcs = vcs
        self.notifier = notifier
        self.jobs = jobs
        self.log = log or (lambda msg, **_: print(msg))
        self.ask_timeout = ask_timeout
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._queued = set()
        self._queued_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stops after the chapter currently being drafted; the rest of
        that book resumes on the next start."""
        self._stop.set()
        self._queue.put(None)
        if self._worker:
            self._worker.join(timeout=timeout)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Blocks until nothing is queued or drafting (used by tests)."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._queued_lock:
                if not self._queued:
                    return True
            time.sleep(0.05)
        return False

    # --- entry points -----------------------------------------------------

    def _library(self) -> str:
        return self.note_agent.pdf_source_dir

    def _signature(self, name: str):
        try:
            return _file_signature(os.path.join(self._library(), name))
        except OSError:
            return None

    def _sync_corpus(self) -> list:
        parsed = ingest_pdfs(self._library(), self.note_agent.corpus_file)
        if parsed:
            self.note_agent.setup_retrieval()
        return parsed

    def catch_up(self) -> None:
        """At startup: offer PDFs nobody was asked about yet, and resume
        interrupted ones (the corpus was synced by NoteAgent's startup)."""
        existing_notes = os.listdir(self.vcs.vault_dir) if os.path.isdir(self.vcs.vault_dir) else []
        for name in sorted(os.listdir(self._library())):
            if not name.lower().endswith(".pdf"):
                continue
            sig = self._signature(name)
            if sig and self.jobs.needs_attention(name, sig, has_notes(name, existing_notes)):
                self.offer(name)

    def on_library_pdf(self, path: str) -> None:
        """VaultWatcher callback for the library folder."""
        name = os.path.basename(path)
        try:
            self._sync_corpus()
        except Exception as e:
            self.log(f"[Quill Watch]: Ingestion failed for '{name}': {e}", fg="red")
            self.notifier.notify("Quill", f"Ingestion failed for '{name}'.")
            return
        sig = self._signature(name)
        existing_notes = os.listdir(self.vcs.vault_dir) if os.path.isdir(self.vcs.vault_dir) else []
        if sig and self.jobs.needs_attention(name, sig, has_notes(name, existing_notes)):
            self.offer(name)

    def on_inbox_pdf(self, path: str) -> None:
        """VaultWatcher callback for an inbox folder such as Downloads: ask
        first, and only copy approved PDFs into the library."""
        name = os.path.basename(path)

        def ask_then_import():
            answer = self.notifier.ask_yes_no(
                "Quill", f"New PDF: {name}. Add it to Quill and generate notes?", timeout=self.ask_timeout
            )
            if answer is not True:
                self.log(f"[Quill Watch]: Not importing '{name}' ({'declined' if answer is False else 'no answer'}).",
                         fg="yellow")
                return
            try:
                imported = self.import_into_library(path)
                self._sync_corpus()
            except Exception as e:
                self.log(f"[Quill Watch]: Couldn't import '{name}': {e}", fg="red")
                return
            self.offer(imported, approved=True)

        threading.Thread(target=ask_then_import, daemon=True).start()

    def import_into_library(self, path: str) -> str:
        """Copies a PDF into the library folder; returns its library name.
        Re-importing an identical file reuses it; a different file with the
        same name gets a numbered name instead of overwriting."""
        name = os.path.basename(path)
        stem, ext = os.path.splitext(name)
        candidate, n = name, 1
        while True:
            dest = os.path.join(self._library(), candidate)
            if not os.path.exists(dest):
                shutil.copy2(path, dest)
                return candidate
            if os.path.getsize(dest) == os.path.getsize(path):
                return candidate
            n += 1
            candidate = f"{stem} ({n}){ext}"

    # --- asking and drafting ----------------------------------------------

    def offer(self, name: str, approved: bool = False) -> None:
        checkpoints = self.note_agent.list_chapters(name)
        if not checkpoints:
            reason = self.note_agent.source_status(name) or f"No content found in '{name}'."
            self.log(f"[Quill Watch]: {reason}", fg="yellow")
            self.notifier.notify("Quill", reason)
            return

        job = self.jobs.get(name)
        sig = self._signature(name)
        if approved or (job and job.get("status") == "in_progress" and job.get("sig") == sig):
            if job and job.get("status") == "in_progress" and not approved:
                done = len(job.get("chapters_done", []))
                self.log(f"[Quill Watch]: Resuming '{name}' ({done}/{len(checkpoints)} chapter(s) already done).",
                         fg="magenta")
            self._enqueue(name)
            return

        def ask():
            existing = os.listdir(self.vcs.vault_dir) if os.path.isdir(self.vcs.vault_dir) else []
            if has_notes(name, existing):
                question = (f"'{name}' changed and already has notes. Regenerate them? "
                            f"Your current versions are kept in note history.")
            else:
                question = f"New PDF: {name} ({len(checkpoints)} chapter(s)). Generate notes?"
            self.log(f"[Quill Watch]: {len(checkpoints)} chapter(s) found in '{name}'. Asking for approval...",
                     fg="magenta")
            answer = self.notifier.ask_yes_no("Quill", question, timeout=self.ask_timeout)
            if answer is True:
                self._enqueue(name)
            elif answer is False:
                self.jobs.decline(name, sig)
                self.log(f"[Quill Watch]: Declined '{name}'; won't ask again unless the file changes.", fg="yellow")
            else:
                self.log(f"[Quill Watch]: No answer for '{name}'; will ask again next time Quill starts.",
                         fg="yellow")

        threading.Thread(target=ask, daemon=True).start()

    def _enqueue(self, name: str) -> None:
        with self._queued_lock:
            if name in self._queued:
                return
            self._queued.add(name)
        self._queue.put(name)

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                name = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if name is None:
                break
            try:
                self.draft(name)
            except Exception as e:
                self.log(f"[Quill Watch]: Drafting '{name}' stopped: {e}. It will resume next start.", fg="red")
            finally:
                with self._queued_lock:
                    self._queued.discard(name)

    def draft(self, name: str) -> None:
        checkpoints = self.note_agent.list_chapters(name)
        titles = [cp["title"] for cp in checkpoints]
        sig = self._signature(name)
        job = self.jobs.get(name)
        if not (job and job.get("status") == "in_progress" and job.get("sig") == sig):
            self.jobs.start(name, sig, titles)

        index_name, index_content = build_index_page(name, titles)
        self.vcs.save(index_name, index_content, f"Index for {name}")

        remaining = self.jobs.remaining_chapters(name, titles)
        for i, title in enumerate(remaining, start=1):
            if self._stop.is_set():
                self.log(f"[Quill Watch]: Stopping; '{name}' will resume from '{title}' next start.", fg="yellow")
                return
            self.log(f"[Quill Watch]: Drafting '{title}' ({i}/{len(remaining)} remaining in '{name}')...",
                     fg="magenta")
            try:
                pages = self.note_agent.draft_multi_page_notes(name, chapters=[title])
                for page_name, content in pages.items():
                    commit_id = self.vcs.save(page_name, content, f"Auto-generated from {name}")
                    self.log(f"[Quill Watch]: Saved '{page_name}' (version #{commit_id})", fg="green")
                self.jobs.chapter_done(name, title)
            except Exception as e:
                self.jobs.chapter_failed(name, title, str(e))
                self.log(f"[Quill Watch]: Failed to draft '{title}': {e}", fg="red")

        job = self.jobs.get(name)
        done, failed = len(job["chapters_done"]), job.get("chapters_failed", {})
        if failed:
            # Left "in_progress", so the failed chapters are retried next start.
            message = f"Generated {done} of {len(titles)} chapter(s) from '{name}'; {len(failed)} failed (will retry)."
            self.log(f"[Quill Watch]: {message}", fg="yellow")
            self.notifier.notify("Quill", message)
            return

        self._add_overview(name, titles)
        self.jobs.finish(name)
        message = f"Finished generating {done} chapter note(s) from '{name}'."
        self.log(f"[Quill Watch]: {message}", fg="green")
        self.notifier.notify("Quill", message)

    def _add_overview(self, name: str, titles: list) -> None:
        pages = []
        for page_name in assign_page_names(name, titles).values():
            path = os.path.join(self.vcs.vault_dir, page_name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    pages.append(f.read())
        if not pages:
            return
        self.log(f"[Quill Watch]: Writing a whole-document overview for '{name}'...", fg="magenta")
        try:
            overview = self.note_agent.draft_overview(name, pages)
        except Exception as e:
            self.log(f"[Quill Watch]: Couldn't write the overview: {e}", fg="yellow")
            return
        index_name, index_content = build_index_page(name, titles, overview)
        self.vcs.save(index_name, index_content, f"Overview for {name}")
