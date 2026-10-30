import os
import json
import threading
from datetime import datetime
from typing import List, Optional


class WatchJobStore:
    """
    Remembers, per PDF, what `quill watch` has done with it, in a JSON file:

    - "in_progress": the user said yes; chapters_done lists finished ones.
      An interrupted run (crash, Ctrl+C, model error) resumes from here on
      the next start instead of being treated as finished -- previously the
      index page, written first, made a half-drafted book look "done".
    - "done": every chapter was attempted (failures are listed).
    - "declined": the user said no; not asked again unless the PDF changes.

    Each record carries the PDF's file signature, so a changed PDF (new
    edition, re-download) is offered again.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        try:
            with open(path, encoding="utf-8") as f:
                self._jobs = json.load(f)
        except (OSError, ValueError):
            self._jobs = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._jobs, f, indent=2)
        os.replace(tmp, self.path)

    def get(self, name: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(name)
            return dict(job) if job else None

    def _update(self, name: str, **fields) -> None:
        with self._lock:
            job = self._jobs.setdefault(name, {})
            job.update(fields, updated=datetime.now().isoformat())
            self._save()

    def start(self, name: str, signature: list, chapters: List[str]) -> None:
        self._update(name, status="in_progress", sig=signature, chapters=chapters,
                     chapters_done=[], chapters_failed={})

    def chapter_done(self, name: str, title: str) -> None:
        with self._lock:
            job = self._jobs[name]
            if title not in job["chapters_done"]:
                job["chapters_done"].append(title)
            job.get("chapters_failed", {}).pop(title, None)
            self._save()

    def chapter_failed(self, name: str, title: str, error: str) -> None:
        with self._lock:
            self._jobs[name].setdefault("chapters_failed", {})[title] = error[:300]
            self._save()

    def finish(self, name: str) -> None:
        self._update(name, status="done")

    def decline(self, name: str, signature: list) -> None:
        self._update(name, status="declined", sig=signature)

    def remaining_chapters(self, name: str, chapters: List[str]) -> List[str]:
        job = self.get(name) or {}
        done = set(job.get("chapters_done", []))
        return [c for c in chapters if c not in done]

    def needs_attention(self, name: str, signature: list, has_notes: bool) -> bool:
        """Whether a PDF should be offered (or resumed) at startup."""
        job = self.get(name)
        if job is None:
            # Never seen by this store: offer it unless notes already exist
            # (processed by a Quill version that predates this store).
            return not has_notes
        if job.get("status") == "in_progress":
            return True
        return job.get("sig") != signature
