import os
import json
import hashlib
import threading
from datetime import datetime
from typing import List, Optional, Tuple

from quill.helpers.note_planner import slugify


class PendingDraftStore:
    """
    Thread-safe holding area for LLM-drafted notes that are waiting on user
    approval before they're written into the vault. The daemon/watcher thread
    populates this; the interactive CLI loop (single-threaded, owns stdin)
    drains it via /pending, /approve, /discard (or the *_all variants, for
    the multiple pages a single large source document can produce).

    With `store_dir`, every draft is also saved there as it's added (and
    removed once approved or discarded), and reloaded on startup. Drafts
    used to live only in memory, so quitting -- or a crash -- after a
    multi-hour /draft-all threw all of that work away.
    """

    def __init__(self, vault_dir: str, store_dir: Optional[str] = None):
        self.vault_dir = vault_dir
        self.store_dir = store_dir
        self._lock = threading.Lock()
        self._drafts = {}
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)
            for filename in os.listdir(store_dir):
                if not filename.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(store_dir, filename), encoding="utf-8") as f:
                        record = json.load(f)
                    self._drafts[record["name"]] = {"content": record["content"], "source": record.get("source", "")}
                except (OSError, ValueError, KeyError):
                    continue  # unreadable leftover; ignore rather than fail startup

    def _record_path(self, name: str) -> str:
        return os.path.join(self.store_dir, hashlib.sha1(name.encode("utf-8")).hexdigest()[:20] + ".json")

    def _persist(self, name: str, draft: Optional[dict]) -> None:
        if not self.store_dir:
            return
        path = self._record_path(name)
        if draft is None:
            if os.path.exists(path):
                os.remove(path)
            return
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"name": name, "source": draft["source"], "content": draft["content"],
                       "saved": datetime.now().isoformat()}, f)
        os.replace(tmp, path)

    def unique_name(self, topic: str) -> str:
        slug = slugify(topic)
        candidate = f"{slug}.md"
        n = 1
        with self._lock:
            existing = set(self._drafts.keys())
            while candidate in existing or os.path.exists(os.path.join(self.vault_dir, candidate)):
                n += 1
                candidate = f"{slug}_{n}.md"
        return candidate

    def add(self, name: str, content: str, source: str = "") -> None:
        with self._lock:
            self._drafts[name] = {"content": content, "source": source}
            self._persist(name, self._drafts[name])

    def list(self) -> List[Tuple[str, str]]:
        with self._lock:
            return sorted((name, d["source"]) for name, d in self._drafts.items())

    def list_for_source(self, source: str) -> List[str]:
        with self._lock:
            return sorted(name for name, d in self._drafts.items() if d["source"] == source)

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            draft = self._drafts.get(name)
        return draft["content"] if draft else None

    def take(self, name: str) -> str:
        """Removes a draft from the pending list and returns its content
        without writing anything, so the caller can persist it through
        version control (NoteVersionControl.save) instead of a bare write."""
        with self._lock:
            draft = self._drafts.pop(name, None)
            if draft is not None:
                self._persist(name, None)
        if draft is None:
            raise KeyError(f"No pending draft named '{name}'")
        return draft["content"]

    def approve(self, name: str) -> str:
        content = self.take(name)

        path = os.path.join(self.vault_dir, name)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def discard(self, name: str) -> None:
        with self._lock:
            if name not in self._drafts:
                raise KeyError(f"No pending draft named '{name}'")
            del self._drafts[name]
            self._persist(name, None)

    def approve_all(self, source: str) -> List[str]:
        """Approves every pending draft tagged with `source`. Returns the
        vault-relative paths written, in the order they were approved."""
        paths = []
        for name in self.list_for_source(source):
            paths.append(self.approve(name))
        return paths

    def discard_all(self, source: str) -> List[str]:
        """Discards every pending draft tagged with `source`. Returns the
        names discarded."""
        names = self.list_for_source(source)
        for name in names:
            self.discard(name)
        return names
