"""
Where Quill keeps everything, resolved in one place.

Precedence for each location: command-line option > environment variable >
default. Defaults: the "home" folder is the current directory if it already
holds a Quill layout (obsidian_vault/ or pdf_drop/), else ~/Quill -- so
running `quill` from some unrelated folder no longer silently starts a new,
empty vault there.

Quill's own bookkeeping (corpus, search index, version history, pending
drafts, watch state) lives in <vault>/.quill/, a hidden folder Obsidian
ignores, instead of corpus.json and chroma_index/ cluttering the vault.
"""
import os
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QuillPaths:
    home: str
    vault: str
    drop: str
    data: str

    @property
    def corpus(self) -> str:
        return os.path.join(self.data, "corpus.json")

    @property
    def index(self) -> str:
        return os.path.join(self.data, "chroma_index")

    @property
    def sessions_db(self) -> str:
        return os.path.join(self.home, "chat_history.db")

    @property
    def pending_drafts(self) -> str:
        return os.path.join(self.data, "pending_drafts")

    @property
    def watch_jobs(self) -> str:
        return os.path.join(self.data, "watch_jobs.json")

    @property
    def watch_lock(self) -> str:
        return os.path.join(self.data, "watch.lock")


def default_home(cwd: Optional[str] = None) -> str:
    cwd = cwd or os.getcwd()
    if os.path.isdir(os.path.join(cwd, "obsidian_vault")) or os.path.isdir(os.path.join(cwd, "pdf_drop")):
        return cwd
    return os.path.join(os.path.expanduser("~"), "Quill")


def resolve_paths(home: Optional[str] = None, vault: Optional[str] = None, drop: Optional[str] = None) -> QuillPaths:
    home = os.path.abspath(home or os.environ.get("QUILL_HOME") or default_home())
    vault = os.path.abspath(vault or os.environ.get("QUILL_VAULT") or os.path.join(home, "obsidian_vault"))
    drop = os.path.abspath(drop or os.environ.get("QUILL_DROP_DIR") or os.path.join(home, "pdf_drop"))
    return QuillPaths(home=home, vault=vault, drop=drop, data=os.path.join(vault, ".quill"))


def migrate_legacy_layout(paths: QuillPaths) -> list:
    """Moves files older versions kept directly in the vault into
    <vault>/.quill/. Only moves when the new location is still empty, so it
    never overwrites anything. Returns the names moved."""
    moved = []
    os.makedirs(paths.data, exist_ok=True)
    for name in ("corpus.json", "corpus.manifest.json", "chroma_index"):
        old, new = os.path.join(paths.vault, name), os.path.join(paths.data, name)
        if os.path.exists(old) and not os.path.exists(new):
            shutil.move(old, new)
            moved.append(name)
    return moved
