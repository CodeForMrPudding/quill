import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quill.helpers.note_autosnapshot import NoteAutoSnapshotter
from quill.helpers.note_versioning import NoteVersionControl


def write(vault, rel, content):
    path = os.path.join(vault, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def make(tmp_path, quiet=0.3):
    vcs = NoteVersionControl(str(tmp_path))
    return NoteAutoSnapshotter(vcs, quiet_seconds=quiet, poll_interval=0.05), vcs, str(tmp_path)


def test_start_catches_up_on_offline_edits_and_takes_daily_snapshot(tmp_path):
    snap, vcs, vault = make(tmp_path)
    write(vault, "tracked.md", "v1\n")
    vcs.commit("tracked.md", "v1")
    write(vault, "tracked.md", "edited while quill was off\n")
    write(vault, "brand_new.md", "new\n")

    snap.start()
    try:
        assert vcs.status("tracked.md") == [("tracked.md", "clean")]
        assert vcs.status("brand_new.md") == [("brand_new.md", "clean")]
        assert len(vcs.list_snapshots()) == 1
    finally:
        snap.stop()


def test_live_edit_is_committed_once_quiet(tmp_path):
    snap, vcs, vault = make(tmp_path)
    snap.start()
    try:
        for i in range(3):  # a burst of saves, like Obsidian while typing
            write(vault, "note.md", f"draft {i}\n")
            time.sleep(0.05)
        assert wait_for(lambda: vcs.status("note.md") == [("note.md", "clean")])
        assert len(vcs.log("note.md")) == 1  # one version for the burst, not three
    finally:
        snap.stop()


def test_rename_carries_history(tmp_path):
    snap, vcs, vault = make(tmp_path)
    write(vault, "old.md", "v1\n")
    snap.start()
    try:
        os.rename(os.path.join(vault, "old.md"), os.path.join(vault, "new.md"))
        assert wait_for(lambda: len(vcs.log("new.md")) == 1)
        assert vcs.log("old.md") == []
    finally:
        snap.stop()


def test_stop_flushes_edits_still_in_quiet_period(tmp_path):
    snap, vcs, vault = make(tmp_path, quiet=60)
    snap.start()
    write(vault, "note.md", "typed right before exit\n")
    assert wait_for(lambda: "note.md" in snap._pending)
    snap.stop()
    assert vcs.status("note.md") == [("note.md", "clean")]


def test_ignores_obsidian_config_and_quill_internals(tmp_path):
    snap, vcs, vault = make(tmp_path)
    snap.start()
    try:
        write(vault, ".obsidian/workspace.md", "config\n")
        write(vault, "chroma_index/x.md", "index\n")
        time.sleep(0.6)
        assert vcs.log(".obsidian/workspace.md") == []
        assert vcs.log("chroma_index/x.md") == []
    finally:
        snap.stop()


def test_no_daily_snapshot_when_recent_one_exists(tmp_path):
    vcs = NoteVersionControl(str(tmp_path))
    write(str(tmp_path), "a.md", "a\n")
    vcs.snapshot("manual")
    snap = NoteAutoSnapshotter(vcs, snapshot_age=timedelta(hours=24))
    snap.start()
    snap.stop()
    assert len(vcs.list_snapshots()) == 1
