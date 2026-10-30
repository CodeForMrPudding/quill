import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quill.helpers.note_versioning import NoteVersionControl


def write_note(vault, rel_path, content):
    abs_path = os.path.join(vault, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


def read_note(vault, rel_path):
    with open(os.path.join(vault, rel_path), "r", encoding="utf-8") as f:
        return f.read()


def test_commit_and_checkout_round_trip(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    write_note(vault, "note.md", "line one\nline two\n")
    first_id = vcs.commit("note.md", "initial")

    write_note(vault, "note.md", "line one\nline TWO edited\nline three\n")
    second_id = vcs.commit("note.md", "edit line two, add line three")

    vcs.checkout("note.md", first_id)
    assert read_note(vault, "note.md") == "line one\nline two\n"

    vcs.checkout("note.md", second_id)
    assert read_note(vault, "note.md") == "line one\nline TWO edited\nline three\n"


def test_dirty_checkout_auto_stashes_and_pops(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    write_note(vault, "note.md", "v1\n")
    first_id = vcs.commit("note.md", "v1")

    write_note(vault, "note.md", "v2\n")
    vcs.commit("note.md", "v2")

    # Uncommitted edit on top of v2
    write_note(vault, "note.md", "v2 with uncommitted edits\n")

    vcs.checkout("note.md", first_id)
    assert read_note(vault, "note.md") == "v1\n"

    stashes = vcs.stash_list("note.md")
    assert len(stashes) == 1

    vcs.stash_pop("note.md")
    assert read_note(vault, "note.md") == "v2 with uncommitted edits\n"
    assert vcs.stash_list("note.md") == []


def test_status_reports_untracked_clean_modified(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    write_note(vault, "tracked.md", "hello\n")
    write_note(vault, "loose.md", "not committed\n")

    assert vcs.status("tracked.md") == [("tracked.md", "untracked")]

    vcs.commit("tracked.md", "first")
    assert vcs.status("tracked.md") == [("tracked.md", "clean")]

    write_note(vault, "tracked.md", "hello, edited\n")
    assert vcs.status("tracked.md") == [("tracked.md", "modified")]

    all_status = dict(vcs.status())
    assert all_status["tracked.md"] == "modified"
    assert all_status["loose.md"] == "untracked"


def test_commit_with_no_changes_raises(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    write_note(vault, "note.md", "same content\n")
    vcs.commit("note.md", "first")

    with pytest.raises(ValueError):
        vcs.commit("note.md", "no-op")


def test_log_orders_newest_first(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    write_note(vault, "note.md", "a\n")
    id1 = vcs.commit("note.md", "first")
    write_note(vault, "note.md", "a\nb\n")
    id2 = vcs.commit("note.md", "second")
    write_note(vault, "note.md", "a\nb\nc\n")
    id3 = vcs.commit("note.md", "third")

    entries = vcs.log("note.md")
    assert [e[0] for e in entries] == [id3, id2, id1]


def test_delta_storage_stays_proportional_to_edit_size(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    base_line = "the quick brown fox jumps over the lazy dog\n"
    content = base_line * 500  # sizeable base file
    write_note(vault, "big.md", content)
    vcs.commit("big.md", "initial big snapshot")
    file_size = len(content.encode("utf-8"))

    # Small single-line edits, staying under SNAPSHOT_INTERVAL so every one of
    # these commits is a pure diff (no periodic full snapshot in the mix).
    diff_sizes = []
    for i in range(8):
        content += f"appended line {i}\n"
        write_note(vault, "big.md", content)
        vcs.commit("big.md", f"append {i}")

        cursor = vcs.conn.cursor()
        cursor.execute(
            "SELECT is_snapshot, opcodes FROM commits WHERE note_path = ? ORDER BY id DESC LIMIT 1",
            ("big.md",)
        )
        is_snapshot, opcodes = cursor.fetchone()
        assert is_snapshot == 0
        diff_sizes.append(len(opcodes.encode("utf-8")))

    # Each single-line append should cost a small fraction of the full file
    # size, not a full copy.
    assert max(diff_sizes) < file_size * 0.05


def test_reconstruct_across_snapshot_boundary(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    write_note(vault, "note.md", "start\n")
    ids = [vcs.commit("note.md", "commit 0")]

    # SNAPSHOT_INTERVAL is 10 — push past two boundaries.
    content = "start\n"
    for i in range(1, 25):
        content += f"line {i}\n"
        write_note(vault, "note.md", content)
        ids.append(vcs.commit("note.md", f"commit {i}"))

    # Reconstructing an old commit must replay correctly across snapshots.
    vcs.checkout("note.md", ids[13])
    expected = "start\n" + "".join(f"line {i}\n" for i in range(1, 14))
    assert read_note(vault, "note.md") == expected

    vcs.checkout("note.md", ids[-1])
    assert read_note(vault, "note.md") == content


def test_save_snapshots_uncommitted_edits_before_overwriting(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    write_note(vault, "note.md", "generated v1\n")
    vcs.commit("note.md", "generated")
    write_note(vault, "note.md", "generated v1\nmy own edits\n")  # never committed

    head = vcs.save("note.md", "generated v2\n", "regenerated")

    assert read_note(vault, "note.md") == "generated v2\n"
    entries = vcs.log("note.md")
    assert entries[0][0] == head
    assert len(entries) == 3  # generated, snapshot of edits, regenerated
    vcs.checkout("note.md", entries[1][0])
    assert read_note(vault, "note.md") == "generated v1\nmy own edits\n"


def test_save_snapshots_existing_untracked_note(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "note.md", "handwritten\n")

    vcs.save("note.md", "generated\n", "generated")

    entries = vcs.log("note.md")
    assert len(entries) == 2
    vcs.checkout("note.md", entries[1][0])
    assert read_note(vault, "note.md") == "handwritten\n"


def test_save_new_note_creates_single_commit(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)

    head = vcs.save("sub/new.md", "fresh\n", "created")

    assert read_note(vault, "sub/new.md") == "fresh\n"
    assert [e[0] for e in vcs.log("sub/new.md")] == [head]
    assert vcs.status("sub/new.md") == [("sub/new.md", "clean")]


def test_save_identical_content_is_a_no_op(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    first = vcs.save("note.md", "same\n", "first")

    assert vcs.save("note.md", "same\n", "again") == first
    assert len(vcs.log("note.md")) == 1


def test_stash_pop_stashes_current_edits_instead_of_overwriting(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "note.md", "v1\n")
    first = vcs.commit("note.md", "v1")
    write_note(vault, "note.md", "v2\n")
    vcs.commit("note.md", "v2")
    write_note(vault, "note.md", "edits A\n")
    vcs.checkout("note.md", first)          # stashes "edits A"
    write_note(vault, "note.md", "edits B\n")  # new unsaved edits on v1

    vcs.stash_pop("note.md")

    assert read_note(vault, "note.md") == "edits A\n"
    vcs.stash_pop("note.md")                 # "edits B" was stashed, not lost
    assert read_note(vault, "note.md") == "edits B\n"


def test_checkout_writes_lf_line_endings(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "note.md", "a\nb\n")
    first = vcs.commit("note.md", "v1")
    write_note(vault, "note.md", "a\nc\n")
    vcs.commit("note.md", "v2")

    vcs.checkout("note.md", first)

    with open(os.path.join(vault, "note.md"), "rb") as f:
        assert f.read() == b"a\nb\n"


def test_rename_moves_history_to_new_path(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "old.md", "v1\n")
    first = vcs.commit("old.md", "v1")
    write_note(vault, "old.md", "v2\n")
    vcs.commit("old.md", "v2")
    os.rename(os.path.join(vault, "old.md"), os.path.join(vault, "new.md"))

    assert vcs.rename("old.md", "new.md") is True

    assert [e[1] for e in vcs.log("new.md")] == ["v2", "v1"]
    assert vcs.log("old.md") == []
    assert vcs.status("new.md") == [("new.md", "clean")]
    vcs.checkout("new.md", first)
    assert read_note(vault, "new.md") == "v1\n"


def test_rename_refuses_to_merge_into_existing_history(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "a.md", "a\n")
    write_note(vault, "b.md", "b\n")
    vcs.commit("a.md", "a")
    vcs.commit("b.md", "b")
    assert vcs.rename("a.md", "b.md") is False
    assert vcs.rename("untracked.md", "x.md") is False


def test_diff_working_copy_and_between_commits(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "note.md", "one\ntwo\n")
    first = vcs.commit("note.md", "v1")
    write_note(vault, "note.md", "one\nTWO\n")
    second = vcs.commit("note.md", "v2")
    write_note(vault, "note.md", "one\nTWO\nthree\n")

    working = vcs.diff("note.md")
    assert "+three" in working and "-two" not in working

    between = vcs.diff("note.md", first, second)
    assert "-two" in between and "+TWO" in between


def test_ancestors_follow_head_and_latest_finds_newest(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "note.md", "v1\n")
    c1 = vcs.commit("note.md", "v1")
    write_note(vault, "note.md", "v2\n")
    c2 = vcs.commit("note.md", "v2")
    vcs.checkout("note.md", c1)
    write_note(vault, "note.md", "v3 branch\n")
    c3 = vcs.commit("note.md", "v3")

    assert [e[0] for e in vcs.ancestors("note.md")] == [c3, c1]  # c2 is on the abandoned line
    vcs.checkout("note.md", c1)
    assert vcs.latest("note.md") == c3
    assert vcs.head("note.md") == c1


def test_delta_chain_stays_bounded_when_committing_on_old_versions(tmp_path):
    # Worst case for a per-note commit *count*: every time a full copy is
    # written, check out its parent and keep committing from there, so the
    # full copies all land on abandoned lines. Measuring the actual parent
    # chain keeps replay length bounded regardless.
    from quill.helpers.note_versioning import SNAPSHOT_INTERVAL
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "note.md", "base\n")
    vcs.commit("note.md", "base")
    for i in range(5 * SNAPSHOT_INTERVAL):
        write_note(vault, "note.md", f"base\nedit {i}\n")
        cid = vcs.commit("note.md", f"edit {i}")
        assert vcs._deltas_since_full_copy(cid) < SNAPSHOT_INTERVAL
        if vcs._get_commit(cid)[5]:  # just wrote a full copy -> abandon it
            vcs.checkout("note.md", vcs._get_commit(cid)[4])


def test_commit_all_commits_only_changed_notes(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "clean.md", "x\n")
    vcs.commit("clean.md", "x")
    write_note(vault, "new.md", "new\n")
    write_note(vault, "sub/changed.md", "a\n")
    vcs.commit("sub/changed.md", "a")
    write_note(vault, "sub/changed.md", "b\n")
    write_note(vault, ".obsidian/workspace.md", "config, not a note\n")

    committed = dict(vcs.commit_all("auto"))

    assert set(committed) == {"new.md", "sub/changed.md"}


def test_vault_snapshot_restore_and_undo(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write_note(vault, "a.md", "a1\n")
    write_note(vault, "b.md", "b1\n")
    snap = vcs.snapshot("first")          # also commits both untracked notes

    write_note(vault, "a.md", "a2\n")
    os.remove(os.path.join(vault, "b.md"))
    write_note(vault, "c.md", "created later\n")

    result = vcs.restore_snapshot(snap)

    assert read_note(vault, "a.md") == "a1\n"
    assert read_note(vault, "b.md") == "b1\n"       # deleted note brought back
    assert read_note(vault, "c.md") == "created later\n"  # newer note kept, not deleted
    assert sorted(result["restored"]) == ["a.md", "b.md"]
    assert result["kept"] == ["c.md"]

    # The safety snapshot taken before restoring undoes the restore.
    vcs.restore_snapshot(result["safety_snapshot"])
    assert read_note(vault, "a.md") == "a2\n"


def test_restore_unknown_snapshot_raises(tmp_path):
    vcs = NoteVersionControl(str(tmp_path))
    with pytest.raises(ValueError):
        vcs.restore_snapshot(99)


def test_ensure_recent_snapshot_only_when_stale(tmp_path):
    from datetime import timedelta
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    assert vcs.ensure_recent_snapshot() is None  # empty vault: nothing to snapshot

    write_note(vault, "a.md", "a\n")
    first = vcs.ensure_recent_snapshot()
    assert first is not None
    assert vcs.ensure_recent_snapshot() is None
    assert vcs.ensure_recent_snapshot(max_age=timedelta(seconds=0)) is not None


def test_two_instances_on_same_vault_do_not_fork_history(tmp_path):
    # Simulates the REPL and `quill watch` (separate processes, separate
    # connections) both auto-committing the same edit.
    vault = str(tmp_path)
    a, b = NoteVersionControl(vault), NoteVersionControl(vault)
    write_note(vault, "note.md", "v1\n")
    a.commit("note.md", "v1")
    write_note(vault, "note.md", "v2\n")

    assert a.commit_if_changed("note.md", "auto") is not None
    assert b.commit_if_changed("note.md", "auto") is None
    assert len(b.log("note.md")) == 2
