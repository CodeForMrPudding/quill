import os
import sys
import json
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quill.agents.version_agent import VersionAgent, parse_time
from quill.helpers.note_versioning import NoteVersionControl


class FakeResponse:
    def __init__(self, content):
        self.content = content


class ScriptedModel:
    """Returns canned replies in order and records the prompts it saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        reply = self.replies.pop(0)
        return FakeResponse(reply if isinstance(reply, str) else json.dumps(reply))


def write(vault, rel, content):
    path = os.path.join(vault, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read(vault, rel):
    with open(os.path.join(vault, rel), encoding="utf-8") as f:
        return f.read()


def make(tmp_path, *replies):
    vcs = NoteVersionControl(str(tmp_path))
    return VersionAgent(ScriptedModel(*replies), vcs), vcs, str(tmp_path)


def test_plan_resolves_loose_note_name_and_lists_candidates(tmp_path):
    agent, vcs, vault = make(tmp_path, {"action": "history", "note": "meeting_2pdf__financials"})
    write(vault, "Meeting_2pdf__Financials.md", "x\n")

    plan = agent.plan("show the history of the financials note")

    assert plan == {"action": "history", "note": "Meeting_2pdf__Financials.md"}
    assert "Meeting_2pdf__Financials.md" in agent.model.prompts[0]


def test_plan_rejects_unknown_note_and_action(tmp_path):
    agent, _, _ = make(tmp_path, {"action": "diff", "note": "Ghost.md"}, {"action": "rm -rf"}, "no json at all")
    assert agent.plan("diff ghost")["action"] == "unknown"
    assert agent.plan("delete everything")["action"] == "unknown"
    assert agent.plan("???")["action"] == "unknown"


def test_plan_uses_recent_note_when_model_omits_it(tmp_path):
    agent, _, vault = make(tmp_path, {"action": "restore_note", "steps_back": 1})
    write(vault, "a.md", "x\n")
    assert agent.plan("undo that", recent_note="a.md")["note"] == "a.md"


def test_undo_after_saved_edit_goes_back_one_version(tmp_path):
    agent, vcs, vault = make(tmp_path)
    vcs.save("a.md", "original\n", "created")
    vcs.save("a.md", "chat edit\n", "edit")

    reply = agent.execute({"action": "restore_note", "note": "a.md", "steps_back": 1}, confirm=lambda q: True)

    assert read(vault, "a.md") == "original\n"
    assert "Restored" in reply


def test_undo_with_unsaved_edits_returns_to_last_saved_version_and_stashes(tmp_path):
    agent, vcs, vault = make(tmp_path)
    vcs.save("a.md", "v1\n", "v1")
    vcs.save("a.md", "v2\n", "v2")
    write(vault, "a.md", "unsaved typing\n")

    reply = agent.execute({"action": "restore_note", "note": "a.md", "steps_back": 1}, confirm=lambda q: True)

    assert read(vault, "a.md") == "v2\n"
    assert "stash" in reply
    assert len(vcs.stash_list("a.md")) == 1


def test_restore_is_cancelled_when_not_confirmed(tmp_path):
    agent, vcs, vault = make(tmp_path)
    vcs.save("a.md", "v1\n", "v1")
    vcs.save("a.md", "v2\n", "v2")
    asked = []

    reply = agent.execute({"action": "restore_note", "note": "a.md", "steps_back": 1},
                          confirm=lambda q: asked.append(q) or False)

    assert read(vault, "a.md") == "v2\n"
    assert "Cancelled" in reply and asked


def test_restore_note_before_time_with_timezone_suffix(tmp_path):
    agent, vcs, vault = make(tmp_path)
    first = vcs.save("a.md", "v1\n", "v1")
    vcs.save("a.md", "v2\n", "v2")
    # Backdate v1 so "before an hour ago" selects it.
    old = (datetime.now() - timedelta(hours=3)).isoformat()
    vcs.conn.execute("UPDATE commits SET timestamp = ? WHERE id = ?", (old, first))
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    agent.execute({"action": "restore_note", "note": "a.md", "before": cutoff}, confirm=lambda q: True)

    assert read(vault, "a.md") == "v1\n"


def test_bad_model_values_produce_an_error_reply_not_a_crash(tmp_path):
    agent, vcs, _ = make(tmp_path)
    vcs.save("a.md", "v1\n", "v1")
    for plan in [
        {"action": "restore_note", "note": "a.md", "commit_id": "abc"},
        {"action": "restore_note", "note": "a.md", "before": "yesterday-ish"},
        {"action": "restore_note", "note": "a.md", "steps_back": 5},
        {"action": "restore_vault", "snapshot_id": 42},
    ]:
        assert agent.execute(plan, confirm=lambda q: True).startswith("Couldn't do that")


def test_save_whole_vault_then_restore_vault_snapshot(tmp_path):
    agent, vcs, vault = make(tmp_path)
    write(vault, "a.md", "a1\n")
    reply = agent.execute({"action": "save", "note": None, "message": "before big edit"}, confirm=lambda q: True)
    assert "snapshot" in reply
    write(vault, "a.md", "a2\n")

    reply = agent.execute({"action": "restore_vault", "snapshot_id": None}, confirm=lambda q: True)

    assert read(vault, "a.md") == "a1\n"
    assert "To undo this restore" in reply


def test_status_and_history_formatting(tmp_path):
    agent, vcs, vault = make(tmp_path)
    vcs.save("a.md", "v1\n", "first")
    write(vault, "b.md", "new\n")

    status = agent.execute({"action": "status"}, confirm=lambda q: True)
    assert "untracked" in status and "b.md" in status

    history = agent.format_history("a.md")
    assert "first" in history and "<- current" in history


def test_parse_time_normalizes_to_naive_local():
    parsed = parse_time("2026-10-29T12:00:00Z")
    assert parsed.tzinfo is None
    assert parse_time("2026-10-29") == datetime(2026, 10, 29)


def test_pin_restore_target_prefers_users_own_words():
    from quill.agents.version_agent import _pin_restore_target
    note_plan = {"action": "restore_note", "note": "a.md", "steps_back": 1, "before": "2026-10-30T02:10", "commit_id": None}

    # Seen live: "undo that" with a spurious before=<now> must stay an undo.
    assert _pin_restore_target(note_plan, "undo that")["before"] is None
    assert _pin_restore_target(note_plan, "I am not happy with it, undo that")["before"] is None
    # Seen live: "version 3" came back as steps_back=1.
    pinned = _pin_restore_target(note_plan, "revert it to version 3")
    assert (pinned["commit_id"], pinned["steps_back"], pinned["before"]) == (3, None, None)
    # A time in the request makes "before" the target.
    pinned = _pin_restore_target(note_plan, "put it back to how it was yesterday")
    assert pinned["steps_back"] is None and pinned["before"]
    # "#2" in a filename isn't a version number.
    assert _pin_restore_target(note_plan, "undo the Meeting #2 financials note")["commit_id"] is None

    vault_plan = {"action": "restore_vault", "snapshot_id": 1, "before": "2026-10-29T23:59"}
    assert _pin_restore_target(vault_plan, "restore the vault to yesterday")["snapshot_id"] is None
    assert _pin_restore_target(vault_plan, "restore snapshot 4")["snapshot_id"] == 4
    assert _pin_restore_target(vault_plan, "restore my vault")["before"] is None
