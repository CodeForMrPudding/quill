import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quill import cli
from quill.agents.note_agent import NoteAgent, EDIT_MAX_NOTE_CHARS
from quill.helpers.note_versioning import NoteVersionControl


class FakeResponse:
    def __init__(self, content):
        self.content = content


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content if isinstance(messages, list) else messages)
        reply = self.replies.pop(0)
        return FakeResponse(reply if isinstance(reply, str) else json.dumps(reply))


def make_agent(tmp_path, *replies):
    corpus_file = os.path.join(str(tmp_path), "corpus.json")
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump([], f)
    agent = NoteAgent.__new__(NoteAgent)
    agent.model = ScriptedModel(*replies)
    agent.directory = str(tmp_path)
    agent.corpus_file = corpus_file
    agent.vector_store = None
    return agent


def write(vault, rel, content):
    with open(os.path.join(vault, rel), "w", encoding="utf-8") as f:
        f.write(content)


def read(vault, rel):
    with open(os.path.join(vault, rel), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("reply, expected", [
    ("EDIT", "EDIT"),
    ("Category: VERSION.", "VERSION"),
    ("Not EXIT -- this is CHAT", "CHAT"),  # old substring check returned EXIT here
    ("'NOTE'", "NOTE"),
    ("I think the user is chatting", "CHAT"),
    ("GIT", "CHAT"),  # retired category
])
def test_parse_intent(reply, expected):
    assert cli.parse_intent(reply) == expected


def test_edit_note_puts_instructions_after_note_and_strips_wrapping(tmp_path):
    agent = make_agent(tmp_path, "Here is the revised note:\n```markdown\n# T\n\nshorter\n```")

    revised = agent.edit_note("a.md", "# T\n\nlong body\n", "make it shorter")

    assert revised == "# T\n\nshorter\n"
    prompt = agent.model.prompts[0]
    assert prompt.index("long body") < prompt.index("make it shorter")


def test_edit_note_refuses_oversized_notes(tmp_path):
    agent = make_agent(tmp_path)
    with pytest.raises(ValueError):
        agent.edit_note("big.md", "x" * (EDIT_MAX_NOTE_CHARS + 1), "summarize")
    assert agent.model.prompts == []


def test_handle_edit_request_saves_a_version_that_undo_can_revert(tmp_path, monkeypatch):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write(vault, "Meeting_2pdf__Financials.md", "# Financials\n\nRevenue grew.\n")
    write(vault, "Other.md", "unrelated\n")
    agent = make_agent(
        tmp_path,
        {"note": "Meeting_2pdf__Financials.md", "instruction": "add a cash flow section"},
        "# Financials\n\nRevenue grew.\n\n## Cash flow\n\nPositive.\n",
    )
    monkeypatch.setattr(cli.click, "confirm", lambda *a, **k: True)

    reply, note = cli.handle_edit_request("in the financials note add a cash flow section", agent, vcs, None)

    assert note == "Meeting_2pdf__Financials.md"
    assert "## Cash flow" in read(vault, note)
    # Pre-edit content was untracked; save() kept it as its own version.
    assert len(vcs.log(note)) == 2
    assert "undo" in reply


def test_handle_edit_request_declined_leaves_note_untouched(tmp_path, monkeypatch):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write(vault, "a.md", "original\n")
    agent = make_agent(tmp_path, {"note": None, "instruction": "rewrite"}, "rewritten\n")
    monkeypatch.setattr(cli.click, "confirm", lambda *a, **k: False)

    reply, note = cli.handle_edit_request("rewrite it", agent, vcs, recent_note="a.md")

    assert note == "a.md"  # "it" resolved via the recent note
    assert read(vault, "a.md") == "original\n"
    assert "discarded" in reply


def test_handle_edit_request_unknown_note(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    write(vault, "a.md", "x\n")
    agent = make_agent(tmp_path, {"note": "Missing.md", "instruction": "x"})

    reply, note = cli.handle_edit_request("edit the missing note", agent, vcs, None)

    assert note is None and "couldn't tell which note" in reply


def test_create_note_with_vcs_starts_as_tracked_version(tmp_path):
    vault = str(tmp_path)
    vcs = NoteVersionControl(vault)
    agent = make_agent(tmp_path, "# Topic\n\nBody\n")

    path = agent.create_note("Linear Algebra", vcs=vcs)

    assert os.path.basename(path) == "Linear_Algebra.md"
    assert vcs.status("Linear_Algebra.md") == [("Linear_Algebra.md", "clean")]
