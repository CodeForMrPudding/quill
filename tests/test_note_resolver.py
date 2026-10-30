import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quill.helpers.note_resolver import (
    candidate_notes,
    list_vault_notes,
    match_note,
    parse_json_object,
    strip_llm_wrapping,
)


def test_list_vault_notes_skips_internal_dirs_newest_first(tmp_path):
    for rel in ["old.md", "sub/new.md", ".obsidian/x.md", ".quill/y.md", "chroma_index/z.md", "corpus.json"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    os.utime(tmp_path / "old.md", (time.time() - 100, time.time() - 100))

    assert list_vault_notes(str(tmp_path)) == ["sub/new.md", "old.md"]


def test_candidate_notes_ranks_name_matches_first():
    notes = ["Meeting_2pdf__Management.md", "Meeting_2pdf__Financials.md", "Recipes.md"]
    assert candidate_notes("add cash flow to the financials note", notes)[0] == "Meeting_2pdf__Financials.md"


def test_candidate_notes_tolerates_typos_and_puts_recent_first():
    notes = ["Linear_Algebra.md", "Eigenvalues.md", "Groceries.md"]
    assert candidate_notes("expand the eigenvalus note", notes)[0] == "Eigenvalues.md"
    assert candidate_notes("make it shorter", notes, recent="Groceries.md")[0] == "Groceries.md"


def test_candidate_notes_respects_limit():
    notes = [f"note_{i}.md" for i in range(40)]
    assert len(candidate_notes("anything", notes, limit=15)) == 15


def test_match_note_variants():
    notes = ["Meeting_2pdf__Financials.md", "folder/Ideas.md"]
    assert match_note("Meeting_2pdf__Financials.md", notes) == "Meeting_2pdf__Financials.md"
    assert match_note("meeting_2pdf__financials", notes) == "Meeting_2pdf__Financials.md"
    assert match_note("[[Ideas]]", notes) == "folder/Ideas.md"
    assert match_note("Nope.md", notes) is None
    assert match_note(None, notes) is None


def test_parse_json_object_tolerates_fences_and_chatter():
    reply = 'Sure! Here you go:\n```json\n{"action": "save", "note": null, "message": "a {brace} in text"}\n```'
    assert parse_json_object(reply) == {"action": "save", "note": None, "message": "a {brace} in text"}


def test_parse_json_object_skips_invalid_candidates():
    assert parse_json_object('{not json} then {"ok": 1}') == {"ok": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_strip_llm_wrapping_removes_preamble_and_fences():
    assert strip_llm_wrapping("Here is the revised note:\n```markdown\n# Title\n\nBody\n```") == "# Title\n\nBody\n"
    assert strip_llm_wrapping("# Title\n\nBody") == "# Title\n\nBody\n"
