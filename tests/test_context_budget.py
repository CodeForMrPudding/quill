import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from quill.helpers.llm_config import input_char_budget, make_chat_model, trim_history, NUM_CTX
from quill.helpers.note_planner import group_drafts_for_merge
from quill.agents.note_agent import NoteAgent, MERGE_CHAR_BUDGET, EDIT_MAX_NOTE_CHARS


def test_input_char_budget_leaves_room_for_reply():
    assert input_char_budget(0, num_ctx=4000) > input_char_budget(2000, num_ctx=4000) > 0
    with pytest.raises(ValueError):
        input_char_budget(4000, num_ctx=4000)


def test_chat_model_sets_context_window_explicitly():
    # The bug being fixed: without num_ctx, Ollama ran llama3.1 at 4096
    # tokens and silently dropped the start of longer prompts.
    assert make_chat_model().num_ctx == NUM_CTX


def test_budgets_derive_from_context_window():
    assert MERGE_CHAR_BUDGET == input_char_budget(2048)
    assert EDIT_MAX_NOTE_CHARS > 8000  # was a fixed 8000 before num_ctx was set


def test_trim_history_keeps_most_recent_that_fit():
    history = [HumanMessage(content="a" * 50), AIMessage(content="b" * 50), HumanMessage(content="c" * 50)]
    assert [m.content[0] for m in trim_history(history, 120)] == ["b", "c"]
    assert [m.content[0] for m in trim_history(history, 10)] == ["c"]  # always keeps the latest
    assert trim_history([], 100) == []


def test_group_drafts_for_merge_respects_budget_and_max_group():
    drafts = ["x" * 100] * 7
    groups = group_drafts_for_merge(drafts, char_budget=350, max_group=5)
    assert [len(g) for g in groups] == [3, 3, 1]
    groups = group_drafts_for_merge(drafts, char_budget=10_000, max_group=5)
    assert [len(g) for g in groups] == [5, 2]


def test_group_drafts_for_merge_always_makes_progress():
    drafts = ["x" * 500] * 3
    groups = group_drafts_for_merge(drafts, char_budget=100, max_group=5)
    assert [len(g) for g in groups] == [2, 1]


class FakeResponse:
    def __init__(self, content):
        self.content = content


class RecordingModel:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return FakeResponse(f"merged {len(self.prompts)}")


def test_reduce_drafts_terminates_and_puts_instructions_last(tmp_path):
    agent = NoteAgent.__new__(NoteAgent)
    agent.model = RecordingModel()

    result = agent._reduce_drafts("Chapter 1", [f"draft {i}" for i in range(12)])

    assert result.startswith("merged")
    first = agent.model.prompts[0]
    assert first.index("draft 0") < first.index("Merge them")
    assert len(agent.model.prompts) == 4  # 12 -> 3 (5+5+2) -> 1


def test_draft_batch_puts_instructions_after_source(tmp_path):
    agent = NoteAgent.__new__(NoteAgent)
    agent.model = RecordingModel()
    agent._draft_batch("Chapter 1", [{"text": "SOURCE TEXT"}])
    prompt = agent.model.prompts[0]
    assert prompt.index("SOURCE TEXT") < prompt.index("Write a detailed")
