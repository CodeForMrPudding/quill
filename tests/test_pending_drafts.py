import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quill.helpers.pending_drafts import PendingDraftStore


def test_add_list_and_approve_writes_file(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    store.add("topic.md", "# Topic\n\nBody text.", source="topic.pdf")

    assert store.list() == [("topic.md", "topic.pdf")]

    path = store.approve("topic.md")
    assert path == os.path.join(str(tmp_path), "topic.md")
    with open(path, "r", encoding="utf-8") as f:
        assert f.read() == "# Topic\n\nBody text."

    # Approving pops it from the pending list.
    assert store.list() == []


def test_discard_removes_without_writing(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    store.add("topic.md", "content", source="topic.pdf")

    store.discard("topic.md")

    assert store.list() == []
    assert not os.path.exists(os.path.join(str(tmp_path), "topic.md"))


def test_approve_unknown_draft_raises(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    with pytest.raises(KeyError):
        store.approve("nonexistent.md")


def test_discard_unknown_draft_raises(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    with pytest.raises(KeyError):
        store.discard("nonexistent.md")


def test_unique_name_avoids_collision_with_pending_drafts(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    name1 = store.unique_name("Linear Algebra")
    store.add(name1, "content1")

    name2 = store.unique_name("Linear Algebra")

    assert name1 != name2
    assert name1 == "Linear_Algebra.md"
    assert name2 == "Linear_Algebra_2.md"


def test_unique_name_avoids_collision_with_existing_vault_file(tmp_path):
    with open(os.path.join(str(tmp_path), "Calculus.md"), "w", encoding="utf-8") as f:
        f.write("existing note")

    store = PendingDraftStore(str(tmp_path))
    name = store.unique_name("Calculus")

    assert name == "Calculus_2.md"


def test_unique_name_sanitizes_topic():
    store = PendingDraftStore(os.getcwd())
    name = store.unique_name("What's 2+2? / A Study!!")
    # No path separators, quotes, or other characters unsafe for a filename.
    assert "/" not in name
    assert "'" not in name
    assert name.endswith(".md")


def test_list_for_source_filters_correctly(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    store.add("ch1.md", "content1", source="book.pdf")
    store.add("ch2.md", "content2", source="book.pdf")
    store.add("other.md", "content3", source="other.pdf")

    assert store.list_for_source("book.pdf") == ["ch1.md", "ch2.md"]
    assert store.list_for_source("other.pdf") == ["other.md"]
    assert store.list_for_source("nonexistent.pdf") == []


def test_approve_all_writes_every_draft_for_source(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    store.add("ch1.md", "chapter one", source="book.pdf")
    store.add("ch2.md", "chapter two", source="book.pdf")
    store.add("other.md", "unrelated", source="other.pdf")

    paths = store.approve_all("book.pdf")

    assert len(paths) == 2
    assert os.path.exists(os.path.join(str(tmp_path), "ch1.md"))
    assert os.path.exists(os.path.join(str(tmp_path), "ch2.md"))
    # Unrelated source untouched, still pending.
    assert store.list() == [("other.md", "other.pdf")]


def test_discard_all_removes_every_draft_for_source(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    store.add("ch1.md", "chapter one", source="book.pdf")
    store.add("ch2.md", "chapter two", source="book.pdf")
    store.add("other.md", "unrelated", source="other.pdf")

    discarded = store.discard_all("book.pdf")

    assert sorted(discarded) == ["ch1.md", "ch2.md"]
    assert store.list() == [("other.md", "other.pdf")]
    assert not os.path.exists(os.path.join(str(tmp_path), "ch1.md"))


def test_approve_all_empty_source_returns_empty_list(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    assert store.approve_all("nonexistent.pdf") == []
    assert store.discard_all("nonexistent.pdf") == []


def test_take_returns_content_without_writing(tmp_path):
    store = PendingDraftStore(str(tmp_path))
    store.add("topic.md", "content", source="topic.pdf")

    assert store.take("topic.md") == "content"
    assert store.list() == []
    assert not os.path.exists(os.path.join(str(tmp_path), "topic.md"))
    with pytest.raises(KeyError):
        store.take("topic.md")
