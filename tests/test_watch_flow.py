import os
import sys
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quill.agents.note_agent import NoteAgent
from quill.helpers.instance_lock import InstanceLock
from quill.helpers.note_versioning import NoteVersionControl
from quill.helpers.pdf_ingestion import _file_signature
from quill.helpers.vault_watcher import wait_until_stable
from quill.helpers.watch_flow import PdfWatchFlow
from quill.helpers.watch_jobs import WatchJobStore


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, fail_on=None):
        self.prompts = []
        self.fail_on = fail_on

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("model crashed")
        return FakeResponse(f"## Draft {len(self.prompts)}")


class FakeNotifier:
    def __init__(self, answer=True):
        self.answer = answer
        self.asked, self.notified = [], []
        self.answered = threading.Event()

    def ask_yes_no(self, title, message, timeout=0):
        self.asked.append(message)
        self.answered.set()
        return self.answer

    def notify(self, title, message):
        self.notified.append(message)


def setup(tmp_path, chapters=("Chapter 1", "Chapter 2", "Chapter 3"), answer=True, fail_on=None):
    vault, drop = tmp_path / "vault", tmp_path / "drop"
    vault.mkdir()
    drop.mkdir()
    (drop / "book.pdf").write_bytes(b"%PDF fake")
    data = vault / ".quill"
    data.mkdir()
    corpus = [{"text": f"Context: {c}\n\ntext of {c}", "source": "book.pdf", "header1": c, "header2": "",
               "page": i, "idx": i, "title": f"t{i}"} for i, c in enumerate(chapters)]
    (data / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    # Manifest matching the PDF, so ingestion treats it as already parsed.
    (data / "corpus.manifest.json").write_text(json.dumps(
        {"book.pdf": {"sig": _file_signature(str(drop / "book.pdf")), "chunks": len(chapters)}}), encoding="utf-8")

    agent = NoteAgent.__new__(NoteAgent)
    agent.model, agent.directory, agent.pdf_source_dir = FakeModel(fail_on), str(vault), str(drop)
    agent.data_dir, agent.corpus_file = str(data), str(data / "corpus.json")
    agent.vector_store, agent._setup_lock = None, threading.Lock()
    vcs = NoteVersionControl(str(vault))
    notifier = FakeNotifier(answer)
    jobs = WatchJobStore(str(data / "watch_jobs.json"))
    flow = PdfWatchFlow(agent, vcs, notifier, jobs, log=lambda *a, **k: None)
    return flow, agent, vcs, notifier, jobs, vault


def test_approved_pdf_is_drafted_with_overview_and_marked_done(tmp_path):
    flow, agent, vcs, notifier, jobs, vault = setup(tmp_path)
    flow.start()
    try:
        flow.catch_up()
        assert notifier.answered.wait(5)
        assert flow.wait_idle(10)
    finally:
        flow.stop()

    assert jobs.get("book.pdf")["status"] == "done"
    names = sorted(os.listdir(vault))
    assert "bookpdf__Chapter_1.md" in names and "bookpdf__index.md" in names
    index = (vault / "bookpdf__index.md").read_text(encoding="utf-8")
    assert "## Overview" in index
    assert "Finished generating 3" in notifier.notified[-1]


def test_interrupted_job_resumes_without_asking_again(tmp_path):
    flow, agent, vcs, notifier, jobs, vault = setup(tmp_path)
    sig = _file_signature(str(tmp_path / "drop" / "book.pdf"))
    jobs.start("book.pdf", sig, ["Chapter 1", "Chapter 2", "Chapter 3"])
    jobs.chapter_done("book.pdf", "Chapter 1")  # crashed after chapter 1

    flow.start()
    try:
        flow.catch_up()
        assert flow.wait_idle(10)
    finally:
        flow.stop()

    assert notifier.asked == []  # already approved earlier
    drafted = [p for p in agent.model.prompts if "chapter/section" in p]
    assert len(drafted) == 2 and not any("'Chapter 1'" in p for p in drafted)
    assert jobs.get("book.pdf")["status"] == "done"


def test_failed_chapter_is_reported_and_retried_next_start(tmp_path):
    flow, agent, vcs, notifier, jobs, vault = setup(tmp_path, fail_on="text of Chapter 2")
    flow.start()
    try:
        flow.catch_up()
        assert notifier.answered.wait(5)
        assert flow.wait_idle(10)
    finally:
        flow.stop()

    job = jobs.get("book.pdf")
    assert job["status"] == "in_progress"
    assert list(job["chapters_failed"]) == ["Chapter 2"]
    assert "2 of 3" in notifier.notified[-1] and "1 failed" in notifier.notified[-1]
    assert WatchJobStore(jobs.path).needs_attention("book.pdf", job["sig"], has_notes=True)


def test_declined_pdf_not_asked_again_until_it_changes(tmp_path):
    flow, agent, vcs, notifier, jobs, vault = setup(tmp_path, answer=False)
    flow.start()
    try:
        flow.catch_up()
        assert notifier.answered.wait(5)
        assert flow.wait_idle(5)
    finally:
        flow.stop()
    sig = _file_signature(str(tmp_path / "drop" / "book.pdf"))
    assert jobs.get("book.pdf")["status"] == "declined"
    assert not jobs.needs_attention("book.pdf", sig, has_notes=False)
    assert jobs.needs_attention("book.pdf", [sig[0] + 1, sig[1]], has_notes=False)


def test_unanswered_toast_is_not_recorded_as_declined(tmp_path):
    flow, agent, vcs, notifier, jobs, vault = setup(tmp_path, answer=None)
    flow.start()
    try:
        flow.catch_up()
        assert notifier.answered.wait(5)
        assert flow.wait_idle(5)
    finally:
        flow.stop()
    assert jobs.get("book.pdf") is None  # asked again next start


def test_scanned_pdf_explains_instead_of_drafting(tmp_path):
    flow, agent, vcs, notifier, jobs, vault = setup(tmp_path, chapters=())
    (tmp_path / "vault" / ".quill" / "corpus.manifest.json").write_text(
        json.dumps({"book.pdf": {"sig": [1, 2], "chunks": 0, "scanned": True}}), encoding="utf-8")
    flow.offer("book.pdf")
    assert notifier.asked == []
    assert "OCR" in notifier.notified[-1]


def test_inbox_import_copies_only_after_yes_and_never_overwrites(tmp_path):
    flow, agent, vcs, notifier, jobs, vault = setup(tmp_path)
    inbox = tmp_path / "Downloads"
    inbox.mkdir()
    (inbox / "book.pdf").write_bytes(b"%PDF a different, longer book")

    name = flow.import_into_library(str(inbox / "book.pdf"))

    assert name == "book (2).pdf"
    assert (tmp_path / "drop" / "book.pdf").read_bytes() == b"%PDF fake"
    assert flow.import_into_library(str(inbox / "book.pdf")) == "book (2).pdf"  # identical re-import reused


def test_instance_lock_is_exclusive(tmp_path):
    a, b = InstanceLock(str(tmp_path / "w.lock")), InstanceLock(str(tmp_path / "w.lock"))
    assert a.acquire() and not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_wait_until_stable_has_no_fixed_time_limit_but_honours_stop(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    assert wait_until_stable(str(empty), poll_interval=0.01, empty_timeout=0.05) is False
    stop = threading.Event()
    stop.set()
    assert wait_until_stable(str(empty), poll_interval=0.01, stop_event=stop) is False
