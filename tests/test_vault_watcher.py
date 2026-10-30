import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quill.helpers.vault_watcher import VaultWatcher


def wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_watcher_detects_a_stable_pdf(tmp_path):
    drop_dir = str(tmp_path / "drop")
    seen = []

    watcher = VaultWatcher(drop_dir, on_pdf_ready=seen.append, poll_interval=0.05, stable_checks=2)
    watcher.start()
    try:
        pdf_path = os.path.join(drop_dir, "textbook.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 fake content" * 10)

        assert wait_for(lambda: len(seen) == 1, timeout=10.0)
        assert seen[0] == pdf_path
    finally:
        watcher.stop()


def test_watcher_ignores_non_pdf_files(tmp_path):
    drop_dir = str(tmp_path / "drop")
    seen = []

    watcher = VaultWatcher(drop_dir, on_pdf_ready=seen.append, poll_interval=0.05, stable_checks=2)
    watcher.start()
    try:
        with open(os.path.join(drop_dir, "notes.txt"), "w") as f:
            f.write("not a pdf")

        # Give it a couple of poll cycles to (not) fire.
        time.sleep(0.5)
        assert seen == []
    finally:
        watcher.stop()


def test_watcher_waits_for_slow_write_to_stabilize(tmp_path):
    drop_dir = str(tmp_path / "drop")
    seen = []

    watcher = VaultWatcher(drop_dir, on_pdf_ready=seen.append, poll_interval=0.1, stable_checks=2)
    watcher.start()
    try:
        pdf_path = os.path.join(drop_dir, "big.pdf")
        with open(pdf_path, "wb") as f:
            for _ in range(4):
                f.write(b"chunk" * 20)
                f.flush()
                time.sleep(0.15)  # each write is far enough apart to reset stability

        # Must not have fired while the file was still growing.
        assert seen == []

        assert wait_for(lambda: len(seen) == 1, timeout=10.0)
        assert seen[0] == pdf_path
    finally:
        watcher.stop()


def test_watcher_does_not_reprocess_same_file(tmp_path):
    drop_dir = str(tmp_path / "drop")
    seen = []

    watcher = VaultWatcher(drop_dir, on_pdf_ready=seen.append, poll_interval=0.05, stable_checks=2)
    watcher.start()
    try:
        pdf_path = os.path.join(drop_dir, "once.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        assert wait_for(lambda: len(seen) == 1, timeout=10.0)
        time.sleep(0.5)
        assert len(seen) == 1
    finally:
        watcher.stop()
