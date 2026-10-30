# 06: Open issues, limitations and decisions left open

## Decisions for the owner

- **Start-on-login for `quill watch`.** Not implemented. Today `quill watch` must be
  started in a terminal. Options to discuss: a Windows Task Scheduler entry, or a
  documented manual setup. This should be the user's explicit choice rather than
  something Quill installs on its own.
- **Push target.** This work was pushed to a feature branch, not `main`, so it can be
  reviewed and merged deliberately.

## Known limitations

- **No built-in OCR.** Scanned PDFs are detected and explained, but must be OCR'd
  externally (e.g. `ocrmypdf in.pdf out.pdf`) before Quill can use them.
- **Cross-process search-index writes.** `corpus.json` writes are atomic and only one
  process watches PDFs, but the chat REPL syncing note embeddings while `quill watch`
  adds PDF chunks can still write to Chroma from two processes at once. This is
  rare and not observed in testing, but not formally guarded.
- **Cross-document filename collisions.** Two PDFs whose names slugify alike (e.g.
  `book.pdf` and `book!.pdf`) still write to the same note names. Overwrites are
  snapshotted, so they're recoverable, but they collide.
- **Model reliability.** All LLM-driven decisions (intent routing, version actions,
  edit target) are validated in code, and unclear output becomes "I couldn't
  understand", never an unchecked action. An 8B local model can still misread unusual
  phrasings.
- **Time phrases.** "this morning" resolves to midnight; there's no fine-grained
  time parsing beyond what the model returns.
- **`detect_source`** needs the filename or at least two of its significant words in
  the question ("the meeting doc" doesn't scope to `Meeting #2.pdf`); use `/focus`.
- **Chat edit size.** Notes over ~20k characters (at the default 16k context) can't be
  edited from chat in one call; there's no section-level editing yet.
- **`PendingDraftStore.approve()`** still writes directly (no snapshot); the CLI
  doesn't use it, only tests do.
- **`find_unprocessed_pdfs`** is no longer used by the CLI (replaced by the
  watch job store). It's kept, with its tests, as a small pure helper; it can be removed.

## One-time costs on the next real launch

- The vault's old `corpus.json` and `chroma_index/` are moved into
  `obsidian_vault/.quill/`.
- There's no manifest yet and the index schema changed, so all PDFs are re-parsed and
  re-embedded once (~4–5 minutes for the current library). After that, launches
  take a fraction of a second.
- Every untracked note gets its first version, and a daily vault snapshot is taken.

## Not yet tested live

- Real Windows toast interaction with the new tri-state answers. The unit tests
  cover the flow with a fake notifier, and the toast code change was minimal: the
  default answer changed from `False` to `None`.
- `--watch-dir` against a real browser download (logic unit-tested with file copies).
