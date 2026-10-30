# 05: Round 4: the remaining fixes

## Packaging and data layout (#19, #20)

- **Package rename:** `src/` → `quill/`, `cli.py` → `quill/cli.py`. The project used
  to install top-level packages literally named `src` and `cli` into site-packages,
  which collide with any other project doing the same. The entry point is now
  `quill = "quill.cli:cli"`, and `python -m quill` also works.
- `pyproject.toml`: lower bounds = tested versions, a `test` extra, pytest config,
  version 0.2.0. `requirements.txt` now pins the exact tested versions instead of
  duplicating the unpinned list.
- **Paths** (`quill/helpers/paths.py`): `--repo/--vault/--drop-dir` flags, then
  `QUILL_HOME/QUILL_VAULT/QUILL_DROP_DIR`, then defaults. The home folder is the
  current directory only if it already has a Quill layout; otherwise `~/Quill`, so
  running `quill` from a random folder no longer creates an empty vault there.
- **Internal data moved into `<vault>/.quill/`** (corpus, manifest, search index,
  pending drafts, watch jobs, lock), a hidden folder Obsidian ignores.
  `migrate_legacy_layout` moves old files once, never overwriting. The dashboard
  reads both layouts.
- `.gitignore`: personal data (vault, PDFs, databases, `.quill/`, `*.pdf`), build
  artifacts and IDE folders; `.idea/` and `.pyc` files removed from tracking (kept on
  disk).

## Ingestion quality (#16)

- The running header/footer stripper now removes matches **only in each page's
  edge lines**, and never counts or strips pure markdown structure (table
  separators, rules). Previously body lines that merely looked like a footer after
  digit-collapsing ("Step 1"/"Step 2") were deleted, and table separators vanished.
- The legal-page filter is **position-aware**: pages are dropped only in the
  front/back matter, or when a short mid-book page has 2 or more distinct legal phrases. A
  bibliography page with ISBNs or a law chapter about copyright is kept.
- **Appendix sections are kept** (they often hold real material). The existing
  test asserting the opposite was updated deliberately.
- **Scanned / image-only PDFs** are detected (fewer than 50 extracted characters per page) and
  recorded in the manifest with page and chunk counts. The CLI and watcher now say
  "looks scanned, needs OCR (e.g. ocrmypdf)" instead of failing silently.

**Bug found while testing:** an image-only PDF still produced a chunk, because the
minimum-length check ran *before* the internal `@@QUILLPAGE:N@@` markers were
stripped; a run of markers alone passed the 40-character minimum and became an empty
"chapter". Length is now checked on marker-free text, per section and per piece.

## Summary structure (#15)

- Generic per-chapter section titles ("Exercises", "Summary", "Review Questions",
  "Key Terms", …) are grouped as `<chapter> — <section>` instead of merging across
  the whole book.
- **Document overview:** `NoteAgent.draft_overview` builds a whole-document summary
  from the chapter pages using the same budgeted hierarchical merge. It's added to the
  index page by `/draft-all` and by the watcher once every chapter succeeds.

## Drafts no longer lost (#6)

- `PendingDraftStore(store_dir=...)` persists each draft as it's added and reloads
  on startup; the REPL tells you how many are waiting.
- `draft_multi_page_notes(on_page=...)` hands back each page as it finishes;
  `/draft-all` stores pages one at a time, so a failure or Ctrl+C keeps everything
  done so far.
- Ctrl+C during `/draft` or `/draft-all` cancels the drafting only, not the whole REPL.

## Watch mode (#8, #10, #11, #13)

The watch logic moved out of the CLI into `quill/helpers/watch_flow.py`
(`PdfWatchFlow`) so it can be tested.

- **#8 Resume:** `WatchJobStore` (`.quill/watch_jobs.json`) records each PDF's status
  (`in_progress` with finished chapters, `done`, `declined`) plus its file signature.
  An interrupted book resumes from the next chapter without asking again. Failed
  chapters are retried next start, and the notification reports real counts
  ("Generated 2 of 3; 1 failed").
- **#13 No blocking:** each question is asked on its own short-lived thread;
  drafting runs on one worker (one GPU). `ask_yes_no` is now tri-state, so **an
  ignored toast is no longer a "No"**: it's asked again next start. A
  `ConsoleNotifier` plus `--auto-approve` makes watch mode usable without Windows toasts.
- **#11 Large files:** no fixed time limit while a file keeps growing. It gives up only
  if the file stays empty for 10 minutes or disappears, and stopping doesn't wait on a
  slow copy. Modified-file events are handled too (unchanged PDFs are a no-op).
- **#10 Downloads and single instance:** `quill watch --watch-dir ~/Downloads`
  offers *new* PDFs from an inbox folder and copies only approved ones into the
  library. This avoids pointing the library at Downloads, which would ingest every
  unrelated PDF there. An identical re-import is reused; a different file with the same
  name gets a numbered name rather than overwriting. An `InstanceLock` ensures one
  watcher per vault, and the REPL skips its own watcher when `quill watch` is
  running.

## Chat retrieval (#17, #18)

- **Your notes are searchable:** `sync_note_index` embeds vault notes alongside PDF
  chunks. It re-reads only notes whose mtime changed and re-embeds only changed
  text, so it's cheap to run before every question. PDF sync no longer deletes note
  entries (index schema v3 tags entries with `kind`).
- **Citations:** retrieved passages are labeled `[Book.pdf, p.12]` or
  `[note: Topic.md]`, and the model is asked to cite them.
- **Scoping:** `detect_source` notices when a question names a PDF; `/focus <pdf>`
  pins the scope and `/focus` alone clears it.
- Indexing errors now hint `ollama pull nomic-embed-text` when the model is missing.

## Problems faced

- The heredoc escaping issue recurred on an ingestion edit (regex backslashes). The
  check failed before writing, and it was redone with the editor tool. From here on,
  no heredoc edits containing backslashes.
- One working session was interrupted partway through. On resuming, the state was
  re-checked (the pending-drafts change had fully applied, and the tests passed)
  before continuing.

## Verification

- **207 tests**, passing 3 runs in a row. New: `test_watch_flow.py` (approve → done
  with overview, resume without re-asking, failed chapter retried, decline
  remembered, unanswered not recorded as declined, scanned PDF explained, inbox
  import never overwrites, lock exclusivity, stable-wait behavior),
  `test_layout_and_retrieval.py` (paths, migration, persistent drafts, generic
  sections, overview, streaming pages, note indexing, citations, scoping),
  `test_ingestion_filters.py`.
- **Live** (scratch copy, real llama3.1): the watch flow drafted all 6 chapters of
  `Meeting #2.pdf` plus a coherent overview in 45 s and marked the job done. A chat
  question was answered from the right note, with a citation.
- `quill --version` → 0.2.0 after reinstalling the editable package; `quill
  dashboard` still reads the real vault's old layout.
