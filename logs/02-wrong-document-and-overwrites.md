# 02: Round 1: wrong-document drafting (#2) and overwritten edits (#5)

## Bugs

1. **Whole-corpus fallback.** `chunk_ids_for_source` returned *every* chunk when no
   chunk matched the requested source. Any typo, image-only PDF, `.PDF` file or PDF
   that failed to parse made Quill list (and draft) chapters from all other
   documents under that name. In `quill watch` the toast then offered to draft the
   whole library.
2. **Whitespace argument parsing.** Slash commands split on spaces, so real
   filenames like `Meeting #2.pdf` and `Project Mary Hail.pdf` were cut at the first
   space, feeding bug 1.
3. **Case-sensitive extension.** The watcher accepted `.PDF`, but ingestion only read
   `.pdf`.
4. **Silent overwrites.** The watch flow and `/approve` wrote notes with
   `open(..., "w")` and committed *after* writing, so a user's uncommitted edits were
   never snapshotted. Re-downloading a PDF regenerated and overwrote its notes.
5. **Filename collisions.** `slugify` stripped non-ASCII and punctuation, so `Q&A`
   and `QA` (and every non-Latin title, which became `untitled`) mapped to the same
   page; one chapter silently replaced another.

## Solutions

| Bug | Fix | Where |
|---|---|---|
| 1 | No fallback for tagged corpora (only legacy corpora with no `source` field at all keep it) | `note_planner.chunk_ids_for_source` |
| 2 | `split_leading_arg`: quoted args, known-name prefix matching (case-insensitive), suffix (`.pdf`/`.md`) detection, then first token | `note_planner.py`, all slash commands |
| 3 | `filename.lower().endswith(".pdf")` | `pdf_ingestion.py` |
| 4 | `NoteVersionControl.save()`: commits uncommitted or untracked existing content first ("Snapshot of existing edits before: …"), then writes and commits. Every generated-note write goes through it | `note_versioning.py`, CLI, watch flow |
| 4 | Toast says "already has notes; regenerate? (current versions kept in history)" on a re-drop | watch flow |
| 5 | Unicode-aware `slugify` (`\w`; still removes Windows-forbidden characters); `assign_page_names` dedupes case-insensitively, computed over the *full* chapter list so `/draft` of one chapter gets the same name the index links to | `note_planner.py` |

## Bug found while testing

**One-page PDFs were stripped to nothing.** `_strip_repeated_lines` counted
`lines[:3] + lines[-3:]`; on pages with 6 or fewer lines those slices overlap, so each line
counted twice and hit the minimum threshold of 2 on its own. All content of one-page
PDFs (and short pages in 2–4 page PDFs) was removed as "boilerplate". Fixed by
counting each normalized line once per page (a set). Regression test added.

## Problems faced

- `cli.py` uses CRLF line endings; a line-index splice split on `\n` and its sanity
  assertion pointed at the wrong line. Nothing was written (the assertion failed
  first); redone with a CRLF-aware splice.

## Verification

- 112 tests passing (19 new).
- End to end on a copy of the real `corpus.json`, with a fake LLM and a fake toast:
  `Meeting #2.pdf` resolved correctly with and without quotes, and `Meeting` found nothing.
  A note edited between two watch runs appeared in history as a snapshot, and
  checking it out restored the edit.
