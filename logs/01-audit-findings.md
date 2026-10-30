# 01: Audit findings

Initial deep dive of the codebase (all source read; test suite run: 93 passing, all
helper-level with fake models; `cli.py` and the git agent had no tests at all).

Severity: **Critical** = a core feature broken or wrong output; **Data loss** = user
work can disappear; **Gap** = a stated requirement not met; **Scale** = breaks down
on large PDFs; **Hygiene** = packaging and maintainability.

| # | Severity | Problem | Status |
|---|---|---|---|
| 1 | Critical | Chat "GIT" intent never ran anything: the LangGraph agent used `interrupt_before=["tools"]` and was never resumed, so replies came back empty. It also ran raw `git` on the code repo, where the vault is gitignored | Fixed (round 2) |
| 2 | Critical | Filenames with spaces broke every slash command, and `chunk_ids_for_source` fell back to the *whole corpus* on no match, so `/chapters Meeting #2.pdf` listed chapters of all PDFs | Fixed (round 1) |
| 3 | Gap | No way to edit notes through the LLM; the NOTE intent only created new notes and never versioned them | Fixed (round 2) |
| 4 | Critical | `num_ctx` never set: Ollama ran at 4,096 tokens and silently dropped the start of long prompts (instructions first) | Fixed (round 3) |
| 5 | Data loss | Generated notes overwrote user edits without snapshotting; slug collisions (`Q&A`/`QA`, non-Latin titles) made chapters overwrite each other | Fixed (round 1) |
| 6 | Data loss | Pending drafts lived only in memory, `/draft-all` returned only at the end (one failure lost everything), and Ctrl+C quit the whole REPL | Fixed (round 4) |
| 7 | Data loss | `stash_pop` overwrote uncommitted edits | Fixed (round 2) |
| 8 | Data loss | An interrupted watch job never resumed (the index page written first made it look "done"); failures were skipped silently and the final toast still said "Finished" | Fixed (round 4) |
| 9 | Gap | Version control was manual and per-file: no auto-snapshots, no vault-wide snapshot or restore, no diff, renames lost history, log mixed abandoned versions | Fixed (round 2) |
| 10 | Gap | Watched `<repo>/pdf_drop` only (not Downloads), hardcoded; not a real background service; `quill` and `quill watch` together double-processed PDFs | Mostly fixed (round 4); autostart left open, see 06 |
| 11 | Data loss | `wait_until_stable` gave up after 60 s, so big slow copies were dropped forever | Fixed (round 4) |
| 12 | Critical | PDFs dropped while Quill was off were never ingested (ingestion only ran when `corpus.json` was missing) | Fixed (round 3) |
| 13 | Gap | One worker thread blocked up to 300 s per toast and through all drafting; an ignored toast counted as "No"; Windows-only | Fixed (round 4) |
| 14 | Scale | Every launch re-embedded the entire corpus; every new PDF re-parsed every PDF; the watcher deleted the index while chat queried it; `corpus.json` written non-atomically | Fixed (round 3) |
| 15 | Gap | Summaries per chapter only (no document overview); repeated headings like "Exercises" merged into one giant chapter | Fixed (round 4) |
| 16 | Critical | Ingestion filters removed real content: repeated-line stripping anywhere on the page, any page mentioning an ISBN dropped, Appendix sections dropped; scanned PDFs silently produced nothing | Fixed (round 4); OCR itself not added, see 06 |
| 17 | Gap | Chat retrieval: an extra LLM call per message, a fragile router (`"EXIT" in reply`), no source scoping, no citations, notes not searchable | Fixed (rounds 2 and 4) |
| 18 | Hygiene | `ChatOllama(...)` construction can't detect Ollama being down; embedding failure silently disabled retrieval | Fixed (rounds 3 and 4) |
| 19 | Hygiene | Data location depended on the current directory; model names hardcoded | Fixed (rounds 3 and 4) |
| 20 | Hygiene | Installed top-level packages named `src` and `cli`; unpinned deps; `.pyc` and `.idea` committed; corpus and index cluttering the vault | Fixed (round 4) |

## Verified at audit time

- `/chapters Meeting #2.pdf` parsed the source as `"Meeting"` and returned all 321
  chunks from 3 documents (whole-corpus fallback).
- `note_page_name("a.pdf","Q&A") == note_page_name("a.pdf","QA")`, and any non-Latin
  title slugified to `untitled`.
- `find_unprocessed_pdfs` showed a PDF as "unprocessed" whose notes would then have
  been drafted from the wrong chunks.
