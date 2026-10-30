# Quill

**A local AI assistant that turns large PDFs into Obsidian notes, answers questions about
them, edits notes on request, and keeps git-style version history of everything.**

Everything runs on your machine with [Ollama](https://ollama.com): no cloud, and no
documents leave your computer.

- **Summarize massive PDFs:** books are split by chapter, summarized in bounded pieces
  and merged, so a 1,000-page book works the same way as a 10-page one. Each chapter
  becomes its own note, plus an index note with a whole-document overview.
- **Background watcher:** drop a PDF in a folder (or let it land in Downloads) and
  Quill asks whether to generate notes, then writes them into your vault chapter by
  chapter. Interrupted books resume where they stopped.
- **Ask and edit:** chat with your documents *and* your notes, with answers citing
  their sources (`[Book.pdf, p.12]`). Ask Quill to change a note ("in the Financials
  note, add a section on cash flow"); you see a diff and confirm before anything is saved.
- **Version history:** every note is versioned automatically as you edit it in
  Obsidian. Undo a change, restore one note to yesterday, or roll the whole vault
  back to a snapshot. Storage is delta-based, so history costs little disk space.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally, with two models:
  ```
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ```
- About 7 GB of free GPU memory for the default 16k-token context (or set a smaller
  `QUILL_NUM_CTX`, see [Configuration](#configuration)).
- Windows for pop-up notifications in watch mode. On other systems use
  `quill watch --auto-approve`.

## Install

```bash
git clone https://github.com/CodeForMrPudding/quill.git
cd quill
pip install -e .            # or: pip install -e ".[test]" to also run the tests
```

This installs the `quill` command (`python -m quill` also works).

## Quick start

```bash
cd quill                    # or anywhere; see "Where Quill keeps things"
quill                       # start a chat session (same as `quill chat`)
```

1. Put PDFs in `pdf_drop/`. They're ingested on startup, or as soon as they arrive
   while Quill runs.
2. In chat, list a PDF's chapters and draft notes:
   ```
   /chapters Meeting #2.pdf
   /draft-all Meeting #2.pdf
   /pending                       # review what was drafted
   /approve-all Meeting #2.pdf    # save the drafts into the vault (versioned)
   ```
3. Open `obsidian_vault/` in Obsidian. Each chapter is a note, and
   `<pdf>__index.md` links them all with an overview.

Or let the watcher do it hands-free:

```bash
quill watch                              # watches pdf_drop/
quill watch --watch-dir ~/Downloads      # also offers new PDFs that land in Downloads
```

## Using the chat

Just type. Quill works out what you want:

| You say | Quill does |
|---|---|
| "What does the meeting doc say about pricing?" | Answers from your PDFs and notes, citing sources |
| "Write a note about photosynthesis" | Creates a new note (versioned from the start) |
| "In the financials note, add a short section on cash flow" | Edits the note: shows a diff, asks, saves a new version |
| "Make it shorter" | Edits the note you last worked on |
| "Undo that" | Restores the previous version (asks first) |
| "What changed in the management note?" | Shows a diff since the last saved version |
| "Put the management note back to how it was yesterday" | Restores by time (asks first) |
| "Save a snapshot of my vault" / "Restore my vault to yesterday" | Vault-wide snapshot / restore |

### Slash commands

Type `/help` in chat for this list.

| Area | Commands |
|---|---|
| Sessions | `/new`, `/list`, `/rename <name>` |
| Note history | `/status [note]`, `/log <note>`, `/diff <note> [from_id] [to_id]`, `/commit <note> <message>`, `/checkout <note> <id\|latest>`, `/stash [note]`, `/stash-pop <note> [id]` |
| Vault snapshots | `/snapshot [message]`, `/snapshots`, `/restore-snapshot <id>` |
| Documents | `/focus <pdf>` (limit questions to one PDF; `/focus` alone clears) |
| PDF drafting | `/chapters <pdf>`, `/draft <pdf> <chapter>`, `/draft-all <pdf>`, `/pending`, `/approve <draft>`, `/approve-all <pdf>`, `/discard <draft>`, `/discard-all <pdf>` |
| Exit | `exit`, `quit`, `q` |

Names with spaces work without quotes (`/draft Meeting #2.pdf Financials`); quotes
also work.

Drafts survive restarts. Ctrl+C during `/draft-all` stops drafting but keeps every
finished page in `/pending`.

## Watch mode

`quill watch` runs in a terminal (Ctrl+C to stop):

- For each new PDF it shows a notification: *"New PDF: X (12 chapters). Generate notes?"*
  - **Yes:** chapters are drafted one at a time and saved into the vault as each
    finishes; the index gets an overview at the end.
  - **No:** not asked again unless the file changes.
  - **No answer:** asked again the next time watch starts.
- PDFs that arrived while nothing was running are offered at startup.
- An interrupted book (crash, Ctrl+C, model error) **resumes from the next chapter**
  on the next start, without asking again. Failed chapters are retried, and the final
  notification reports real counts.
- `--watch-dir DIR` also watches an inbox such as Downloads. Only PDFs you approve are
  copied into the library; PDFs already there at startup are ignored.
- Only one watcher runs per vault. If `quill watch` is running, chat sessions leave
  new PDFs to it.
- Regenerating notes for a PDF never loses your edits: the current version is
  saved in history first.

Scanned (image-only) PDFs have no text to read. Quill detects them and tells you to
OCR them first (for example `ocrmypdf in.pdf out.pdf`).

## Version history

Quill's version control is built for notes, separate from git:

- **Automatic:** a note is saved as a new version 30 seconds after you stop editing it
  in Obsidian. Edits made while Quill wasn't running are captured at the next start.
  Renames carry history along.
- **Per note:** history, diffs, restore any version, `latest`. Unsaved edits are
  stashed, never lost, when you restore.
- **Whole vault:** snapshots record every note's version at a moment; a daily
  snapshot is taken automatically. Restoring a snapshot first snapshots the current
  state (so a restore is undoable) and never deletes notes created later.
- **Efficient:** each version stores only a diff against its parent, with a full copy
  every 10 versions to keep restores fast.

## Configuration

Every setting can be passed as a flag or set as an environment variable.

| Flag | Environment variable | Default |
|---|---|---|
| `--repo` | `QUILL_HOME` | The current folder if it contains `obsidian_vault/` or `pdf_drop/`, else `~/Quill` |
| `--vault` | `QUILL_VAULT` | `<home>/obsidian_vault` (point it at your existing Obsidian vault) |
| `--drop-dir` | `QUILL_DROP_DIR` | `<home>/pdf_drop` (the PDF library) |
| — | `QUILL_MODEL` | `llama3.1` |
| — | `QUILL_EMBEDDING_MODEL` | `nomic-embed-text` |
| — | `QUILL_NUM_CTX` | `16384` tokens |

**Why `QUILL_NUM_CTX` matters:** Ollama does not use a model's full context window by
default, and silently drops the start of prompts that are too long. Quill always sets
the window explicitly and sizes every prompt to fit it. Lower it if you have less
GPU memory; raise it to edit larger notes in one go.

## Where Quill keeps things

```
<home>/
├── pdf_drop/                 your PDF library (ingested)
├── chat_history.db           chat sessions
└── obsidian_vault/           your notes (open this in Obsidian)
    └── .quill/               Quill's data (hidden from Obsidian)
        ├── versions.db       note history and vault snapshots
        ├── corpus.json       extracted PDF text, by chapter
        ├── corpus.manifest.json
        ├── chroma_index/     search index (PDF passages + your notes)
        ├── pending_drafts/   drafts awaiting approval
        └── watch_jobs.json   watch-mode progress per PDF
```

Older versions kept `corpus.json` and `chroma_index/` directly in the vault; they are
moved into `.quill/` automatically.

## How it works

1. **Ingest.** Each PDF is converted to Markdown (`pymupdf4llm`); running
   headers/footers, front-matter legal pages, references and indexes are removed; text
   is split by headings into chunks of ≤ 3,000 characters, tagged with source and page.
   Only new or changed PDFs are parsed.
2. **Index.** Chunks and your notes are embedded into a local Chroma index. Only new or
   changed text is embedded, so restarts are near-instant.
3. **Draft.** Per chapter, chunks are summarized in budgeted batches (map), then merged
   hierarchically (reduce), so no single model call exceeds the context window
   regardless of book size. The chapter notes are then condensed into a document
   overview.
4. **Chat.** Each message is routed (question / new note / edit / version request).
   Questions retrieve relevant passages and cite them; edits and version actions are
   planned by the model as structured actions, then validated and executed by Quill,
   with confirmation for anything that changes notes.

## Development

```bash
pip install -e ".[test]"
pytest                        # 207 tests; no Ollama needed (fake models)
```

```
quill/
├── cli.py                    commands, chat loop, slash commands
├── agents/
│   ├── note_agent.py         ingestion sync, search index, drafting, overview, editing
│   └── version_agent.py      natural-language → version-control actions
└── helpers/
    ├── pdf_ingestion.py      PDF → cleaned, chunked corpus (incremental)
    ├── note_planner.py       chapter grouping, page names, argument parsing
    ├── note_versioning.py    delta-based note history and vault snapshots
    ├── note_autosnapshot.py  automatic versioning of edits made in Obsidian
    ├── watch_flow.py         watch-mode logic (ask, draft, resume, inbox)
    ├── watch_jobs.py         per-PDF watch progress
    ├── vault_watcher.py      file-arrival detection
    ├── pending_drafts.py     drafts awaiting approval (persistent)
    ├── llm_config.py         model settings and prompt budgets
    ├── paths.py              where everything lives
    └── …                     dashboard, notifications, sessions, lock, resolver
```

`logs/` records the October 2026 overhaul: audit findings, each fix and why it was
made, problems hit along the way, and open issues.

## Troubleshooting

- **"Could not reach Ollama model"**: start Ollama and pull the models (see
  Requirements).
- **The first launch after upgrading is slow**: Quill re-parses and re-embeds your
  library once, after an index format change. Later launches are fast.
- **"looks scanned/image-only"**: the PDF has no text layer; OCR it first.
- **"`quill watch` is already running"**: only one watcher per vault; stop the other
  one, or just use it.
