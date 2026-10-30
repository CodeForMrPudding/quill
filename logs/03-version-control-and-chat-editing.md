# 03: Round 2: version control (#1, #7, #9) and chat editing (#3)

## Bugs and gaps

- **#1 Git agent.** `GitAgent` compiled its LangGraph with
  `interrupt_before=["tools"]`, but the CLI never resumed the graph, so chat
  "save/commit/revert" requests returned an empty reply and left a dangling tool call.
  It also targeted *real git* on the code repo (the vault is gitignored) with
  arbitrary model-chosen commands.
- **#3** No way to edit an existing note from chat.
- **#7** `stash_pop` overwrote uncommitted working-copy edits.
- **#9** Versioning was manual, per-file only: nothing captured edits made in
  Obsidian, there was no vault-wide snapshot or restore, no diff, renames lost history, and
  the log listed abandoned versions with no HEAD marker. The full-copy cadence used
  a per-note commit *count*, so commits on checked-out old versions could grow
  replay chains without bound.

## Solutions

- **Removed `GitAgent`** (and the unused `src/states`, plus the langgraph
  dependency). Added **`VersionAgent`** (`quill/agents/version_agent.py`): *one* plain
  LLM call returns a JSON action (`status`, `save`, `history`, `diff`,
  `restore_note`, `list_snapshots`, `restore_vault`), validated and executed in
  Python. No tool-calling loop (a small local model was previously seen looping on
  tool calls), no raw commands, and every restore asks for confirmation (default No).
- **Chat editing** (EDIT intent): candidate notes are ranked by fuzzy name match
  plus the note last worked on; the LLM picks one and restates the instruction; the
  note is rewritten (with source-PDF excerpts for generated notes); a coloured diff
  is shown; after confirmation it's saved via `save()`, so "undo that" works.
- **`NoteVersionControl`** moved to `quill/helpers/note_versioning.py` and extended:
  - vault snapshots (`snapshot`, `list_snapshots`, `restore_snapshot`): a restore
    first takes a safety snapshot (so it's undoable) and never deletes newer notes
  - `diff`, `ancestors` (true history to HEAD), `latest`, `show`, `rename`,
    `commit_all`, `ensure_recent_snapshot`
  - full copies are now triggered by **parent-chain depth**, not commit count
  - `stash_pop` swaps: current unsaved edits are stashed first
  - restored files are written with LF (Obsidian's convention)
  - `BEGIN IMMEDIATE` write transactions plus a thread lock, so the REPL and
    `quill watch` can both auto-commit without forking history
- **`NoteAutoSnapshotter`**: watches the vault; commits a note after 30 s of quiet
  (one editing burst = one version), catches up on edits made while Quill was off,
  follows renames, takes a daily vault snapshot, and flushes pending edits on exit.
  It ignores `.obsidian/`, `.trash/`, `.quill/`, `chroma_index/`.
- New commands: `/snapshot`, `/snapshots`, `/restore-snapshot`, `/diff`,
  `/checkout <note> latest`, `/help`; `/log` marks the current version.

## Problems faced

1. **Intent parsing.** A test showed the new "first category word wins" parser still
   read `"Not EXIT -- this is CHAT"` as EXIT. The fix: if the reply *starts* with a
   category take it, otherwise take the *last* category mentioned. A guessed EXIT now
   also asks before ending the session.
2. **Live llama3.1 quirks** (the fake-model tests could not catch these):
   - "undo that" came back with `steps_back: 1` **and** `before: <now>`; since
     `before` took precedence, the "undo" would have restored the current version
     (a no-op).
   - "revert … to version 3" came back with `commit_id: null, steps_back: 1`.

   Fixed with `_pin_restore_target`: explicit "version N"/"snapshot N" numbers are
   read from the user's own words, a time-based target applies only if the message
   mentions a time, and few-shot examples were added to the prompt. A first version
   of the regex had false positives (`am` in "I am", `may` the verb, and a bare `#2`
   matching the filename "Meeting #2"), all tightened and covered by tests.
3. **Timezones.** Models return `...Z` timestamps; comparing aware with naive
   datetimes raises `TypeError`. `parse_time` normalizes to naive local time.
4. **Shell heredoc escaping.** Editing test files through a bash heredoc turned
   `"\n"` escapes into real newlines, producing a syntax error. Fixed with the
   editor tool; from then on, heredocs were not used for content containing
   backslashes.

## Verification

- 162 tests passing.
- Live (scratch copy of the vault): the intent router classified 9/9 sample messages
  correctly, including "I'm done with the financials, what's next?" → CHAT. After the
  fix, all restore plans were correct. A chat edit added a section, kept all original text,
  and "undo that" restored the note byte-for-byte.
