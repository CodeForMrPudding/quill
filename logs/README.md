# Quill engineering logs

A record of the October 2026 overhaul: what the audit found, what was changed and
why, what went wrong along the way, and how each fix was verified.

| File | Covers |
|---|---|
| [01-audit-findings.md](01-audit-findings.md) | The initial deep dive: 20 problems, ranked by severity |
| [02-wrong-document-and-overwrites.md](02-wrong-document-and-overwrites.md) | Round 1: notes drafted from the wrong PDF (#2), generated notes overwriting edits (#5) |
| [03-version-control-and-chat-editing.md](03-version-control-and-chat-editing.md) | Round 2: broken git agent (#1), chat editing (#3), automatic and vault-wide versioning (#9), stash data loss (#7) |
| [04-context-window-and-indexing.md](04-context-window-and-indexing.md) | Round 3: silent prompt truncation (#4), re-parsing and re-embedding the whole library (#14), PDFs missed at startup (#12) |
| [05-remaining-fixes.md](05-remaining-fixes.md) | Round 4: everything else: drafts lost on exit, watch resume, watcher design, ingestion filters, summary structure, chat citations, data layout, packaging |
| [06-open-issues.md](06-open-issues.md) | What is still not done, known limitations, and decisions left open |

Issue numbers (#1–#20) refer to the list in `01-audit-findings.md` throughout.

## Verification approach (all rounds)

- **Unit tests** with fake models, for fast, deterministic checks of logic and
  regressions. The suite grew from 93 to 207 tests and passes consistently (run 3 times
  in a row at the end).
- **Live checks** against the real local model (Ollama, `llama3.1` 8B, on an RTX 4070 with 12 GB),
  always on *scratch copies* of the vault and PDFs, never the real vault. These
  caught problems the fakes could not (see rounds 2 and 3).
