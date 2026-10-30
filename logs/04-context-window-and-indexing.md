# 04: Round 3: silent truncation (#4), re-indexing cost (#14), missed PDFs (#12)

## #4: The context window

**Measured, not assumed:**

- `ollama ps` showed llama3.1 running with a **4,096-token** context (the model
  supports 128k).
- A 10,541-token test prompt was evaluated as **2,050 tokens**: about 80% of the
  prompt, including an instruction placed at the top, was dropped **with no
  error**.
- With `num_ctx=16384` the whole prompt was evaluated; the model used 7.0 GB and
  stayed 100% on the GPU (RTX 4070, 12 GB).
- Measured ~3.2 characters per token (number-heavy text) → used as a conservative
  budgeting ratio.

**Fix** (`quill/helpers/llm_config.py`):

- `NUM_CTX = 16384` (env `QUILL_NUM_CTX`), passed explicitly via
  `make_chat_model()`; `QUILL_MODEL` / `QUILL_EMBEDDING_MODEL` overrides.
- `input_char_budget(reserved_output_tokens)` sizes every prompt:
  - **Merging drafts:** groups packed by budget (≤ 5 per call), with a progress
    guarantee so a too-small window can't loop forever (`group_drafts_for_merge`).
  - **Chat:** history trimmed oldest-first (`trim_history`), so the retrieved
    context in the system message is no longer the first thing cut.
  - **Edits:** limit derived from the window (~20k characters, up from a fixed 8,000);
    source excerpts fill whatever room is left.
- Instructions are placed **after** content in drafting, merging and editing prompts,
  because Ollama truncates from the front.
- `check_ollama()` fails fast at startup with a clear message (constructing
  `ChatOllama` doesn't connect, so "Ollama is down" used to surface mid-chat).

**Problem faced:** ChatOllama's per-call `options=` *replaces* all options,
`num_ctx` included. To cap reply length (`num_predict`) for drafting calls only, a
`model_copy(update={"num_predict": ...})` is used instead. It has the same
`num_ctx`, so Ollama doesn't reload the model.

## #14 and #12: Re-parsing and re-embedding everything

**Before:** every launch deleted and re-embedded the entire vector index; every new
PDF re-parsed every PDF in the folder; and the watcher thread deleted the index
while chat might be querying it.

**Fix:**

- **Incremental ingestion** (`pdf_ingestion.ingest_pdfs`): a sidecar
  `corpus.manifest.json` records each PDF's size and mtime. Unchanged PDFs reuse their
  stored chunks, changed or new ones are parsed, removed ones drop out, and failed
  parses are retried next run. `pymupdf4llm` is imported only when something needs
  parsing. Writes are atomic (temp file + `os.replace`).
- **Startup always syncs** (now nearly free), which fixes #12: PDFs dropped while
  Quill was off are ingested on the next start.
- **Incremental vector index** (`NoteAgent.setup_retrieval`): stable ids (SHA-1 of
  source + text, plus a duplicate counter) mean only new chunks are embedded and stale
  ones deleted, in batches of 64 with progress output. A small marker file forces
  a one-time rebuild if the embedding model or index schema changes. The index is
  never deleted while in use.
- `NoteAgent.search()` is the single, never-raising retrieval entry point.

**Measured** (copies of 2 real PDFs, real embeddings):

| Run | Time |
|---|---|
| First launch: parse + embed 317 chunks | 275 s |
| Relaunch, nothing changed | **0.1 s** (previously re-embedded everything) |
| Relaunch after adding a small 1-page PDF | **3.0 s** (4 new chunks embedded) |

## Problems faced

- The heredoc escaping issue again, on a CLI edit containing `"\n\n".join`. The
  check failed before anything was written; applied with the editor tool instead.

## Verification

- 180 tests passing. New tests use real Chroma with a deterministic fake embedder
  to prove only new chunks are embedded, removed sources are deleted, scoped search
  works, and a model change triggers exactly one rebuild.
