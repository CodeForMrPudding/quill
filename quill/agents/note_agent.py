import os
import re
import json
import hashlib
import threading
from collections import Counter
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage

from quill.helpers.pdf_ingestion import ingest_pdfs
from quill.helpers.note_planner import (
    group_chunks_by_chapter,
    batch_chunks_by_char_budget,
    chunk_ids_for_source,
    slugify,
    assign_page_names,
    build_index_page,
    has_notes,
    group_drafts_for_merge,
)
from quill.helpers.note_resolver import parse_json_object, strip_llm_wrapping
from quill.helpers.llm_config import EMBEDDING_MODEL_NAME, MAX_DRAFT_TOKENS, input_char_budget

# Per-LLM-call character budget for multi-page drafting. Keeps any single
# call well under the context window regardless of source document size --
# a 1000-page book just produces more, smaller batches.
DEFAULT_CHAR_BUDGET = min(6000, input_char_budget(MAX_DRAFT_TOKENS))

# Merging drafts: as many as fit the context window (each is at most
# MAX_DRAFT_TOKENS long), but no more than this per call -- the merged reply
# is itself capped, so folding too many drafts into one loses detail.
MAX_MERGE_GROUP = 5
MERGE_CHAR_BUDGET = input_char_budget(MAX_DRAFT_TOKENS)

# Chapter pages summarized together per call when building a document overview.
OVERVIEW_GROUP = 8

# A chat edit sends the whole note and gets the whole note back, so the
# note counts against the context window roughly twice (plus some room for
# source excerpts).
EDIT_TOTAL_CHAR_BUDGET = input_char_budget(0)
EDIT_MAX_NOTE_CHARS = int(EDIT_TOTAL_CHAR_BUDGET * 0.4)

# Bumped whenever what goes into the vector index changes, forcing a
# one-time rebuild of indexes written by older versions.
INDEX_SCHEMA_VERSION = 3
EMBED_BATCH_SIZE = 64
NOTE_PIECE_CHARS = 3000


def _where(**conditions) -> Optional[dict]:
    """Chroma metadata filter from keyword conditions (None values skipped);
    Chroma needs an explicit $and to combine more than one."""
    conds = [{k: v} for k, v in conditions.items() if v is not None]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def format_context(docs: List[tuple]) -> str:
    """Retrieved (text, metadata) pairs as prompt context, each labeled with
    where it came from so answers can cite it: [Book.pdf, p.12] for PDF
    passages, [note: path.md] for the user's own notes."""
    parts = []
    for text, meta in docs:
        if meta.get("kind") == "note":
            label = f"[note: {meta.get('path', '?')}]"
        else:
            page = meta.get("page")
            label = f"[{meta.get('source', '?')}" + (f", p.{int(page) + 1}]" if page is not None else "]")
        parts.append(f"{label}\n{text}")
    return "\n\n".join(parts)


def chunk_ids(corpus: List[dict]) -> List[str]:
    """
    Stable vector-index ids for corpus chunks: derived from each chunk's
    source and text, not its position, so re-ingesting (which renumbers
    every chunk's "idx") leaves unchanged chunks with unchanged ids and only
    genuinely new text gets embedded. A counter disambiguates identical text
    repeated within one document.
    """
    seen = Counter()
    ids = []
    for chunk in corpus:
        base = hashlib.sha1(f"{chunk.get('source', '')}\x00{chunk['text']}".encode("utf-8")).hexdigest()[:24]
        ids.append(f"{base}-{seen[base]}")
        seen[base] += 1
    return ids


class NoteAgent:
    def __init__(self, model, text_directory: str, pdf_source_dir: Optional[str] = None,
                 data_dir: Optional[str] = None):
        self.model = model
        self.directory = text_directory
        # PDFs are watched/ingested from a separate drop folder by default so
        # source PDFs and internal indexes don't clutter the Obsidian vault.
        self.pdf_source_dir = pdf_source_dir or text_directory
        # Corpus and search index live in the vault's hidden .quill/ folder.
        self.data_dir = data_dir or os.path.join(text_directory, ".quill")
        self.corpus_file = os.path.join(self.data_dir, "corpus.json")
        self.index_dir = os.path.join(self.data_dir, "chroma_index")
        self._setup_lock = threading.Lock()

        os.makedirs(text_directory, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.pdf_source_dir, exist_ok=True)

        # Always sync the corpus with the drop folder on startup. Ingestion
        # only parses new/changed PDFs, so this is near-free when nothing
        # changed -- and it means PDFs dropped while Quill wasn't running
        # get ingested (previously only a *missing* corpus.json triggered
        # ingestion, so they were silently absent from the corpus).
        ingest_pdfs(self.pdf_source_dir, self.corpus_file)

        self.vector_store = None
        self.setup_retrieval()

    def _drafting_model(self):
        """The chat model with replies capped at MAX_DRAFT_TOKENS, so merge
        prompts built from several drafts have a known maximum size. Same
        num_ctx as the shared model, so Ollama doesn't reload it."""
        copy = getattr(self.model, "model_copy", None)
        return copy(update={"num_predict": MAX_DRAFT_TOKENS}) if copy else self.model

    def setup_retrieval(self):
        """
        Brings the local Chroma vector index in line with corpus.json,
        embedding only chunks that aren't indexed yet and deleting ones whose
        source changed or was removed. The old approach -- delete the whole
        index and re-embed every chunk -- ran on every launch and after every
        PDF drop, so its cost grew with the size of the whole library.
        """
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings
            from langchain_core.documents import Document
        except ImportError:
            print("Warning: langchain-chroma not installed. Retrieval features disabled.")
            return

        with self._setup_lock:
            if not os.path.exists(self.corpus_file):
                with open(self.corpus_file, 'w') as f:
                    json.dump([], f)

            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
            persist_dir = self.index_dir
            info_path = os.path.join(persist_dir, "quill_index.json")
            wanted_info = {"embedding_model": EMBEDDING_MODEL_NAME, "schema": INDEX_SCHEMA_VERSION}

            def open_store():
                return Chroma(persist_directory=persist_dir, embedding_function=embeddings,
                              collection_name="quill_notes")

            try:
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    corpus = json.load(f)
                wanted = dict(zip(chunk_ids(corpus), corpus))

                store = self.vector_store or open_store()
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                except (OSError, json.JSONDecodeError):
                    info = {}
                if info != wanted_info:
                    # Different embedding model or index layout: vectors
                    # aren't comparable, so start over (one-time cost).
                    store.delete_collection()
                    store = open_store()

                # Only PDF chunks are synced here; notes are handled by
                # sync_note_index and must not be deleted as "stale".
                existing = set(store.get(where={"kind": "chunk"}, include=[])["ids"])
                stale = sorted(existing - wanted.keys())
                missing = [cid for cid in wanted if cid not in existing]

                if stale:
                    store.delete(ids=stale)
                if missing:
                    print(f"Indexing {len(missing)} new chunk(s) for retrieval "
                          f"({len(existing) - len(stale)} already indexed)...")
                for start in range(0, len(missing), EMBED_BATCH_SIZE):
                    batch_ids = missing[start:start + EMBED_BATCH_SIZE]
                    docs = []
                    for cid in batch_ids:
                        chunk = wanted[cid]
                        # "source" lets retrieval be scoped to one PDF (e.g.
                        # the document a note being edited came from).
                        # Chroma rejects None metadata values, hence the filter.
                        metadata = {k: chunk.get(k) for k in ("source", "header1", "page") if chunk.get(k) is not None}
                        metadata["kind"] = "chunk"
                        docs.append(Document(page_content=chunk["text"], metadata=metadata))
                    store.add_documents(docs, ids=batch_ids)
                    if len(missing) > EMBED_BATCH_SIZE:
                        print(f"  ...{min(start + EMBED_BATCH_SIZE, len(missing))}/{len(missing)}")

                os.makedirs(persist_dir, exist_ok=True)
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(wanted_info, f)
                self.vector_store = store
            except Exception as e:
                hint = f" (is the embedding model pulled? ollama pull {EMBEDDING_MODEL_NAME})" \
                    if "not found" in str(e).lower() else ""
                print(f"Vector store indexing warning: {e}{hint}")
                # Keep serving the previous index, if any, rather than
                # disabling retrieval because one update failed.

    def sync_note_index(self) -> int:
        """
        Makes the vault's own notes searchable alongside the PDF chunks, so
        chat answers can draw on (and cite) notes the user wrote or edited,
        not only the raw PDFs. Incremental: notes are re-read only when their
        modification time changes and re-embedded only when their text
        changes, so calling this before every chat question is cheap.
        Returns how many note pieces were embedded.
        """
        from langchain_core.documents import Document
        from quill.helpers.note_resolver import list_vault_notes
        from quill.helpers.pdf_ingestion import _split_oversized_text

        store = self.vector_store
        if store is None:
            return 0
        cache = self.__dict__.setdefault("_note_cache", {})  # rel -> (mtime_ns, [(id, text)])
        with self._setup_lock:
            try:
                wanted = {}
                for rel in list_vault_notes(self.directory):
                    mtime = os.stat(os.path.join(self.directory, *rel.split("/"))).st_mtime_ns
                    if rel not in cache or cache[rel][0] != mtime:
                        with open(os.path.join(self.directory, *rel.split("/")), encoding="utf-8") as f:
                            text = f.read()
                        seen = Counter()
                        pieces = []
                        for piece in _split_oversized_text(text.strip(), NOTE_PIECE_CHARS):
                            if not piece.strip():
                                continue
                            base = hashlib.sha1(f"note\x00{rel}\x00{piece}".encode("utf-8")).hexdigest()[:24]
                            pieces.append((f"n{base}-{seen[base]}", piece))
                            seen[base] += 1
                        cache[rel] = (mtime, pieces)
                    for pid, piece in cache[rel][1]:
                        wanted[pid] = (rel, piece)
                for rel in [r for r in cache if r not in {v[0] for v in wanted.values()}]:
                    del cache[rel]

                existing = set(store.get(where={"kind": "note"}, include=[])["ids"])
                stale = sorted(existing - wanted.keys())
                missing = [pid for pid in wanted if pid not in existing]
                if stale:
                    store.delete(ids=stale)
                for start in range(0, len(missing), EMBED_BATCH_SIZE):
                    batch = missing[start:start + EMBED_BATCH_SIZE]
                    docs = []
                    for pid in batch:
                        rel, piece = wanted[pid]
                        meta = {"kind": "note", "path": rel}
                        source = self.source_for_note(rel)
                        if source:
                            meta["source"] = source
                        docs.append(Document(page_content=piece, metadata=meta))
                    store.add_documents(docs, ids=batch)
                return len(missing)
            except Exception as e:
                print(f"Note indexing warning: {e}")
                return 0

    def search_docs(self, query: str, k: int = 4, source: Optional[str] = None,
                    kind: Optional[str] = None) -> List[tuple]:
        """The `k` indexed passages most relevant to `query` as (text,
        metadata) pairs, optionally limited to one source PDF and/or one kind
        ("chunk" = PDF passage, "note" = vault note). Never raises: retrieval
        is an enhancement, so a failure means 'no context', not a crash."""
        store = self.vector_store
        if store is None:
            return []
        try:
            where = _where(source=source, kind=kind)
            kwargs = {"filter": where} if where else {}
            return [(doc.page_content, doc.metadata) for doc in store.similarity_search(query, k=k, **kwargs)]
        except Exception:
            return []

    def search(self, query: str, k: int = 4, source: Optional[str] = None, kind: Optional[str] = None) -> List[str]:
        """Text-only form of search_docs."""
        return [text for text, _ in self.search_docs(query, k=k, source=source, kind=kind)]

    def detect_source(self, text: str) -> Optional[str]:
        """The ingested PDF a message names, if any -- by filename (with or
        without .pdf) or all of its significant words -- so a question like
        "what does Project Mary Hail say about X" searches only that book."""
        lowered = text.casefold()
        best = None
        for source in self.list_sources():
            stem = os.path.splitext(source)[0].casefold()
            words = [w for w in re.split(r"[^\w]+", stem) if len(w) > 3]
            if stem in lowered or (len(words) >= 2 and all(w in lowered for w in words)):
                if best is None or len(source) > len(best):
                    best = source
        return best

    def draft_note(self, topic: str) -> str:
        """
        Generates markdown note content for `topic` directly from the model
        with one plain (no tool-calling) LLM call, using retrieved context
        when available. The caller decides whether to persist the result
        (see PendingDraftStore / create_note) rather than this method writing
        anything to disk itself.

        Deliberately avoids LangGraph tool-calling: an earlier bind_tools-based
        version let a small local model re-issue the same retrieval tool call
        indefinitely (confirmed live -- it called a tool even when explicitly
        told not to), hanging for 10+ minutes on what should be a fast, simple
        draft. Doing retrieval as a direct Python call before the one LLM
        invocation sidesteps that failure mode entirely.
        """
        context = "\n\n".join(self.search(topic, k=5, kind="chunk"))[:input_char_budget(MAX_DRAFT_TOKENS)]

        if context:
            # `topic` is usually derived from a filename, which is often a poor
            # title (slug, version tag, ISBN...). When we have real retrieved
            # content, its actual subject matter should drive the title, not
            # the filename-derived string.
            prompt = (
                f"Source content:\n{context}\n\n"
                f"The source content above was extracted from a document originally named '{topic}' "
                "(the filename may not describe the content well, e.g. it could be a generic or technical "
                "slug -- do not use it as the title). "
                "Write a detailed Obsidian markdown note that accurately summarizes this source content. "
                "Choose a top-level '# ' heading that reflects the actual subject matter of the content "
                "above. Return ONLY the markdown content."
            )
        else:
            prompt = (
                f"Write a detailed Obsidian markdown note about: {topic}\n\n"
                "No indexed source context is available, so write from general knowledge. "
                "Return ONLY the markdown content, starting with a top-level '# ' heading."
            )
        response = self.model.invoke([HumanMessage(content=prompt)])
        return response.content

    def create_note(self, topic: str, vcs=None) -> str:
        """
        Drafts a note about `topic` (via draft_note) and writes it straight
        into the vault under a deterministic, collision-avoiding filename.
        Used by the interactive chat's NOTE intent, where the user expects
        the note to just appear -- unlike the watcher's flow, this skips the
        PendingDraftStore approval step. With `vcs` (a NoteVersionControl),
        the note is written through it so it starts life as version #1.
        Returns the path written.
        """
        content = self.draft_note(topic)

        base_name = slugify(topic)
        filename = f"{base_name}.md"
        path = os.path.join(self.directory, filename)
        n = 1
        while os.path.exists(path):
            n += 1
            filename = f"{base_name}_{n}.md"
            path = os.path.join(self.directory, filename)

        if vcs is not None:
            vcs.save(filename, content, f"Created from chat: {topic[:60]}")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        return path

    def choose_edit_target(self, request: str, candidates: List[str], recent_note: Optional[str] = None) -> dict:
        """
        One LLM call that picks which note an edit request is about (from
        `candidates`, see note_resolver.candidate_notes) and restates the
        requested change as a standalone instruction. Returns
        {"note": <name or None>, "instruction": <str>}; the caller validates
        the name against the vault.
        """
        note_lines = "\n".join(f"- {n}" for n in candidates)
        prompt = (
            "The user wants to change one of their existing Obsidian notes.\n\n"
            f"Note the conversation last worked on: {recent_note or 'none'}\n"
            f"Candidate notes:\n{note_lines}\n\n"
            f"User request: {request}\n\n"
            "Reply with ONE JSON object and nothing else: "
            '{"note": "<note name exactly as listed, or null if unclear>", '
            '"instruction": "<the requested change, restated on its own>"}. '
            "Words like 'it' or 'this note' refer to the note the conversation last worked on."
        )
        reply = self.model.invoke([HumanMessage(content=prompt)]).content
        try:
            plan = parse_json_object(reply)
        except ValueError:
            return {"note": None, "instruction": request}
        return {"note": plan.get("note"), "instruction": str(plan.get("instruction") or request)}

    def source_for_note(self, note_name: str) -> Optional[str]:
        """The ingested PDF a generated note came from (generated pages are
        prefixed with the source's slug), or None for hand-written notes."""
        basename = note_name.rsplit("/", 1)[-1]
        for source in self.list_sources():
            if has_notes(source, [basename]):
                return source
        return None

    def edit_note(self, note_name: str, current: str, instruction: str) -> str:
        """
        Returns the full revised text of a note with `instruction` applied.
        When the note was generated from an ingested PDF, a few relevant
        excerpts from *that* PDF are included, so requests like "add more
        detail on X" draw on the source rather than general knowledge.
        Writes nothing -- the caller shows a diff and saves via version
        control on approval.
        """
        if len(current) > EDIT_MAX_NOTE_CHARS:
            raise ValueError(
                f"'{note_name}' is {len(current):,} characters; chat edits are limited to "
                f"{EDIT_MAX_NOTE_CHARS:,} so the whole note fits in one model call. Edit it in Obsidian instead."
            )

        # Whatever the note (in and back out, plus ~10% growth) leaves of the
        # context window can carry source excerpts.
        excerpt_budget = EDIT_TOTAL_CHAR_BUDGET - int(2.1 * len(current)) - len(instruction)
        excerpts = ""
        source = self.source_for_note(note_name)
        if source and excerpt_budget > 500:
            excerpts = "\n\n".join(self.search(instruction, k=4, source=source, kind="chunk"))[:excerpt_budget]

        # The instructions come *after* the note on purpose: when a prompt
        # overflows the model's context window, Ollama drops the beginning,
        # and losing part of the note is far less harmful than losing the
        # instructions.
        sections = [f"Current note '{note_name}':\n<<<NOTE\n{current}\nNOTE>>>"]
        if excerpts:
            sections.append(f"Excerpts from the source document '{source}' (use only if the request needs them):\n{excerpts}")
        sections.append(
            "You are editing the Obsidian markdown note above. Apply this change:\n"
            f"{instruction}\n\n"
            "Return the COMPLETE revised note. Keep everything the change doesn't touch exactly as it is "
            "(wording, headings, [[wikilinks]], tags, frontmatter). Return only the note's markdown -- "
            "no commentary, no code fences, no <<<NOTE markers."
        )
        response = self.model.invoke([HumanMessage(content="\n\n".join(sections))])
        return strip_llm_wrapping(response.content)

    def list_sources(self) -> List[str]:
        """Source PDF filenames present in corpus.json, for resolving
        slash-command arguments that contain spaces."""
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        return sorted({c["source"] for c in corpus if c.get("source")})

    def source_status(self, source: str) -> str:
        """Why `source` has no chapters to draft, phrased for the user ('' if
        it does have content)."""
        from quill.helpers.pdf_ingestion import load_manifest

        entry = load_manifest(self.corpus_file).get(source)
        if entry is None:
            if os.path.exists(os.path.join(self.pdf_source_dir, source)):
                return f"'{source}' couldn't be parsed (see the error printed during ingestion)."
            return f"No ingested PDF is named '{source}' -- check the exact filename (the dashboard lists them)."
        if entry.get("scanned"):
            return (f"'{source}' looks scanned/image-only: it has no text layer, so it needs OCR first "
                    f"(for example: ocrmypdf input.pdf output.pdf), then drop the OCR'd copy in.")
        if entry.get("chunks") == 0:
            return f"No usable text was left in '{source}' after removing boilerplate."
        return ""

    def list_chapters(self, source_filename: str) -> List[dict]:
        """
        Cheap, LLM-free listing of the chapters available for `source_filename`
        (already ingested into corpus.json) -- so a user can see what a large
        document contains and pick which chapters are worth actually drafting,
        instead of paying for full generation on every chapter up front.
        Returns [{"title", "chunk_count", "char_count", "preview"}, ...].
        """
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        chunks = chunk_ids_for_source(corpus, source_filename)
        groups = group_chunks_by_chapter(chunks)

        checkpoints = []
        for title, chapter_chunks in groups.items():
            char_count = sum(len(c["text"]) for c in chapter_chunks)
            # Chunk text is prefixed with "Context: <headers>\n\n" -- skip that
            # for the preview so it shows actual content, not the header echo.
            raw_text = chapter_chunks[0]["text"]
            body = raw_text.split("\n\n", 1)[1] if "\n\n" in raw_text else raw_text
            preview = body[:180].strip()
            checkpoints.append({
                "title": title,
                "chunk_count": len(chapter_chunks),
                "char_count": char_count,
                "preview": preview,
            })
        return checkpoints

    def draft_multi_page_notes(
        self, source_filename: str, chapters: Optional[List[str]] = None, char_budget: int = DEFAULT_CHAR_BUDGET,
        on_page: Optional[Callable[[str, str], None]] = None, overview: bool = False,
    ) -> Dict[str, str]:
        """
        Produces one markdown page per chapter (grouped by Header 1) for all
        corpus chunks belonging to `source_filename`, instead of one flat
        note. Each chapter's chunks are processed in char-budgeted batches
        (map) and, when a chapter needed more than one batch, merged back
        into a single coherent page via hierarchical reduction (merge in
        small groups, repeat) -- so no LLM call in this whole pipeline ever
        receives more than a bounded slice of text, regardless of how large
        the source document is.

        `chapters`, if given, restricts drafting to just those chapter titles
        (as returned by list_chapters) -- e.g. for an on-demand /draft of a
        single chapter rather than the whole (expensive) document. When None,
        every chapter is drafted and an index page linking them all is added.

        Returns {filename: markdown_content}; the caller decides whether/how
        to persist each page (see PendingDraftStore). `on_page(name, content)`
        is called as each page finishes, so a caller that persists there
        keeps every completed chapter even if a later one fails or the run
        is interrupted -- a large book can take hours. With `overview=True`
        (and all chapters), the index page also gets a whole-document
        overview generated from the chapter pages.
        """
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        chunks = chunk_ids_for_source(corpus, source_filename)
        groups = group_chunks_by_chapter(chunks)
        # Named over the full chapter list, before filtering, so a single
        # chapter drafted on its own gets the same filename the index links to.
        page_names = assign_page_names(source_filename, list(groups.keys()))

        if chapters is not None:
            wanted = set(chapters)
            groups = {title: chunks for title, chunks in groups.items() if title in wanted}

        pages: Dict[str, str] = {}

        for chapter_title, chapter_chunks in groups.items():
            batches = batch_chunks_by_char_budget(chapter_chunks, char_budget)
            batch_drafts = [self._draft_batch(chapter_title, batch) for batch in batches]
            page_body = self._reduce_drafts(chapter_title, batch_drafts) if len(batch_drafts) > 1 else batch_drafts[0]

            name, content = page_names[chapter_title], f"# {chapter_title}\n\n{page_body}"
            pages[name] = content
            if on_page:
                on_page(name, content)

        if chapters is None:
            summary = self.draft_overview(source_filename, list(pages.values())) if overview and pages else ""
            index_name, index_content = build_index_page(source_filename, list(groups.keys()), summary)
            pages[index_name] = index_content
            if on_page:
                on_page(index_name, index_content)

        return pages

    def draft_overview(self, source_filename: str, chapter_pages: List[str]) -> str:
        """
        A whole-document overview built from the chapter pages (not the raw
        source -- far less text for a large book), using the same
        budget-bounded hierarchical merge as chapter drafting. Each page is
        trimmed so a group of several fits in one call.
        """
        per_page = max(1500, MERGE_CHAR_BUDGET // OVERVIEW_GROUP)
        current = [page[:per_page] for page in chapter_pages]
        model = self._drafting_model()
        first_round = True
        while first_round or len(current) > 1:
            next_round = []
            for group in group_drafts_for_merge(current, MERGE_CHAR_BUDGET, OVERVIEW_GROUP):
                what = "chapter notes" if first_round else "partial overviews"
                prompt = (
                    "\n\n---\n\n".join(group) + "\n\n---\n\n"
                    f"The text above is {len(group)} {what} (separated by ---) from the document "
                    f"'{source_filename}'. Write a concise overview of what they cover: a few short "
                    "paragraphs, then a '### Key themes' bullet list. No top-level heading. "
                    "Return ONLY the markdown."
                )
                next_round.append(model.invoke([HumanMessage(content=prompt)]).content.strip())
            current, first_round = next_round, False
        return current[0]

    # In both prompts below the instructions come *after* the content: if a
    # prompt ever overflows the context window, Ollama drops the beginning,
    # and losing some source text is far less harmful than losing the task.

    def _draft_batch(self, chapter_title: str, batch: List[dict]) -> str:
        batch_text = "\n\n".join(c["text"] for c in batch)
        prompt = (
            f"Source content:\n{batch_text}\n\n"
            f"The source content above is from the chapter/section '{chapter_title}'. "
            "Write a detailed Obsidian markdown section covering this content. Do not include a "
            "top-level '# ' heading (that's added separately) -- start with '## ' subheadings. "
            "Return ONLY the markdown content."
        )
        response = self._drafting_model().invoke([HumanMessage(content=prompt)])
        return response.content

    def _reduce_drafts(self, chapter_title: str, drafts: List[str]) -> str:
        current = drafts
        model = self._drafting_model()
        while len(current) > 1:
            next_round = []
            for group in group_drafts_for_merge(current, MERGE_CHAR_BUDGET, MAX_MERGE_GROUP):
                if len(group) == 1:
                    next_round.append(group[0])
                    continue
                combined = "\n\n---\n\n".join(group)
                prompt = (
                    f"{combined}\n\n---\n\n"
                    f"The text above is {len(group)} draft sections (separated by ---) covering different "
                    f"parts of the chapter '{chapter_title}'. Merge them into one coherent, well-organized "
                    "set of Obsidian markdown '## ' subsections, removing redundancy but preserving all "
                    "distinct information. Return ONLY the markdown content."
                )
                response = model.invoke([HumanMessage(content=prompt)])
                next_round.append(response.content)
            current = next_round
        return current[0]

