import os
import re
from collections import OrderedDict
from typing import Iterable, List, Tuple


def slugify(text: str) -> str:
    # \w is Unicode-aware, so accented and non-Latin titles keep their
    # letters instead of all collapsing to "untitled". It still excludes
    # every character Windows forbids in filenames (<>:"/\|?*).
    cleaned = re.sub(r"[^\w\- ]", "", text or "")
    slug = re.sub(r"\s+", "_", cleaned.strip())
    return slug or "untitled"


def note_page_name(source_filename: str, chapter_title: str) -> str:
    return f"{slugify(source_filename)}__{slugify(chapter_title)}.md"


def index_page_name(source_filename: str) -> str:
    return f"{slugify(source_filename)}__index.md"


def assign_page_names(source_filename: str, chapter_titles: List[str]) -> "OrderedDict[str, str]":
    """
    Maps each chapter title to a page filename that's unique within the
    document. Distinct titles can slugify identically ("Q&A" / "QA"), or
    differ only by case, which is the same file on Windows/macOS -- without
    deduplication one chapter's page silently overwrites another's. Must be
    computed over the document's *full* chapter list (not a filtered subset)
    so a single-chapter /draft gets the same filename the index links to.
    """
    taken = {index_page_name(source_filename).casefold()}
    names: "OrderedDict[str, str]" = OrderedDict()
    for title in chapter_titles:
        base = note_page_name(source_filename, title)[:-3]
        candidate, n = f"{base}.md", 1
        while candidate.casefold() in taken:
            n += 1
            candidate = f"{base}_{n}.md"
        taken.add(candidate.casefold())
        names[title] = candidate
    return names


def build_index_page(source_filename: str, chapter_titles: List[str], overview: str = "") -> Tuple[str, str]:
    """
    Builds the (page_name, markdown_content) for a source document's index
    page, linking to each chapter's page (as named by assign_page_names) via
    Obsidian wikilinks. Pure/deterministic -- no LLM call needed, so it's
    cheap enough to write immediately, before any chapter has actually been
    drafted. `overview`, once the chapters exist (NoteAgent.draft_overview),
    adds a whole-document summary above the chapter list.
    """
    index_lines = [f"# {source_filename}", ""]
    if overview.strip():
        index_lines += ["## Overview", "", overview.strip(), ""]
    index_lines += ["## Chapters", ""]
    for title, page_name in assign_page_names(source_filename, chapter_titles).items():
        index_lines.append(f"- [[{page_name[:-3]}|{title}]]")
    return index_page_name(source_filename), "\n".join(index_lines)


# Section titles textbooks repeat in every chapter. Grouping by the bare
# title merged, say, every chapter's "Exercises" into one giant "chapter"
# spanning the whole book; these get qualified with their chapter instead.
GENERIC_SECTION_RE = re.compile(
    r"^(chapter\s+)?(summary|exercises?|problems?|review( questions)?|questions|"
    r"key (terms|concepts|points|takeaways)|conclusions?|introduction|overview|notes|practice( problems)?|"
    r"further reading|discussion( questions)?|self[- ]test|quiz|(learning )?objectives|case stud(y|ies))"
    r"[\s:.\d]*$",
    re.IGNORECASE,
)


def group_chunks_by_chapter(corpus: List[dict]) -> "OrderedDict[str, List[dict]]":
    """
    Groups corpus chunks by their Header 1 (chapter), preserving first-seen
    order. Falls back to the chunk's source filename, then "Untitled", if no
    header was captured during ingestion -- so a document with no detected
    headers still produces one usable group instead of being dropped.
    Generic per-chapter section titles (see GENERIC_SECTION_RE) are keyed as
    "<chapter> — <section>" so each chapter keeps its own.
    """
    groups: "OrderedDict[str, List[dict]]" = OrderedDict()
    last_specific = None
    for chunk in corpus:
        header = (chunk.get("header1") or "").strip()
        if header and GENERIC_SECTION_RE.match(header) and last_specific:
            key = f"{last_specific} — {header}"
        else:
            key = header or chunk.get("source") or "Untitled"
            if header:
                last_specific = header
        groups.setdefault(key, []).append(chunk)
    return groups


def batch_chunks_by_char_budget(chunks: List[dict], max_chars: int) -> List[List[dict]]:
    """
    Splits a chapter's chunks into batches whose combined text stays under
    `max_chars`, so a single LLM call never receives more than a bounded
    slice of text regardless of how large the source document is. A single
    chunk longer than `max_chars` becomes its own (oversized) batch rather
    than being dropped or split mid-sentence.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    batches: List[List[dict]] = []
    current: List[dict] = []
    current_len = 0

    for chunk in chunks:
        chunk_len = len(chunk["text"])
        if current and current_len + chunk_len > max_chars:
            batches.append(current)
            current = []
            current_len = 0
        current.append(chunk)
        current_len += chunk_len

    if current:
        batches.append(current)
    return batches


def group_drafts_for_merge(drafts: List[str], char_budget: int, max_group: int) -> List[List[str]]:
    """
    Splits drafts into consecutive groups for one merge round: each group
    fits within `char_budget` and holds at most `max_group` drafts. Always
    makes progress -- if no two neighbors fit together (a context window
    configured too small for the draft size), drafts are paired anyway, so
    hierarchical merging still terminates instead of looping forever.
    """
    groups: List[List[str]] = []
    current: List[str] = []
    current_len = 0
    for draft in drafts:
        if current and (len(current) >= max_group or current_len + len(draft) > char_budget):
            groups.append(current)
            current, current_len = [], 0
        current.append(draft)
        current_len += len(draft)
    if current:
        groups.append(current)

    if len(drafts) > 1 and len(groups) == len(drafts):
        print(f"[Quill] Warning: drafts are too large to merge within the context window "
              f"({char_budget:,} chars); merging in pairs anyway. Consider raising QUILL_NUM_CTX.")
        groups = [drafts[i:i + 2] for i in range(0, len(drafts), 2)]
    return groups


def chunk_ids_for_source(corpus: List[dict], source: str) -> List[dict]:
    """
    Chunks belonging to `source`. Falls back to the whole corpus only for
    older corpora where *no* chunk carries a "source" field. A tagged corpus
    with no match returns [] -- falling back there would hand every other
    document's chunks to a typo'd name, an image-only PDF with no extractable
    text, or a PDF that failed to parse, and draft notes from all of them.
    """
    if not any("source" in c for c in corpus):
        return corpus
    return [c for c in corpus if c.get("source") == source]


def split_leading_arg(arg_text: str, suffix: str = "", known: Iterable[str] = ()) -> Tuple[str, str]:
    """
    Splits a slash command's argument text into (first_arg, rest), allowing
    the first argument to contain spaces -- real filenames like
    "Meeting #2.pdf" or Obsidian notes like "My Note.md" otherwise get cut
    at the first space. Tried in order:

    1. A quoted first argument: "My Note.md" rest...
    2. The longest entry of `known` that the text starts with (case-
       insensitive), returned in its canonical spelling.
    3. Everything up to the first occurrence of `suffix` (e.g. ".pdf")
       that ends a word.
    4. The first whitespace-delimited token.
    """
    text = arg_text.strip()
    if not text:
        return "", ""

    if text[0] in "\"'":
        end = text.find(text[0], 1)
        if end != -1:
            return text[1:end], text[end + 1:].strip()

    lowered = text.casefold()
    for name in sorted(known, key=len, reverse=True):
        n = len(name)
        if lowered.startswith(name.casefold()) and (len(text) == n or text[n].isspace()):
            return name, text[n:].strip()

    if suffix:
        match = re.search(re.escape(suffix) + r"(?=\s|$)", text, re.IGNORECASE)
        if match:
            return text[:match.end()], text[match.end():].strip()

    first, _, rest = text.partition(" ")
    return first, rest.strip()


def has_notes(source_filename: str, existing_notes: Iterable[str]) -> bool:
    """True if any note filename in `existing_notes` was generated from
    `source_filename` (generated pages are always prefixed with its slug)."""
    prefix = f"{slugify(source_filename)}__"
    return any(note.startswith(prefix) for note in existing_notes)


def find_unprocessed_pdfs(drop_dir: str, vault_dir: str) -> List[str]:
    """
    Lists .pdf filenames sitting in `drop_dir` that don't yet have any notes
    in `vault_dir` -- used on watcher startup to catch PDFs that arrived
    while nothing was watching (the watcher only fires on new-file
    filesystem events, not on a directory's existing contents at launch).

    Deliberately checked against generated note filenames (always prefixed
    with the source's slug, per note_page_name), not against corpus.json
    membership: NoteAgent auto-ingests everything in the drop folder into
    the corpus on startup regardless of whether the user has actually been
    asked, so corpus membership can't distinguish "ingested" from "the user
    was actually offered a chance to approve it" -- only real note files can.
    """
    if not os.path.isdir(drop_dir):
        return []
    existing_notes = set(os.listdir(vault_dir)) if os.path.isdir(vault_dir) else set()
    unprocessed = []
    for filename in sorted(os.listdir(drop_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        if not has_notes(filename, existing_notes):
            unprocessed.append(filename)
    return unprocessed
