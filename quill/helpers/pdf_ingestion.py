import os
import re
import json
import time
from collections import Counter

# Section headers that are boilerplate, not content — dropped from the corpus
# regardless of how much text they contain.
# Appendices are deliberately NOT in this list: they routinely hold real
# material (proofs, data tables, worked solutions) that notes should cover.
BOILERPLATE_HEADER_PATTERN = re.compile(
    r"^(references?|bibliography|works cited|index|(table of )?contents|"
    r"acknowledge?ments?|about the author|dedication|glossary)$",
    re.IGNORECASE,
)

# Copyright/legal front matter (found via live testing: "This is a work of
# fiction...Copyright (c) 2021...All rights reserved..."). Checked per PAGE,
# not per header-split section -- these notices are almost always confined
# to their own dedicated page(s), and a document with no real headers (see
# the page-block fallback below) could otherwise have its entire front
# matter folded into one giant section that we must not drop wholesale just
# because part of it was a copyright notice.
LEGAL_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"all rights reserved",
        r"is a work of fiction",
        r"no part of this (book|publication) may be reproduced",
        r"library of congress",
        r"isbn[\s:]*[\d-]",
        r"copyright\s*(©|\(c\))",
    )
]

# A legal-looking phrase only gets a page dropped when the page is in the
# front/back matter (where copyright pages actually live), or when several
# distinct legal phrases appear on one short page. Previously any page
# mentioning an ISBN or "copyright (c)" was dropped wherever it was -- a
# mid-book bibliography page, or a law textbook's chapter on copyright.
FRONT_MATTER_MIN_PAGES = 4
FRONT_MATTER_RATIO = 0.05
BACK_MATTER_PAGES = 2
LEGAL_SHORT_PAGE_CHARS = 2500

# Average extracted characters per page below which a PDF is treated as
# scanned/image-only (no text layer -- it would need OCR).
SCANNED_CHARS_PER_PAGE = 50

# Chunks shorter than this are almost always stray fragments (a lone caption,
# a leftover header/footer line the splitter picked up, etc.), not real content.
MIN_CHUNK_CHARS = 40

# How many lines from the top/bottom of each page to scan for repeated
# running headers/footers.
EDGE_LINES = 3

# A line counts as a running header/footer once it (near-)repeats on at
# least this fraction of pages.
REPEAT_RATIO = 0.4

# Hard cap on a single corpus chunk's size. Markdown-header splitting alone
# doesn't bound chunk size -- a chapter with sparse headers (e.g. one H1
# spanning 50 pages of a real textbook) would otherwise become one enormous
# chunk that overflows both the embedding model's context window and any
# downstream LLM drafting call. This is what actually guarantees
# token-overflow safety for large documents, not the header split by itself.
MAX_CHUNK_CHARS = 3000
CHUNK_OVERLAP_CHARS = 200

# Per-page marker smuggled into the concatenated markdown text so each
# header-split section can be traced back to its starting page, without
# restructuring the header splitter itself. Must be plain ASCII: confirmed
# via live testing that MarkdownHeaderTextSplitter silently strips NUL bytes
# (and likely other control characters) from its output, which silently
# discarded an earlier \x00-based marker with no error -- ordinary printable
# text survives untouched.
PAGE_MARKER_RE = re.compile(r"@@QUILLPAGE:(\d+)@@")


def _page_marker(page_number: int) -> str:
    return f"@@QUILLPAGE:{page_number}@@"


# pymupdf4llm transitively imports onnxruntime (via pymupdf_layout's table/
# layout detection model), whose native DLL loading has shown intermittent
# failures on Windows during live testing -- confirmed to succeed on a plain
# retry, not a permanently broken environment. Same class of issue as the
# chromadb_rust_bindings DLL failure seen earlier in this project.
IMPORT_RETRY_ATTEMPTS = 5
IMPORT_RETRY_DELAY_SECONDS = 2.0


def _retry_import(import_fn, attempts: int = IMPORT_RETRY_ATTEMPTS, delay: float = IMPORT_RETRY_DELAY_SECONDS):
    """Calls `import_fn` (a zero-arg callable performing an import and
    returning whatever's needed), retrying on ImportError/OSError."""
    last_error = None
    for attempt in range(attempts):
        try:
            result = import_fn()
            if attempt > 0:
                print(f"    Import succeeded on attempt {attempt + 1}/{attempts}.")
            return result
        except (ImportError, OSError) as e:
            last_error = e
            print(f"    Import failed (attempt {attempt + 1}/{attempts}): {e}")
            if attempt < attempts - 1:
                time.sleep(delay)
    raise last_error


# Fallback grouping for documents where header detection clearly didn't find
# real structure (e.g. chapter dividers rendered as images with no
# extractable text -- confirmed to happen with real ebook-to-PDF
# conversions). Chunks get grouped into fixed-size page blocks instead of
# collapsing into one giant "chapter".
PAGE_BLOCK_SIZE = 25
MIN_PAGES_FOR_FALLBACK_CHECK = 15
DOMINANT_GROUP_RATIO = 0.7


_STRUCTURAL_LINE_RE = re.compile(r"^[\s|:\-*_=>`~.]*$")


def _is_structural_line(line: str) -> bool:
    """Blank lines and pure markdown punctuation (table separators like
    |---|---|, horizontal rules, stray emphasis markers). These repeat on
    nearly every page of real documents but aren't headers/footers, and
    stripping them broke tables."""
    return bool(_STRUCTURAL_LINE_RE.match(line))


def _normalize_line(line: str) -> str:
    """Collapses digits so 'Page 12' and 'Page 47' are recognized as the same
    running footer despite the page number changing."""
    return re.sub(r"\d+", "#", line.strip())


def _strip_repeated_lines(pages: list, edge_lines: int = EDGE_LINES, min_ratio: float = REPEAT_RATIO) -> list:
    """
    Detects lines that repeat near-identically across a large fraction of
    pages — i.e. running headers/footers — and strips them from every page.

    Args:
        pages: list of page dicts as returned by pymupdf4llm.to_markdown(..., page_chunks=True),
            each with a "text" key.

    Returns:
        List of cleaned page text strings, same order as `pages`.
    """
    if not pages:
        return []

    line_counts = Counter()
    per_page_lines = []

    for page in pages:
        lines = page["text"].splitlines()
        per_page_lines.append(lines)
        # A set, so each line counts at most once per page: on a short page
        # (<= 2 * edge_lines lines) the top and bottom slices overlap, and
        # double-counting there made every line of a one-page document hit
        # the threshold of 2 and get stripped as "boilerplate".
        edge = {
            _normalize_line(line) for line in lines[:edge_lines] + lines[-edge_lines:]
            if not _is_structural_line(line)
        }
        for normalized in edge:
            if normalized:
                line_counts[normalized] += 1

    threshold = max(2, int(len(pages) * min_ratio))
    boilerplate = {norm for norm, count in line_counts.items() if count >= threshold}

    # Only each page's edge lines are candidates for removal. Removing
    # matches anywhere on the page also deleted body lines that merely
    # resembled a footer after digit-collapsing ("Step 1" vs "Step 2").
    cleaned_pages = []
    for lines in per_page_lines:
        n = len(lines)
        kept = [
            line for i, line in enumerate(lines)
            if not ((i < edge_lines or i >= n - edge_lines)
                    and not _is_structural_line(line)
                    and _normalize_line(line) in boilerplate)
        ]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def _is_legal_boilerplate_page(text: str) -> bool:
    """True if a whole page's text looks like copyright/legal front matter
    (ISBN notices, "All Rights Reserved", "is a work of fiction"
    disclaimers, etc.) rather than real content."""
    return any(p.search(text) for p in LEGAL_BOILERPLATE_PATTERNS)


def _should_drop_legal_page(text: str, page_index: int, total_pages: int) -> bool:
    """Position-aware version of _is_legal_boilerplate_page used during
    ingestion: see FRONT_MATTER_* above."""
    hits = sum(1 for p in LEGAL_BOILERPLATE_PATTERNS if p.search(text))
    if not hits:
        return False
    front = max(FRONT_MATTER_MIN_PAGES, int(total_pages * FRONT_MATTER_RATIO))
    if page_index < front or page_index >= total_pages - BACK_MATTER_PAGES:
        return True
    return hits >= 2 and len(text) < LEGAL_SHORT_PAGE_CHARS


def _split_oversized_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list:
    """
    Splits `text` into pieces no longer than `max_chars`, preferring to break
    on paragraph boundaries so sentences aren't severed mid-thought. Falls
    back to a hard cut (with overlap, to avoid losing context at the seam)
    only when a single paragraph itself exceeds `max_chars`.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    pieces = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(para) <= max_chars:
            current = para
        else:
            start = 0
            while start < len(para):
                end = start + max_chars
                pieces.append(para[start:end])
                start = end - overlap if end < len(para) else end

    if current:
        pieces.append(current)
    return pieces


def _is_boilerplate_chunk(metadata: dict, text: str, min_chars: int = MIN_CHUNK_CHARS) -> bool:
    """True if a header-split chunk is boilerplate (references/index/etc.)
    or too short to be meaningful content."""
    for level in ("Header 1", "Header 2", "Header 3"):
        header = (metadata.get(level) or "").strip()
        if header and BOILERPLATE_HEADER_PATTERN.match(header):
            return True
    return len(text.strip()) < min_chars


def _extract_page_and_clean(text: str) -> tuple:
    """
    Pulls the first \\x00PAGE:N\\x00 sentinel out of `text` (inserted between
    pages before header-splitting so each resulting section can be traced
    back to roughly where it started) and returns (page_number_or_None,
    text_with_all_markers_stripped).
    """
    match = PAGE_MARKER_RE.search(text)
    page = int(match.group(1)) if match else None
    cleaned = PAGE_MARKER_RE.sub("", text).strip()
    return page, cleaned


def _attribute_pages_to_pieces(pieces: list, carry_from=None) -> list:
    """
    Given text pieces that may each contain zero or more embedded
    \\x00PAGE:N\\x00 markers, returns [(page_number_or_None, cleaned_text),
    ...] in the same order, with all markers stripped from the text.

    A piece with no marker of its own inherits the most recently seen page
    number rather than getting None -- it's still part of that same page's
    content (e.g. one page's paragraphs got split across multiple
    size-capped pieces, or a giant header-delimited section spanning dozens
    of pages got broken into pieces where only the first piece from each
    page carries that page's marker).
    """
    current_page = carry_from
    result = []
    for piece in pieces:
        marker_page, cleaned = _extract_page_and_clean(piece)
        if marker_page is not None:
            current_page = marker_page
        result.append((current_page, cleaned))
    return result


def _should_fallback_to_page_blocks(header_counts: dict, total_chunks: int, total_pages: int) -> bool:
    """
    True when header-based detection clearly failed to find real document
    structure -- either no headers at all, or one header's group swallowed
    almost everything -- for a document long enough that real structure
    would be expected. Short documents are left alone even with few/no
    headers, since a handful of page-block "Parts" would be pointless there.
    """
    if total_chunks == 0 or total_pages < MIN_PAGES_FOR_FALLBACK_CHECK:
        return False

    distinct_headers = {h for h in header_counts if h}
    if not distinct_headers:
        return True

    largest_group = max(header_counts.values())
    return (largest_group / total_chunks) >= DOMINANT_GROUP_RATIO


def _page_block_label(page: int, total_pages: int, block_size: int = PAGE_BLOCK_SIZE) -> str:
    """Labels a fallback page-block group, e.g. 'Part (pages 1-25)'."""
    start = (page // block_size) * block_size + 1
    end = min(start + block_size - 1, total_pages)
    return f"Part (pages {start}-{end})"


def _parse_pdf(filepath: str, filename: str, pymupdf4llm, markdown_splitter) -> tuple:
    """
    Converts one PDF to Markdown, filters out boilerplate (running
    headers/footers, references/index/bibliography sections, stray
    fragments), splits the rest by headers, and returns (chunks, stats):
    chunks without the corpus-wide "idx"/"title" fields (ingest_pdfs assigns
    those), and {"pages", "text_chars", "scanned", "chunks"}.
    """
    pages = pymupdf4llm.to_markdown(filepath, page_chunks=True)
    total_pages = len(pages)
    text_chars = sum(len(p["text"].strip()) for p in pages)
    stats = {
        "pages": total_pages,
        "text_chars": text_chars,
        "scanned": total_pages > 0 and text_chars / total_pages < SCANNED_CHARS_PER_PAGE,
    }
    if stats["scanned"]:
        print(f"    '{filename}' has almost no extractable text ({text_chars} chars over {total_pages} "
              f"page(s)) -- it looks scanned/image-only and would need OCR first.")

    cleaned_pages = _strip_repeated_lines(pages)
    # Replace (not remove) legal-boilerplate pages with empty
    # text so page indices/markers stay aligned with `pages`.
    cleaned_pages = [
        "" if _should_drop_legal_page(p, i, total_pages) else p for i, p in enumerate(cleaned_pages)
    ]
    md_text = "\n\n".join(
        f"{_page_marker(i)}\n{page_text}" for i, page_text in enumerate(cleaned_pages)
    )

    docs = markdown_splitter.split_text(md_text)

    chunks = []
    sections_kept = 0
    header_counts = Counter()
    last_seen_page = None

    for doc in docs:
        # Length is judged without the page markers: a run of
        # "@@QUILLPAGE:N@@" markers alone (an image-only PDF) used to pass
        # the minimum-length check and become an empty "chapter".
        if _is_boilerplate_chunk(doc.metadata, PAGE_MARKER_RE.sub("", doc.page_content)):
            continue
        sections_kept += 1

        header1 = doc.metadata.get('Header 1', '')
        header2 = doc.metadata.get('Header 2', '')
        header_context = f"{header1} > {header2}"

        # Page markers must survive into _split_oversized_text
        # (not be extracted beforehand) -- a section spanning
        # many pages (e.g. the page-block fallback trigger case)
        # embeds one marker per page throughout its raw text, and
        # each resulting sub-piece needs its *own* nearest page,
        # not the whole section's single starting page.
        raw_pieces = _split_oversized_text(doc.page_content.strip())
        attributed = _attribute_pages_to_pieces(raw_pieces, carry_from=last_seen_page)
        if attributed:
            last_seen_page = attributed[-1][0]

        for start_page, sub_text in attributed:
            if len(sub_text.strip()) < MIN_CHUNK_CHARS:
                continue  # e.g. a sliver left over from splitting around markers
            chunks.append({
                "text": f"Context: {header_context}\n\n{sub_text}",
                "source": filename,
                "header1": header1,
                "header2": header2,
                "page": start_page,
            })
            # Counted per resulting corpus chunk (post size-cap),
            # not per section, so the dominance ratio below
            # compares like-for-like -- a section that explodes
            # into 50 oversized sub-chunks must weigh 50, not 1.
            header_counts[header1] += 1

    if _should_fallback_to_page_blocks(header_counts, len(chunks), total_pages):
        print(
            f"    Header detection unreliable for {filename} "
            f"({len(header_counts)} heading(s) found across {total_pages} pages); "
            f"falling back to page-block grouping."
        )
        for chunk in chunks:
            page = chunk["page"] if chunk["page"] is not None else 0
            chunk["header1"] = _page_block_label(page, total_pages)

    print(
        f"    Kept {sections_kept}/{len(docs)} sections after boilerplate filtering "
        f"({len(chunks)} corpus chunk(s) after size-capping)."
    )
    stats["chunks"] = len(chunks)
    return chunks, stats


def manifest_path_for(output_json_path: str) -> str:
    """Sidecar file recording which version of each PDF corpus.json holds."""
    root, _ = os.path.splitext(output_json_path)
    return f"{root}.manifest.json"


def _file_signature(path: str) -> list:
    stat = os.stat(path)
    return [stat.st_size, stat.st_mtime_ns]


def load_manifest(output_json_path: str) -> dict:
    """{filename: {"sig": [size, mtime_ns], "pages", "chunks", "scanned", ...}}.
    Entries from the first manifest format (a bare signature list) are
    upgraded on read."""
    raw = _load_json(manifest_path_for(output_json_path), {})
    return {name: (entry if isinstance(entry, dict) else {"sig": entry}) for name, entry in raw.items()}


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: str, data) -> None:
    """Writes to a temp file and renames it into place, so a reader on
    another thread or process (the chat loop, `quill watch`) never sees a
    half-written file."""
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)


def ingest_pdfs(pdf_directory: str, output_json_path: str) -> list:
    """
    Brings corpus.json in line with the PDFs in `pdf_directory`: parses new
    or changed PDFs, reuses the stored chunks of unchanged ones (tracked by
    size + modification time in a sidecar manifest), and drops chunks of
    PDFs that were removed. Parsing a large PDF can take minutes, so
    re-parsing every PDF in the folder whenever one new file arrives -- the
    old behavior -- made each drop cost as much as the whole library.

    Returns the filenames that were (re)parsed this run.
    """
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory {pdf_directory} does not exist.")
        return []

    manifest_path = manifest_path_for(output_json_path)
    old_corpus = _load_json(output_json_path, []) if os.path.exists(output_json_path) else []
    old_manifest = load_manifest(output_json_path)
    old_chunks_by_source = {}
    for chunk in old_corpus:
        old_chunks_by_source.setdefault(chunk.get("source"), []).append(chunk)

    pdf_names = sorted(f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf"))
    to_parse = [
        name for name in pdf_names
        if old_manifest.get(name, {}).get("sig") != _file_signature(os.path.join(pdf_directory, name))
    ]

    if to_parse:
        print(f"Scanning {pdf_directory}: {len(to_parse)} new or changed PDF(s) to parse...")

        def _do_import():
            import pymupdf4llm as _pymupdf4llm
            from langchain_text_splitters import MarkdownHeaderTextSplitter as _MarkdownHeaderTextSplitter
            return _pymupdf4llm, _MarkdownHeaderTextSplitter

        pymupdf4llm, MarkdownHeaderTextSplitter = _retry_import(_do_import)
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])

    new_manifest = {}
    chunks_by_source = {}
    parsed = []
    for name in pdf_names:
        path = os.path.join(pdf_directory, name)
        if name not in to_parse:
            chunks_by_source[name] = old_chunks_by_source.get(name, [])
            new_manifest[name] = old_manifest[name]
            continue
        print(f"  - Parsing {name}...")
        try:
            chunks, stats = _parse_pdf(path, name, pymupdf4llm, markdown_splitter)
            chunks_by_source[name] = chunks
            new_manifest[name] = {"sig": _file_signature(path), **stats}
            parsed.append(name)
        except Exception as e:
            # Left out of the manifest, so it's retried on the next run.
            print(f"    Error processing {name}: {e}")
            chunks_by_source[name] = []

    corpus = []
    for name in pdf_names:
        for chunk in chunks_by_source[name]:
            idx = len(corpus)
            corpus.append({**chunk, "title": f"{name} - Chunk {idx}", "idx": idx})

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    _write_json_atomic(output_json_path, corpus)
    _write_json_atomic(manifest_path, new_manifest)

    if to_parse:
        print(f"Successfully processed {len(corpus)} chunks into {output_json_path}")
    return parsed
