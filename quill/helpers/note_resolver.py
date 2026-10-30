import os
import re
import json
from difflib import SequenceMatcher
from typing import List, Optional

from quill.helpers.note_versioning import IGNORED_VAULT_DIRS


def list_vault_notes(vault_dir: str) -> List[str]:
    """Every .md note in the vault as a '/'-separated vault-relative path,
    most recently modified first."""
    found = []
    for root, dirs, files in os.walk(vault_dir):
        dirs[:] = [d for d in dirs if d not in IGNORED_VAULT_DIRS and not d.startswith(".")]
        for filename in files:
            if filename.endswith(".md"):
                abs_path = os.path.join(root, filename)
                rel = os.path.relpath(abs_path, vault_dir).replace(os.sep, "/")
                found.append((os.path.getmtime(abs_path), rel))
    return [rel for _, rel in sorted(found, reverse=True)]


def _words(text: str) -> List[str]:
    return [w for w in re.split(r"[^\w]+", text.casefold()) if len(w) > 2 and w != "md"]


def candidate_notes(request: str, notes: List[str], recent: Optional[str] = None, limit: int = 15) -> List[str]:
    """
    The notes a request most plausibly refers to, best first, for an LLM to
    pick from -- a vault can hold thousands of notes, far too many to list in
    a prompt. Scored by how many of a note's name words (fuzzily) appear in
    the request; ties and no-match cases fall back to most-recently-modified.
    The note the conversation last touched is always included first, so
    follow-ups like "make it shorter" resolve.
    """
    request_words = _words(request)

    def score(note: str) -> float:
        total = 0.0
        for word in _words(note.replace("__", " ")):
            best = max((SequenceMatcher(None, word, r).ratio() for r in request_words), default=0.0)
            if best >= 0.8:
                total += best
        return total

    scored = sorted(((score(n), i, n) for i, n in enumerate(notes)), key=lambda t: (-t[0], t[1]))
    ranked = [n for _, _, n in scored]
    if recent in notes:
        ranked = [recent] + [n for n in ranked if n != recent]
    return ranked[:limit]


def match_note(name: Optional[str], notes: List[str]) -> Optional[str]:
    """Maps an LLM-returned note name onto an actual vault note: exact,
    then case-insensitive, then with '.md' added, then by bare filename."""
    if not name:
        return None
    name = name.strip().strip("\"'[]").replace("\\", "/")
    by_fold = {n.casefold(): n for n in notes}
    for candidate in (name, f"{name}.md"):
        if candidate in notes:
            return candidate
        if candidate.casefold() in by_fold:
            return by_fold[candidate.casefold()]
    basenames = {n.rsplit("/", 1)[-1].casefold(): n for n in notes}
    for candidate in (name, f"{name}.md"):
        hit = basenames.get(candidate.rsplit("/", 1)[-1].casefold())
        if hit:
            return hit
    return None


def parse_json_object(text: str) -> dict:
    """Extracts the first JSON object from an LLM reply, tolerating code
    fences and chatter around it (small local models rarely return bare
    JSON reliably). Raises ValueError if there isn't one."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    raise ValueError(f"No JSON object found in model reply: {text[:200]!r}")


_PREAMBLE_RE = re.compile(r"^\s*(here('s| is| are)|sure|certainly|below is|okay|ok)\b[^\n]*:\s*\n", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def strip_llm_wrapping(text: str) -> str:
    """Removes the 'Here is the revised note:' preamble and ```markdown
    fences models often wrap a returned document in."""
    text = _PREAMBLE_RE.sub("", text, count=1)
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1)
    return text.strip("\n") + "\n"
