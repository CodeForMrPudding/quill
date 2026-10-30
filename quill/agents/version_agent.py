import re
from datetime import datetime, timedelta
from typing import Callable, Optional

from langchain_core.messages import HumanMessage

from quill.helpers.note_versioning import NoteVersionControl
from quill.helpers.note_resolver import candidate_notes, list_vault_notes, match_note, parse_json_object

ACTIONS = ("status", "save", "history", "diff", "restore_note", "list_snapshots", "restore_vault")


# Only numbers the user tied to the word "version"/"commit"/"snapshot" count:
# a bare "#2" is often part of a filename ("Meeting #2.pdf").
_VERSION_NUMBER_RE = re.compile(r"\b(?:version|commit)\s*#?\s*(\d+)\b", re.IGNORECASE)
_SNAPSHOT_NUMBER_RE = re.compile(r"\bsnapshot\s*#?\s*(\d+)\b", re.IGNORECASE)
_TIME_WORDS_RE = re.compile(
    r"\b(yesterday|today|tonight|ago|morning|afternoon|evening|last night|week|month|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|o'clock|"
    r"january|february|march|april|june|july|august|september|october|november|december)\b"
    r"|\d\s*(?:am|pm)\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)


def _explicit_number(pattern: re.Pattern, text: str) -> Optional[int]:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def _pin_restore_target(plan: dict, request: str) -> dict:
    """
    Leaves exactly one way of choosing the version to restore, deciding from
    the user's own words where they're unambiguous. Live testing with
    llama3.1 showed it fills several target fields at once ("undo that" came
    back with steps_back=1 *and* before=<now>, which would restore the
    current version, i.e. do nothing) and misses explicit numbers ("version
    3" came back as steps_back=1).
    """
    if plan["action"] == "restore_note":
        number = _explicit_number(_VERSION_NUMBER_RE, request)
        if number is not None:
            return {**plan, "commit_id": number, "steps_back": None, "before": None}
        if plan.get("before") and _TIME_WORDS_RE.search(request):
            return {**plan, "commit_id": None, "steps_back": None}
        return {**plan, "commit_id": None, "before": None}

    if plan["action"] == "restore_vault":
        number = _explicit_number(_SNAPSHOT_NUMBER_RE, request)
        if number is not None:
            return {**plan, "snapshot_id": number, "before": None}
        if plan.get("before") and _TIME_WORDS_RE.search(request):
            return {**plan, "snapshot_id": None}
        return {**plan, "before": None}
    return plan


def parse_time(value) -> datetime:
    """Parses an LLM-supplied ISO time into a naive local datetime, the form
    commit timestamps are stored in. Models often add 'Z' or an offset, and
    comparing an aware datetime to a naive one raises TypeError."""
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


class VersionAgent:
    """
    Turns natural-language version-control requests ("save my notes",
    "undo that", "what changed in the Financials note?", "put the vault back
    to how it was yesterday") into operations on NoteVersionControl.

    Deliberately one plain LLM call that returns a JSON action, validated
    and executed in Python -- not an LLM tool-calling loop: a small local
    model given tools has been seen to loop on tool calls indefinitely (see
    NoteAgent.draft_note), and a raw-command tool would let it run anything.
    Every restore goes through a confirmation callback first.
    """

    def __init__(self, model, vcs: NoteVersionControl):
        self.model = model
        self.vcs = vcs

    def plan(self, request: str, recent_note: Optional[str] = None) -> dict:
        notes = list_vault_notes(self.vcs.vault_dir)
        candidates = candidate_notes(request, notes, recent=recent_note)
        snapshots = self.vcs.list_snapshots(limit=15)

        note_lines = "\n".join(f"- {n}" for n in candidates) or "- (no notes yet)"
        snapshot_lines = "\n".join(
            f"- #{sid} at {ts[:16]}: {msg} ({count} notes)" for sid, msg, ts, count in snapshots
        ) or "- (no vault snapshots yet)"

        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        prompt = f"""You manage version history for the user's Obsidian notes. Translate the user's
request into exactly ONE JSON object and reply with nothing else.

Current time: {datetime.now().isoformat(timespec="minutes")}
Note the conversation last worked on: {recent_note or "none"}

Candidate notes:
{note_lines}

Vault snapshots (newest first):
{snapshot_lines}

Possible actions:
{{"action": "status"}}  -- which notes have unsaved changes
{{"action": "save", "note": "<note>" or null, "message": "<short description>"}}  -- save a version of one note; null note = snapshot the whole vault
{{"action": "history", "note": "<note>"}}  -- list a note's saved versions
{{"action": "diff", "note": "<note>"}}  -- show what changed in a note since its last saved version
{{"action": "restore_note", "note": "<note>", "steps_back": <int or null>, "before": "<ISO time or null>", "commit_id": <int or null>}}  -- restore one note; "undo"/"revert that" means steps_back 1
{{"action": "list_snapshots"}}
{{"action": "restore_vault", "snapshot_id": <int or null>, "before": "<ISO time or null>"}}  -- restore every note to a vault snapshot
{{"action": "unknown", "reason": "<why>"}}

Use note names exactly as listed. Phrases like "it"/"that note" mean the note the conversation last worked on.
For restores, fill in only ONE of steps_back / before / commit_id (or snapshot_id / before) and set the rest to null.

Examples:
"undo that" -> {{"action": "restore_note", "note": "<last note>", "steps_back": 1, "before": null, "commit_id": null}}
"put X back to how it was yesterday" -> {{"action": "restore_note", "note": "X", "steps_back": null, "before": "{yesterday}T23:59", "commit_id": null}}
"go back to version 12 of X" -> {{"action": "restore_note", "note": "X", "steps_back": null, "before": null, "commit_id": 12}}
"restore my vault to yesterday" -> {{"action": "restore_vault", "snapshot_id": null, "before": "{yesterday}T23:59"}}
"snapshot everything" -> {{"action": "save", "note": null, "message": "Manual snapshot"}}

User request: {request}"""

        reply = self.model.invoke([HumanMessage(content=prompt)]).content
        try:
            plan = parse_json_object(reply)
        except ValueError:
            return {"action": "unknown", "reason": "couldn't understand the request"}

        if plan.get("action") not in ACTIONS:
            return {"action": "unknown", "reason": plan.get("reason") or "unsupported request"}

        if "note" in plan and plan["note"] is not None:
            resolved = match_note(str(plan["note"]), notes)
            if resolved is None:
                return {"action": "unknown", "reason": f"no note named '{plan['note']}' in the vault"}
            plan["note"] = resolved
        elif plan["action"] in ("history", "diff", "restore_note"):
            if recent_note in notes:
                plan["note"] = recent_note
            else:
                return {"action": "unknown", "reason": "which note? Mention it by name"}
        return _pin_restore_target(plan, request)

    def _resolve_note_target(self, plan: dict) -> int:
        note = plan["note"]
        ancestry = self.vcs.ancestors(note)
        if not ancestry:
            raise ValueError(f"'{note}' has no saved versions yet")

        if plan.get("commit_id") is not None:
            commit_id = int(plan["commit_id"])
            if commit_id not in {cid for cid, _, _ in self.vcs.log(note)}:
                raise ValueError(f"'{note}' has no version #{commit_id}")
            return commit_id
        if plan.get("before"):
            cutoff = parse_time(plan["before"])
            for cid, _, ts in ancestry:
                if datetime.fromisoformat(ts) <= cutoff:
                    return cid
            raise ValueError(f"'{note}' has no version from before {cutoff:%Y-%m-%d %H:%M}")

        steps = int(plan.get("steps_back") or 1)
        # With unsaved edits on disk, "undo" means "back to the last saved
        # version" -- that's one step, not two.
        if self.vcs.status(note)[0][1] == "modified":
            steps -= 1
        if steps >= len(ancestry):
            raise ValueError(f"'{note}' only has {len(ancestry)} saved version(s)")
        return ancestry[max(steps, 0)][0]

    def _resolve_snapshot(self, plan: dict) -> int:
        snapshots = self.vcs.list_snapshots(limit=1000)
        if not snapshots:
            raise ValueError("No vault snapshots exist yet")
        if plan.get("snapshot_id") is not None:
            sid = int(plan["snapshot_id"])
            if sid not in {s[0] for s in snapshots}:
                raise ValueError(f"No vault snapshot #{sid}")
            return sid
        if plan.get("before"):
            cutoff = parse_time(plan["before"])
            for sid, _, ts, _ in snapshots:
                if datetime.fromisoformat(ts) <= cutoff:
                    return sid
            raise ValueError(f"No vault snapshot from before {cutoff:%Y-%m-%d %H:%M}")
        return snapshots[0][0]

    def execute(self, plan: dict, confirm: Callable[[str], bool]) -> str:
        """Runs a plan from plan(). `confirm(question)` is called before
        anything that rewrites notes on disk; returning False cancels."""
        action = plan["action"]
        try:
            if action == "unknown":
                return (f"I couldn't map that to a version-control action ({plan.get('reason')}). "
                        "Try e.g. 'save my notes', 'undo that', or 'show the history of <note>'.")

            if action == "status":
                changed = [(p, s) for p, s in self.vcs.status() if s != "clean"]
                if not changed:
                    return "Every note is saved -- no unsaved changes."
                return "Notes with unsaved changes:\n" + "\n".join(f"  {s:>10}  {p}" for p, s in changed)

            if action == "save":
                message = plan.get("message") or "Saved from chat"
                if plan.get("note"):
                    commit_id = self.vcs.commit_if_changed(plan["note"], message)
                    if commit_id is None:
                        return f"'{plan['note']}' has no unsaved changes."
                    return f"Saved '{plan['note']}' as version #{commit_id}."
                sid = self.vcs.snapshot(message)
                return f"Saved a snapshot of the whole vault (#{sid})."

            if action == "history":
                return self.format_history(plan["note"])

            if action == "diff":
                diff = self.vcs.diff(plan["note"])
                return diff if diff else f"'{plan['note']}' has no changes since its last saved version."

            if action == "restore_note":
                note = plan["note"]
                target = self._resolve_note_target(plan)
                _, message, ts = next(e for e in self.vcs.log(note) if e[0] == target)
                if not confirm(f"Restore '{note}' to version #{target} ({ts[:16]}, \"{message}\")?"):
                    return "Cancelled -- nothing was changed."
                dirty = self.vcs.status(note)[0][1] == "modified"
                self.vcs.checkout(note, target)
                reply = f"Restored '{note}' to version #{target}."
                if dirty:
                    reply += " Your unsaved edits were stashed -- /stash-pop to bring them back."
                return reply

            if action == "list_snapshots":
                return self.format_snapshots()

            if action == "restore_vault":
                sid = self._resolve_snapshot(plan)
                _, message, ts, count = next(s for s in self.vcs.list_snapshots(limit=1000) if s[0] == sid)
                if not confirm(f"Restore all {count} note(s) in vault snapshot #{sid} ({ts[:16]}, \"{message}\")?"):
                    return "Cancelled -- nothing was changed."
                result = self.vcs.restore_snapshot(sid)
                return format_restore_result(sid, result)
        except (ValueError, TypeError) as e:  # bad ids/times from the model, or invalid state
            return f"Couldn't do that: {e}"
        return "Nothing to do."

    def format_history(self, note: str) -> str:
        head = self.vcs.head(note)
        entries = self.vcs.log(note)
        if not entries:
            return f"No saved versions of '{note}'."
        lines = [f"Versions of '{note}' (newest first):"]
        for cid, message, ts in entries:
            marker = "  <- current" if cid == head else ""
            lines.append(f"  #{cid}  {ts[:16]}  {message}{marker}")
        return "\n".join(lines)

    def format_snapshots(self) -> str:
        snapshots = self.vcs.list_snapshots()
        if not snapshots:
            return "No vault snapshots yet. Say 'save a snapshot of my vault' or run /snapshot."
        return "Vault snapshots (newest first):\n" + "\n".join(
            f"  #{sid}  {ts[:16]}  {msg}  ({count} notes)" for sid, msg, ts, count in snapshots
        )


def format_restore_result(snapshot_id: int, result: dict) -> str:
    lines = [f"Restored vault snapshot #{snapshot_id}: {len(result['restored'])} note(s) changed."]
    if result["kept"]:
        lines.append(f"Kept {len(result['kept'])} note(s) created after that snapshot: " + ", ".join(result["kept"]))
    lines.append(f"To undo this restore, restore snapshot #{result['safety_snapshot']}.")
    return "\n".join(lines)

