import click
import os
import time
from typing import Literal

# LangChain / Graph Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import Agents
from quill.agents.note_agent import NoteAgent, format_context
from quill.agents.version_agent import VersionAgent, format_restore_result

# Note version control
from quill.helpers.note_versioning import NoteVersionControl
from quill.helpers.note_autosnapshot import NoteAutoSnapshotter
from quill.helpers.note_resolver import candidate_notes, list_vault_notes, match_note
from quill.helpers.llm_config import (
    CHAT_REPLY_TOKENS, MODEL_NAME, NUM_CTX, check_ollama, input_char_budget, make_chat_model, trim_history,
)

# Import session manager
from quill.helpers.session_manager import SessionManager

# Import daemon/watcher components
from quill.helpers.pending_drafts import PendingDraftStore
from quill.helpers.vault_watcher import VaultWatcher
from quill.helpers.pdf_ingestion import ingest_pdfs
from quill.helpers.dashboard import gather_dashboard_stats, human_size
from quill.helpers.note_planner import split_leading_arg
from quill.helpers.paths import migrate_legacy_layout, resolve_paths
from quill.helpers.instance_lock import InstanceLock
from quill.helpers.watch_flow import PdfWatchFlow
from quill.helpers.watch_jobs import WatchJobStore

# Keep in sync with the `version` field in pyproject.toml.
__version__ = "0.2.0"


def print_logo():
    logo = r"""
        (`/\
        `=\/\   quill
         `=\/\    v1
          `=\/
             \
    """
    click.secho(logo, fg="cyan", bold=True)


EXIT_WORDS = frozenset(['exit', 'quit', 'q', '/exit', '/quit', '/q', 'bye', 'goodbye'])


INTENTS = ("EXIT", "VERSION", "EDIT", "NOTE", "CHAT")


def parse_intent(response: str) -> Literal["EXIT", "VERSION", "EDIT", "NOTE", "CHAT"]:
    """
    Reads the category out of the router's reply, matching whole words only.
    A reply that starts with a category (the requested format) is taken at
    its word; otherwise the *last* category mentioned wins, so chatty
    replies like "Not EXIT -- this is CHAT" resolve to CHAT. The old
    substring check returned EXIT there and ended the session.
    """
    words = [w.strip(".,:;*!()[]-") for w in response.upper().replace("'", " ").replace('"', " ").split()]
    found = [w for w in words if w in INTENTS]
    if not found:
        return "CHAT"
    return words[0] if words[0] in INTENTS else found[-1]


def determine_intent(user_input: str, model) -> Literal["EXIT", "VERSION", "EDIT", "NOTE", "CHAT"]:
    """
    Simple router to decide which agent to handle the request.
    """
    prompt = f"""Classify the user's message into exactly one category:
EXIT    - wants to end the chat session ("goodbye", "let's stop here")
VERSION - save or snapshot notes, see a note's history or what changed, undo, revert, or restore notes or the whole vault to an earlier version
EDIT    - change an EXISTING note: rewrite, expand, shorten, fix, reorganize, or add to it
NOTE    - create a NEW note about a topic
CHAT    - anything else, including questions about the user's documents or notes

Message: "{user_input}"

Answer with the category word only."""
    return parse_intent(model.invoke(prompt).content)


def print_diff(diff: str, max_lines: int = 80) -> None:
    lines = diff.splitlines()
    for line in lines[:max_lines]:
        if line.startswith("+") and not line.startswith("+++"):
            click.secho(line, fg="green")
        elif line.startswith("-") and not line.startswith("---"):
            click.secho(line, fg="red")
        elif line.startswith("@@"):
            click.secho(line, fg="cyan")
        else:
            click.secho(line, dim=True)
    if len(lines) > max_lines:
        click.secho(f"... ({len(lines) - max_lines} more diff lines)", dim=True)


def handle_edit_request(user_input: str, note_agent: NoteAgent, note_vcs: NoteVersionControl, recent_note):
    """
    Chat-driven note editing: picks the note the request is about, has the
    model rewrite it, shows the diff, and on approval saves it as a new
    version (so "undo that" can revert it). Returns (reply, note_or_None).
    """
    from difflib import unified_diff

    notes = list_vault_notes(note_agent.directory)
    if not notes:
        return "There are no notes in the vault to edit yet.", None

    plan = note_agent.choose_edit_target(user_input, candidate_notes(user_input, notes, recent=recent_note), recent_note)
    note = match_note(plan["note"], notes) or (recent_note if plan["note"] is None and recent_note in notes else None)
    if note is None:
        return "I couldn't tell which note you mean -- mention it by name (e.g. 'in Meeting_2pdf__Financials, ...').", None

    with open(os.path.join(note_agent.directory, *note.split("/")), "r", encoding="utf-8") as f:
        current = f.read()

    click.secho(f"[Quill]: Editing '{note}': {plan['instruction']}", fg="magenta")
    try:
        revised = note_agent.edit_note(note, current, plan["instruction"])
    except ValueError as e:
        return str(e), note

    diff = "".join(unified_diff(
        current.splitlines(keepends=True), revised.splitlines(keepends=True),
        fromfile=f"{note} (current)", tofile=f"{note} (edited)",
    ))
    if not diff:
        return f"The edit didn't change anything in '{note}'.", note

    print_diff(diff)
    if not click.confirm("Apply this edit?", default=True):
        return "Edit discarded -- the note is unchanged.", note

    commit_id = note_vcs.save(note, revised, f"Chat edit: {plan['instruction'][:60]}")
    return f"Updated '{note}' (version #{commit_id}). Say 'undo that' to revert it.", note


HELP_TEXT = """\
Chat: ask questions about your documents, or just say what you want:
  "write a note on X"  "in <note>, add a section on Y"  "undo that"
  "what changed in <note>?"  "save a snapshot of my vault"
  "restore my vault to yesterday"

Sessions:      /new  /list  /rename <name>
Note history:  /status [note]  /log <note>  /diff <note> [from_id] [to_id]
               /commit <note> <message>  /checkout <note> <id|latest>
               /stash [note]  /stash-pop <note> [id]
Vault:         /snapshot [message]  /snapshots  /restore-snapshot <id>
Documents:     /focus <pdf>  (limit questions to one PDF; /focus alone clears)
PDF drafts:    /chapters <pdf>  /draft <pdf> <chapter>  /draft-all <pdf>
               /pending  /approve <draft>  /approve-all <pdf>
               /discard <draft>  /discard-all <pdf>
Exit:          exit / quit / q
Edits you make in Obsidian are versioned automatically after 30s of quiet."""


def print_chapter_checkpoints(note_agent: NoteAgent, source: str):
    """
    Prints the cheap, LLM-free chapter listing for an already-ingested source
    document, so the user can see what's available and pick which chapters
    are worth the cost of actually drafting.
    """
    checkpoints = note_agent.list_chapters(source)
    if not checkpoints:
        click.secho(note_agent.source_status(source) or f"No chapters found for '{source}'.", fg="yellow")
        return

    click.secho(f"\n[Quill]: {len(checkpoints)} chapter(s) found in '{source}':", fg="magenta")
    for cp in checkpoints:
        click.secho(f"  - {cp['title']}  ({cp['chunk_count']} chunk(s), ~{cp['char_count']} chars)", fg="yellow")
        click.secho(f"      \"{cp['preview']}...\"", dim=True)
    click.secho(
        f"Draft one with /draft {source} <chapter title>, or everything with /draft-all {source}.",
        fg="magenta"
    )


def build_pdf_handler(note_agent: NoteAgent, pending_store: PendingDraftStore):
    """
    Returns the callback VaultWatcher invokes (on its own background thread)
    once a newly-dropped PDF's write has stabilized. Ingests it and lists the
    chapters found (cheap -- no LLM calls, near-instant regardless of book
    size) rather than eagerly drafting every chapter, which for a large
    document would mean many expensive LLM calls before the user even knows
    what's in it. Actual drafting happens on demand via /draft or /draft-all.
    Never writes into the vault directly and never touches stdin, since it
    runs off the main thread.
    """

    def handle_new_pdf(pdf_path: str):
        filename = os.path.basename(pdf_path)
        try:
            parsed = ingest_pdfs(note_agent.pdf_source_dir, note_agent.corpus_file)
            if parsed:
                note_agent.setup_retrieval()
        except Exception as e:
            click.secho(f"[Quill Watcher]: Ingestion failed for '{filename}': {e}", fg="red")
            return
        # Modified-file events for a PDF that's already ingested and
        # unchanged parse nothing; stay quiet for those.
        if filename in parsed:
            click.secho(f"\n[Quill Watcher]: Ingested new PDF '{filename}'.", fg="magenta")
            print_chapter_checkpoints(note_agent, filename)

    return handle_new_pdf


def select_session(session_manager: SessionManager):
    """
    Displays the last 5 sessions and lets the user choose or start new.
    """
    sessions = session_manager.get_recent_sessions(limit=5)

    if not sessions:
        return session_manager.create_session(name="New Session")

    click.secho("\n--- Recent Sessions ---", fg="yellow")
    for idx, (sid, name, time) in enumerate(sessions):
        short_time = str(time).split('.')[0]
        click.secho(f"{idx + 1}. {name} ({short_time})", fg="cyan")

    click.secho("N. Start New Chat", fg="green")

    choice = click.prompt("Select a session", default="N")

    if choice.upper() == 'N':
        return session_manager.create_session(name="New Session")

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            click.secho(f"Resuming '{sessions[idx][1]}'...", dim=True)
            return sessions[idx][0]
        else:
            click.secho("Invalid selection, starting new session.", fg="red")
            return session_manager.create_session()
    except ValueError:
        return session_manager.create_session()


def path_options(f):
    """--repo/--vault/--drop-dir on every command (see quill.helpers.paths
    for the defaults and the matching environment variables)."""
    f = click.option('--drop-dir', default=None,
                     help='PDF library folder that gets ingested [env QUILL_DROP_DIR; default <repo>/pdf_drop].')(f)
    f = click.option('--vault', default=None,
                     help='Obsidian vault for notes [env QUILL_VAULT; default <repo>/obsidian_vault].')(f)
    f = click.option('--repo', default=None,
                     help='Quill home folder [env QUILL_HOME; default: this folder if it has a Quill layout, else ~/Quill].')(f)
    return f


def prepare_paths(repo, vault, drop_dir):
    paths = resolve_paths(home=repo, vault=vault, drop=drop_dir)
    moved = migrate_legacy_layout(paths)
    if moved:
        click.secho(f"[Quill]: Moved {', '.join(moved)} into {paths.data} (Quill's hidden data folder).", dim=True)
    return paths


@click.group(invoke_without_command=True)
@path_options
@click.version_option(version=__version__, prog_name='quill')
@click.pass_context
def cli(ctx, repo, vault, drop_dir):
    """Quill: The Obsidian AI Assistant."""
    if ctx.invoked_subcommand is None:
        run_repl(prepare_paths(repo, vault, drop_dir))


@cli.command()
@path_options
def dashboard(repo, vault, drop_dir):
    """Show note locations, storage usage, and index/version-control stats."""
    stats = gather_dashboard_stats(paths=resolve_paths(home=repo, vault=vault, drop=drop_dir))

    click.secho("\n=== Quill Dashboard ===", fg="cyan", bold=True)
    click.secho(f"Repo:     {stats['repo_dir']}", fg="white")
    click.secho(f"Vault:    {stats['vault_dir']}", fg="white")
    click.secho(f"PDF drop: {stats['drop_dir']}", fg="white")

    notes = stats["notes"]
    click.secho(f"\nNotes: {len(notes)}", fg="yellow", bold=True)
    for name, size in notes[:20]:
        click.secho(f"  - {name}  ({human_size(size)})", fg="white")
    if len(notes) > 20:
        click.secho(f"  ... and {len(notes) - 20} more", dim=True)

    pdfs = stats["pending_pdfs"]
    click.secho(f"\nPDFs waiting in drop folder: {len(pdfs)}", fg="yellow", bold=True)
    for name, size in pdfs:
        click.secho(f"  - {name}  ({human_size(size)})", fg="white")

    corpus = stats["corpus"]
    click.secho(
        f"\nIngested corpus: {corpus['chunk_count']} chunk(s) from {corpus['source_count']} source document(s)",
        fg="yellow", bold=True
    )
    click.secho(f"Vector index size: {human_size(stats['vector_index_size'])}", fg="white")

    vcs = stats["version_control"]
    click.secho(
        f"\nVersion control: {vcs['tracked_notes']} tracked note(s), "
        f"{vcs['commits']} commit(s), {vcs['stashes']} stash(es)",
        fg="yellow", bold=True
    )
    click.secho(f"Version store size: {human_size(vcs['db_size'])}", fg="white")

    sessions = stats["sessions"]
    click.secho(
        f"\nChat sessions: {sessions['session_count']} session(s), {sessions['message_count']} message(s)",
        fg="yellow", bold=True
    )
    click.secho(f"Session store size: {human_size(sessions['db_size'])}", fg="white")

    click.secho(f"\nTotal vault disk usage: {human_size(stats['vault_total_size'])}", fg="cyan", bold=True)


@cli.command()
@path_options
def chat(repo, vault, drop_dir):
    """
    Start an interactive chat session with Quill: shows the logo, restores
    previous session history, and answers using context from your ingested
    notes/documents when relevant (falling back to general knowledge
    otherwise). Equivalent to running `quill` with no subcommand.
    """
    run_repl(prepare_paths(repo, vault, drop_dir))


@cli.command()
@path_options
@click.option('--watch-dir', default=None,
              help='Also watch this folder (e.g. your Downloads) for new PDFs; approved ones are copied into '
                   'the PDF library. PDFs already there when Quill starts are not offered.')
@click.option('--auto-approve', is_flag=True,
              help='Draft every new PDF without asking (for systems without Windows notifications).')
def watch(repo, vault, drop_dir, watch_dir, auto_approve):
    """
    Standalone background watcher: monitors the PDF folder independently of
    any chat session. Asks (via a Windows notification) whether to generate
    notes for each new PDF; on "Yes", drafts each chapter and saves it into
    the vault as soon as it's ready. Interrupted books resume on the next
    start. Leave this running in a terminal -- Ctrl+C to stop.
    """
    paths = prepare_paths(repo, vault, drop_dir)
    lock = InstanceLock(paths.watch_lock)
    if not lock.acquire():
        click.secho("Error: another `quill watch` (or a chat session's watcher) is already watching this vault.",
                    fg="red")
        return

    # Deliberately imported before windows_toasts below: confirmed via live
    # testing that windows_toasts' WinRT/COM initialization, if it runs
    # first, makes pymupdf4llm's onnxruntime dependency fail to load its
    # native DLL for the rest of the process's lifetime (deterministic
    # ordering conflict, not a transient race -- reproduced 3/3 each way).
    from quill.helpers.pdf_ingestion import _retry_import

    def _warm_up_pymupdf4llm():
        import pymupdf4llm
        return pymupdf4llm

    try:
        _retry_import(_warm_up_pymupdf4llm)
    except ImportError as e:
        click.secho(f"Error: Could not load pymupdf4llm: {e}", fg="red")
        return

    notifier = None
    if not auto_approve:
        try:
            from quill.helpers.notifications import ToastNotifier
            notifier = ToastNotifier("Quill")
        except Exception as e:
            click.secho(f"[Quill Watch]: Windows notifications unavailable ({e}); "
                        f"new PDFs will be listed but not drafted unless you pass --auto-approve.", fg="yellow")
    if notifier is None:
        from quill.helpers.notifications import ConsoleNotifier
        notifier = ConsoleNotifier(auto_approve=auto_approve)

    llm = make_chat_model()
    try:
        check_ollama(llm)
    except RuntimeError as e:
        click.secho(f"Error: {e}", fg="red")
        return
    click.secho(f"[Quill]: Using {MODEL_NAME} with a {NUM_CTX:,}-token context window.", dim=True)

    try:
        note_agent = NoteAgent(llm, paths.vault, pdf_source_dir=paths.drop, data_dir=paths.data)
        note_vcs = NoteVersionControl(paths.vault)
    except Exception as e:
        click.secho(f"Agent Init Error: {e}", fg="red")
        return

    flow = PdfWatchFlow(note_agent, note_vcs, notifier, WatchJobStore(paths.watch_jobs), log=click.secho)
    flow.start()
    # Offer PDFs that arrived while nothing was running, and resume any
    # book whose drafting was interrupted.
    flow.catch_up()

    snapshotter = NoteAutoSnapshotter(note_vcs)
    snapshotter.start()
    watchers = [VaultWatcher(paths.drop, on_pdf_ready=flow.on_library_pdf)]
    if watch_dir and os.path.abspath(watch_dir) != paths.drop:
        watchers.append(VaultWatcher(os.path.abspath(watch_dir), on_pdf_ready=flow.on_inbox_pdf))
    for w in watchers:
        w.start()
        click.secho(f"[Quill Watch]: Watching '{w.drop_dir}' for new PDFs.", fg="cyan", bold=True)
    click.secho(f"[Quill Watch]: Versioning edits to notes in '{paths.vault}' automatically. Ctrl+C to stop.",
                dim=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.secho("\n[Quill Watch]: Stopping (an in-progress book resumes next start)...", fg="red")
    finally:
        for w in watchers:
            w.stop()
        flow.stop()
        snapshotter.stop()
        lock.release()


def run_repl(paths):
    """The interactive Quill assistant loop -- the default action when no subcommand is given."""
    knowledge_dir = paths.vault
    pdf_drop_dir = paths.drop

    # 1. Initialize Components
    session_manager = SessionManager(paths.sessions_db)

    llm = make_chat_model()
    try:
        check_ollama(llm)
    except RuntimeError as e:
        click.secho(f"Error: {e}", fg="red")
        return
    click.secho(f"[Quill]: Using {MODEL_NAME} with a {NUM_CTX:,}-token context window.", dim=True)

    try:
        note_agent = NoteAgent(llm, knowledge_dir, pdf_source_dir=pdf_drop_dir, data_dir=paths.data)
        note_vcs = NoteVersionControl(knowledge_dir)
        version_agent = VersionAgent(llm, note_vcs)
    except Exception as e:
        click.secho(f"Agent Init Error: {e}", fg="red")
        return

    pending_store = PendingDraftStore(knowledge_dir, store_dir=paths.pending_drafts)
    if pending_store.list():
        click.secho(f"[Quill]: {len(pending_store.list())} draft(s) from earlier are waiting -- /pending to review.",
                    fg="yellow")
    # Only one process watches the PDF folder; if `quill watch` is running,
    # it handles new PDFs and this session doesn't duplicate the work.
    watch_lock = InstanceLock(paths.watch_lock)
    watcher = None
    if watch_lock.acquire():
        watcher = VaultWatcher(pdf_drop_dir, on_pdf_ready=build_pdf_handler(note_agent, pending_store))
        watcher.start()
        click.secho(f"[Quill]: Watching '{pdf_drop_dir}' for new PDFs.", dim=True)
    else:
        click.secho("[Quill]: `quill watch` is running, so it's handling new PDFs.", dim=True)
    # Chat retrieval scope set with /focus (None = all documents).
    chat_focus = None
    snapshotter = NoteAutoSnapshotter(note_vcs)
    snapshotter.start()
    click.secho("[Quill]: Edits to your notes are versioned automatically. Type /help for commands.", dim=True)

    # The note the conversation last created, edited, or restored, so
    # follow-ups like "make it shorter" or "undo that" know what "it" is.
    recent_note = None

    print_logo()

    # 2. Session Selection
    current_session_id = select_session(session_manager)

    # 3. Load History
    history = session_manager.load_history(current_session_id)
    if history:
        click.secho(f"\n[Restored {len(history)} messages from history]\n", dim=True)
        # Replay last message for context if needed, or just let user type
        if isinstance(history[-1], AIMessage):
            click.secho(f"[Last Reply]: {history[-1].content}", fg="cyan")

    # 4. Main Loop
    while True:
        try:
            user_input = click.prompt(click.style("\n> You", fg="white", bold=True))

            # --- CLI Commands ---
            if user_input.strip().lower() in EXIT_WORDS:
                click.secho("[Quill]: Goodbye!", fg="cyan")
                break

            if user_input.lower() == '/new':
                current_session_id = session_manager.create_session()
                history = []
                click.secho("--- New Session Started ---", fg="green")
                continue

            if user_input.lower() == '/list':
                # Quick switch logic
                current_session_id = select_session(session_manager)
                history = session_manager.load_history(current_session_id)
                click.secho(f"[Quill]: Switched to session: {current_session_id}", fg="yellow")
                continue

            if user_input.lower().startswith('/rename '):
                new_name = user_input[8:].strip()
                session_manager.rename_session(current_session_id, new_name)
                click.secho(f"[Quill]: Session renamed to: {new_name}", fg="green")
                continue

            # --- Note Version Control Commands ---
            if user_input.startswith('/'):
                cmd, _, arg_text = user_input.strip().partition(' ')
                cmd = cmd.lower()
                arg_text = arg_text.strip()

                if cmd == '/commit':
                    note_path, message = split_leading_arg(arg_text, suffix=".md")
                    if not note_path or not message:
                        click.secho("Usage: /commit <note_path> <message>", fg="red")
                        continue
                    try:
                        commit_id = note_vcs.commit(note_path, message)
                        click.secho(f"[Quill]: Committed '{note_path}' as #{commit_id}", fg="green")
                    except (FileNotFoundError, ValueError) as e:
                        click.secho(f"Error: {e}", fg="red")
                    continue

                if cmd == '/status':
                    target = split_leading_arg(arg_text, suffix=".md")[0] or None
                    for path, state in note_vcs.status(target):
                        click.secho(f"  {state:>10}  {path}", fg="yellow")
                    continue

                if cmd == '/help':
                    click.secho(HELP_TEXT, fg="cyan")
                    continue

                if cmd == '/log':
                    note_path = split_leading_arg(arg_text, suffix=".md")[0]
                    if not note_path:
                        click.secho("Usage: /log <note_path>", fg="red")
                        continue
                    click.secho(version_agent.format_history(note_path), fg="cyan")
                    continue

                if cmd == '/checkout':
                    note_path, rest = split_leading_arg(arg_text, suffix=".md")
                    if not note_path or not rest:
                        click.secho("Usage: /checkout <note_path> <commit_id|latest>", fg="red")
                        continue
                    try:
                        target = rest.split()[0].lstrip('#')
                        commit_id = note_vcs.latest(note_path) if target.lower() == "latest" else int(target)
                        if commit_id is None:
                            raise ValueError(f"'{note_path}' has no saved versions")
                        note_vcs.checkout(note_path, commit_id)
                        click.secho(f"[Quill]: '{note_path}' checked out to #{commit_id}", fg="green")
                        recent_note = note_path
                    except (ValueError, OSError) as e:
                        click.secho(f"Error: {e}", fg="red")
                    continue

                if cmd == '/diff':
                    note_path, rest = split_leading_arg(arg_text, suffix=".md")
                    if not note_path:
                        click.secho("Usage: /diff <note_path> [from_id] [to_id]", fg="red")
                        continue
                    try:
                        ids = [int(x.lstrip('#')) for x in rest.split()[:2]]
                        diff = note_vcs.diff(note_path, *ids)
                    except (ValueError, OSError) as e:
                        click.secho(f"Error: {e}", fg="red")
                        continue
                    if diff:
                        print_diff(diff, max_lines=400)
                    else:
                        click.secho("No differences.", fg="yellow")
                    continue

                if cmd == '/snapshot':
                    snapshot_id = note_vcs.snapshot(arg_text or "Manual snapshot")
                    click.secho(f"[Quill]: Saved vault snapshot #{snapshot_id}", fg="green")
                    continue

                if cmd == '/snapshots':
                    click.secho(version_agent.format_snapshots(), fg="cyan")
                    continue

                if cmd == '/restore-snapshot':
                    try:
                        snapshot_id = int(arg_text.split()[0].lstrip('#'))
                    except (ValueError, IndexError):
                        click.secho("Usage: /restore-snapshot <snapshot_id>  (see /snapshots)", fg="red")
                        continue
                    if not click.confirm(f"Restore every note to vault snapshot #{snapshot_id}?", default=False):
                        click.secho("Cancelled.", fg="yellow")
                        continue
                    try:
                        result = note_vcs.restore_snapshot(snapshot_id)
                        click.secho(f"[Quill]: {format_restore_result(snapshot_id, result)}", fg="green")
                    except ValueError as e:
                        click.secho(f"Error: {e}", fg="red")
                    continue

                if cmd == '/stash':
                    target = split_leading_arg(arg_text, suffix=".md")[0] or None
                    entries = note_vcs.stash_list(target)
                    if not entries:
                        click.secho("No stashed changes.", fg="yellow")
                    for stash_id, path, timestamp in entries:
                        click.secho(f"  #{stash_id}  {timestamp}  {path}", fg="cyan")
                    continue

                if cmd == '/stash-pop':
                    note_path, rest = split_leading_arg(arg_text, suffix=".md")
                    if not note_path:
                        click.secho("Usage: /stash-pop <note_path> [stash_id]", fg="red")
                        continue
                    try:
                        stash_id = int(rest.split()[0].lstrip('#')) if rest else None
                        note_vcs.stash_pop(note_path, stash_id)
                        click.secho(f"[Quill]: Restored stash onto '{note_path}'", fg="green")
                    except ValueError as e:
                        click.secho(f"Error: {e}", fg="red")
                    continue

                # --- Watcher-drafted notes awaiting approval ---
                if cmd == '/pending':
                    drafts = pending_store.list()
                    if not drafts:
                        click.secho("No pending drafts.", fg="yellow")
                    for name, source in drafts:
                        click.secho(f"  {name}  (from {source})", fg="yellow")
                    continue

                if cmd == '/approve':
                    draft_names = [name for name, _ in pending_store.list()]
                    draft_name = split_leading_arg(arg_text, suffix=".md", known=draft_names)[0]
                    if not draft_name:
                        click.secho("Usage: /approve <draft_name>", fg="red")
                        continue
                    try:
                        content = pending_store.take(draft_name)
                        commit_id = note_vcs.save(draft_name, content, "Initial version from watcher draft")
                        click.secho(f"[Quill]: Saved '{draft_name}' and committed as #{commit_id}", fg="green")
                    except KeyError as e:
                        click.secho(f"Error: {e}", fg="red")
                    continue

                if cmd == '/discard':
                    draft_names = [name for name, _ in pending_store.list()]
                    draft_name = split_leading_arg(arg_text, suffix=".md", known=draft_names)[0]
                    if not draft_name:
                        click.secho("Usage: /discard <draft_name>", fg="red")
                        continue
                    try:
                        pending_store.discard(draft_name)
                        click.secho(f"[Quill]: Discarded draft '{draft_name}'", fg="green")
                    except KeyError as e:
                        click.secho(f"Error: {e}", fg="red")
                    continue

                # Drafts are tagged with the source they came from; resolve
                # against both pending and ingested sources so names with
                # spaces ("Meeting #2.pdf") work unquoted.
                pending_sources = {source for _, source in pending_store.list()}
                known_sources = sorted(pending_sources | set(note_agent.list_sources()))

                if cmd == '/approve-all':
                    source = split_leading_arg(arg_text, suffix=".pdf", known=known_sources)[0]
                    if not source:
                        click.secho("Usage: /approve-all <source_filename>", fg="red")
                        continue
                    names = pending_store.list_for_source(source)
                    if not names:
                        click.secho(f"No pending drafts for '{source}'", fg="yellow")
                        continue
                    for name in names:
                        content = pending_store.take(name)
                        commit_id = note_vcs.save(name, content, f"Initial version from watcher draft ({source})")
                        click.secho(f"[Quill]: Saved '{name}' and committed as #{commit_id}", fg="green")
                    continue

                if cmd == '/discard-all':
                    source = split_leading_arg(arg_text, suffix=".pdf", known=known_sources)[0]
                    if not source:
                        click.secho("Usage: /discard-all <source_filename>", fg="red")
                        continue
                    discarded = pending_store.discard_all(source)
                    if not discarded:
                        click.secho(f"No pending drafts for '{source}'", fg="yellow")
                    else:
                        click.secho(f"[Quill]: Discarded {len(discarded)} draft(s) for '{source}'", fg="green")
                    continue

                # --- Chapter checkpoints: list cheaply, draft on demand ---
                if cmd == '/chapters':
                    source = split_leading_arg(arg_text, suffix=".pdf", known=known_sources)[0]
                    if not source:
                        click.secho("Usage: /chapters <source_filename>", fg="red")
                        continue
                    print_chapter_checkpoints(note_agent, source)
                    continue

                if cmd == '/draft':
                    source, chapter_title = split_leading_arg(arg_text, suffix=".pdf", known=known_sources)
                    chapter_title = chapter_title.strip().strip("\"'")
                    if not source or not chapter_title:
                        click.secho("Usage: /draft <source_filename> <chapter title>", fg="red")
                        continue
                    if not note_agent.list_chapters(source):
                        click.secho(note_agent.source_status(source), fg="yellow")
                        continue
                    click.secho(f"[Quill]: Drafting '{chapter_title}' from '{source}'... (Ctrl+C to cancel)",
                                fg="magenta")
                    try:
                        pages = note_agent.draft_multi_page_notes(source, chapters=[chapter_title])
                    except KeyboardInterrupt:
                        click.secho("[Quill]: Drafting cancelled.", fg="yellow")
                        continue
                    except Exception as e:
                        click.secho(f"Error: Drafting failed: {e}", fg="red")
                        continue
                    if not pages:
                        click.secho(
                            f"No chapter matching '{chapter_title}' found for '{source}'. "
                            f"Run /chapters {source} to see exact titles.",
                            fg="yellow"
                        )
                        continue
                    for page_name, content in pages.items():
                        pending_store.add(page_name, content, source=source)
                    click.secho(f"[Quill]: Draft ready. Run /pending to review, /approve-all {source} to save.", fg="green")
                    continue

                if cmd == '/draft-all':
                    source = split_leading_arg(arg_text, suffix=".pdf", known=known_sources)[0]
                    if not source:
                        click.secho("Usage: /draft-all <source_filename>", fg="red")
                        continue
                    checkpoints = note_agent.list_chapters(source)
                    if not checkpoints:
                        click.secho(note_agent.source_status(source), fg="yellow")
                        continue
                    click.secho(
                        f"[Quill]: Drafting {len(checkpoints)} chapter(s) from '{source}'... this may take a while. "
                        f"Each page is kept as soon as it's done (Ctrl+C stops; finished pages stay in /pending).",
                        fg="magenta"
                    )
                    drafted = []

                    def keep_page(page_name, content, source=source, drafted=drafted):
                        pending_store.add(page_name, content, source=source)
                        drafted.append(page_name)
                        click.secho(f"  drafted {page_name}", dim=True)

                    try:
                        note_agent.draft_multi_page_notes(source, on_page=keep_page, overview=True)
                    except KeyboardInterrupt:
                        click.secho(f"[Quill]: Stopped. {len(drafted)} page(s) kept in /pending.", fg="yellow")
                        continue
                    except Exception as e:
                        click.secho(f"Error: Drafting stopped: {e}. {len(drafted)} page(s) kept in /pending.", fg="red")
                        continue
                    click.secho(
                        f"[Quill]: {len(drafted)} page(s) drafted from '{source}'. Run /pending to review, "
                        f"/approve-all {source} to save all, or /discard-all {source} to drop them.",
                        fg="green"
                    )
                    continue

                if cmd == '/focus':
                    if not arg_text:
                        chat_focus = None
                        click.secho("[Quill]: Questions now search all documents and notes.", fg="green")
                        continue
                    source = split_leading_arg(arg_text, suffix=".pdf", known=known_sources)[0]
                    if source not in known_sources:
                        click.secho(f"No ingested PDF named '{source}'.", fg="red")
                        continue
                    chat_focus = source
                    click.secho(f"[Quill]: Questions now search only '{source}' (/focus to clear).", fg="green")
                    continue

            # --- Logic ---
            intent = determine_intent(user_input, llm)

            if intent == "EXIT":
                # Only the explicit exit words above end the session without
                # asking -- a classifier guess ("I'm done with chapter 3...")
                # must not silently drop the user out.
                if click.confirm("[Quill]: End this session?", default=True):
                    click.secho("[Quill]: Goodbye!", fg="cyan")
                    break
                intent = "CHAT"

            history.append(HumanMessage(content=user_input))
            session_manager.save_message(current_session_id, "human", user_input)

            response_content = ""

            if intent == "VERSION":
                plan = version_agent.plan(user_input, recent_note=recent_note)
                response_content = version_agent.execute(
                    plan, confirm=lambda question: click.confirm(question, default=False)
                )
                if plan.get("note"):
                    recent_note = plan["note"]

            elif intent == "EDIT":
                response_content, edited = handle_edit_request(user_input, note_agent, note_vcs, recent_note)
                recent_note = edited or recent_note

            elif intent == "NOTE":
                try:
                    note_path = note_agent.create_note(user_input, vcs=note_vcs)
                    recent_note = os.path.relpath(note_path, knowledge_dir).replace(os.sep, "/")
                    response_content = f"Note created at: {note_path}"
                    # Auto-rename session based on note topic if it's the first note
                    if len(history) < 5:
                        session_manager.rename_session(current_session_id, f"Note: {user_input[:20]}")
                except Exception as e:
                    response_content = f"Note creation failed: {e}"

            else:
                # Notes the user wrote or edited are searchable too, not just
                # the raw PDFs; re-embeds only notes that changed.
                note_agent.sync_note_index()
                scope = chat_focus or note_agent.detect_source(user_input)
                context = format_context(note_agent.search_docs(user_input, k=4, source=scope))

                if context:
                    role = SystemMessage(content=(
                        "You will take on the persona of a notetaker named Quill. "
                        "The following passages were retrieved from the user's documents and notes; each starts "
                        "with a label in square brackets saying where it came from. Use them to answer if relevant, "
                        "and cite the labels you rely on, e.g. [Book.pdf, p.12] or [note: Topic.md]. If they don't "
                        "cover the question, say so and answer from general knowledge.\n\n"
                        f"Passages:\n{context}"
                    ))
                else:
                    role = SystemMessage(content="You will take on the persona of a notetaker of the name Quill.")

                history_budget = input_char_budget(CHAT_REPLY_TOKENS) - len(role.content)
                response = llm.invoke([role] + trim_history(history, history_budget))
                response_content = response.content

            click.secho(f"[Quill]: {response_content}", fg="cyan")

            history.append(AIMessage(content=response_content))
            session_manager.save_message(current_session_id, "ai", response_content)

        except KeyboardInterrupt:
            click.secho("\nExiting...", fg="red")
            break
        except Exception as e:
            click.secho(f"Error: {e}", fg="red")

    if watcher:
        watcher.stop()
    watch_lock.release()
    snapshotter.stop()


if __name__ == "__main__":
    cli()
