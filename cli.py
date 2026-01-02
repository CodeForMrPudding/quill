import click
import os
from typing import Literal

# LangChain / Graph Imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import Agents
from src.agents.git_agent import GitAgent
from src.agents.note_agent import NoteAgent

# Import session manager
from src.helpers.session_manager import SessionManager

# --- Configuration ---
MODEL_NAME = "llama3.1"  # Or "gpt-4o" if using OpenAI
REPO_DIR = os.getcwd()  # Default to current directory
KNOWLEDGE_DIR = os.path.join(REPO_DIR, "obsidian_vault")


def print_logo():
    logo = r"""
        (`/\
        `=\/\   quill
         `=\/\    v1
          `=\/
             \
    """
    click.secho(logo, fg="cyan", bold=True)


def determine_intent(user_input: str, model) -> Literal["GIT", "NOTE", "CHAT"]:
    """
    Simple router to decide which agent to handle the request.
    """
    prompt = f"""
    Classify the following user input into one of these categories:
    1. 'GIT': If the user wants to save, commit, push, check status, or revert changes.
    2. 'NOTE': If the user wants to write a note, research a topic, or create content.
    3. 'CHAT': General conversation.

    User Input: "{user_input}"

    Return ONLY the category word.
    """
    response = model.invoke(prompt).content.strip().upper()
    if "GIT" in response:
        return "GIT"
    if "NOTE" in response:
        return "NOTE"
    return "CHAT"


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


@click.command()
@click.option('--repo', default=os.getcwd(), help='Path to the git repository/vault.')
def start(repo):
    """Quill: The Obsidian AI Assistant."""
    global REPO_DIR
    REPO_DIR = repo

    # 1. Initialize Components
    session_manager = SessionManager()

    try:
        llm = ChatOllama(model=MODEL_NAME, temperature=0)
    except Exception:
        click.secho("Error: Could not connect to Ollama.", fg="red")
        return

    try:
        git_agent = GitAgent(llm, REPO_DIR)
        note_agent = NoteAgent(llm, KNOWLEDGE_DIR)
    except Exception as e:
        click.secho(f"Agent Init Error: {e}", fg="red")
        return

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
            if user_input.lower() in ['exit', 'quit', 'q']:
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

            # --- Logic ---
            intent = determine_intent(user_input, llm)

            history.append(HumanMessage(content=user_input))
            session_manager.save_message(current_session_id, "human", user_input)

            response_content = ""

            if intent == "GIT":
                inputs = {"messages": history, "repo_dir": REPO_DIR}
                config = {"configurable": {"thread_id": current_session_id}}

                final_state = git_agent.graph.invoke(inputs, config=config)

                if final_state['messages']:
                    response_content = final_state['messages'][-1].content

            elif intent == "NOTE":
                inputs = {"messages": history, "note_topic": user_input, "directory": KNOWLEDGE_DIR}
                result = note_agent.graph.invoke(inputs)

                if result.get("final_filepath"):
                    response_content = f"Note created at: {result['final_filepath']}"
                    # Auto-rename session based on note topic if it's the first note
                    if len(history) < 5:
                        session_manager.rename_session(current_session_id, f"Note: {user_input[:20]}")
                else:
                    response_content = result["messages"][-1].content

            else:
                role = SystemMessage(content="You will take on the persona of a notetaker of the name Quill.")
                response = llm.invoke([role] + history)
                response_content = response.content

            click.secho(f"[Quill]: {response_content}", fg="cyan")

            history.append(AIMessage(content=response_content))
            session_manager.save_message(current_session_id, "ai", response_content)

        except KeyboardInterrupt:
            click.secho("\nExiting...", fg="red")
            break
        except Exception as e:
            click.secho(f"Error: {e}", fg="red")


if __name__ == "__main__":
    start()
