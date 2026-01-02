import os
import json
import sqlite3

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.states.agent_state import AgentState
from src.helpers.pdf_ingestion import ingest_pdfs


class NoteAgent:
    def __init__(self, model, text_directory: str):
        self.model = model
        self.directory = text_directory
        self.corpus_file = os.path.join(text_directory, "corpus.json")

        # Ensure Directory Exists
        os.makedirs(text_directory, exist_ok=True)
        self.init_db()

        # --- Ingestion Integration ---
        # If the corpus file doesn't exist, run the ingestion logic automatically
        if not os.path.exists(self.corpus_file):
            print(f"[NoteAgent] Corpus not found. initializing ingestion from {text_directory}...")
            ingest_pdfs(self.directory, self.corpus_file)

        # Initialize HippoRAG lazily
        self.hipporag = None
        self.setup_hipporag()

        self.graph = self._build_graph()

    @classmethod
    def init_db(cls):
        conn = sqlite3.connect("notes_db.sqlite")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS notes (name TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

    def setup_hipporag(self):
        """
        Imports and initializes HippoRAG only when needed.
        This prevents multiprocessing crash on Windows.
        """
        try:
            # MOVED IMPORT HERE
            from hipporag import HippoRAG
        except ImportError:
            print("Warning: HippoRAG not installed. Retrieval features disabled.")
            return

        # We assume corpus exists now because we ran ingest_pdfs in __init__
        if not os.path.exists(self.corpus_file):
            with open(self.corpus_file, 'w') as f:
                json.dump([], f)

        print("Initializing HippoRAG (this may take a moment)...")
        self.hipporag = HippoRAG(
            save_dir=os.path.join(self.directory, "hipporag_index"),
            llm_model_name="llama3.1",
            llm_base_url="http://localhost:11434/v1",
            embedding_model_name="sbert_all-MiniLM-L6-v2",
        )

        # Force indexing if needed
        if hasattr(self.hipporag, 'index'):
            try:
                self.hipporag.index([self.corpus_file])
            except Exception as e:
                print(f"HippoRAG indexing warning: {e}")

    def _build_graph(self):

        @tool
        def check_duplicate_and_reserve(name: str) -> str:
            """Check if filename exists. Returns 'Available' or 'Duplicate'."""
            conn = sqlite3.connect("notes_db.sqlite")
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO notes (name) VALUES (?)", (name,))
                conn.commit()
                return "Available"
            except sqlite3.IntegrityError:
                return "Duplicate"
            finally:
                conn.close()

        @tool
        def hipporag_retrieve(query: str) -> str:
            """Retrieves context from existing knowledge base."""
            if not self.hipporag:
                return "Retrieval unavailable."
            results = self.hipporag.retrieve([query])
            return "\n".join([r['text'] for r in results[:3]])

        self.tools = [check_duplicate_and_reserve, hipporag_retrieve]
        self.model_with_tools = self.model.bind_tools(self.tools)

        # --- Nodes ---
        def namer_node(state: AgentState):
            topic = state.get("note_topic", "General Note")
            msg = ([SystemMessage(content=f"Create a unique .md filename for topic: {topic}. Check if available.")] +
                   state["messages"])
            return {"messages": [self.model_with_tools.invoke(msg)]}

        def drafter_node(state: AgentState):
            prompt = (
                "The filename is secure. Write a detailed Obsidian markdown note. "
                "Use 'hipporag_retrieve' for context. Return ONLY the markdown content."
            )
            return {"messages": [self.model_with_tools.invoke(state["messages"] + [HumanMessage(content=prompt)])]}

        def validator_node(state: AgentState):
            last_msg = state["messages"][-1].content
            if len(last_msg) < 50:
                return {"validation_error": "Too short", "messages": [HumanMessage(content="Expand the note.")]}
            return {"note_content": last_msg, "validation_error": None}

        def writer_node(state: AgentState):
            content = state.get("note_content")
            filename = "generated_note.md"
            path = os.path.join(self.directory, filename)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return {"final_filepath": path, "messages": [HumanMessage(content=f"File written to {path}")]}

        # --- Graph ---
        workflow = StateGraph(AgentState)
        workflow.add_node("namer", namer_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("drafter", drafter_node)
        workflow.add_node("validator", validator_node)
        workflow.add_node("writer", writer_node)

        workflow.add_edge(START, "namer")
        workflow.add_conditional_edges("namer", tools_condition)
        workflow.add_edge("tools", "drafter")

        workflow.add_conditional_edges("drafter", tools_condition, {"tools": "tools", "__end__": "validator"})

        def check_valid(state):
            if state.get("validation_error"):
                return "drafter"
            return "writer"

        workflow.add_conditional_edges("validator", check_valid)
        workflow.add_edge("writer", END)

        return workflow.compile()
