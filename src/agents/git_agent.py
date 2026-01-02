import subprocess
import shlex

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from src.states.agent_state import AgentState


class GitAgent:
    def __init__(self, model, repo_dir: str):
        self.model = model
        self.repo_dir = repo_dir
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):

        @tool
        def run_git_command(command: str):
            """
            Runs a raw git command.
            Input should be the arguments after 'git'.
            Example: 'commit -m "fix bug"' or 'status'.
            """
            # Security: Basic sanitization is recommended in production
            full_cmd = ["git"] + shlex.split(command)

            print(f"  [Executing]: git {command}")  # Visual feedback for CLI

            try:
                result = subprocess.run(
                    full_cmd,
                    cwd=self.repo_dir,
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    return f"STDOUT:\n{result.stdout}"
                else:
                    return f"STDERR:\n{result.stderr}"
            except Exception as e:
                return f"Execution Error: {str(e)}"

        self.tools = [run_git_command]
        self.model_with_tools = self.model.bind_tools(self.tools)

        def agent_node(state: AgentState):
            return {"messages": [self.model_with_tools.invoke(state["messages"])]}

        # Build Graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        # Interrupt before tools to allow CLI to ask user for confirmation
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["tools"]
        )
