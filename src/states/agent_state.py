from typing import Optional
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    Shared state for the Quill CLI system.
    """
    # Context
    repo_dir: str

    # Note creation specific
    note_topic: Optional[str]
    note_title: Optional[str]
    note_content: Optional[str]
    validation_error: Optional[str]
    final_filepath: Optional[str]

    # Tool specific
    last_tool_output: Optional[str]
