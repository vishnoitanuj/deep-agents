from langchain.agents import AgentState
from typing import NotRequired, Annotated
from typing import Literal
from typing_extensions import TypedDict

class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows

    Attributes:
        content: Short specific description of task
        status: Current state -  pending, in_progress, or completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]

    
def file_reducer(left, right):
    """Merge two file dictionaries, with the right side taking precedence.
    Used as a reducer function for the file field in agent state,a
    allowing incremental updates to the virtual file system.

    Args:
        left: Left side of dictionary (existing files)
        right: Ride side of dictionary (new/updated files)

    Returns:
        Merged dictionaries with right values overriding left values
    """

    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}

class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system
    
    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items from task planning and progress tracking
    - files: Virtula file system stored as dict mapping filenames to content
    """
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str,str]], file_reducer]