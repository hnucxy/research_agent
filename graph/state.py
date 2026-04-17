import operator
from typing import Annotated, Any, Dict, List, TypedDict


class AgentState(TypedDict):
    current_function: str
    task_input: str
    resource_context: str
    selected_image_path: str
    chat_history: str

    search_source: str
    semantic_sort_by: str

    plan: List[str]
    planned_tools: List[str]
    retry_count: int
    replan_count: int
    current_step_index: int
    step_history: Annotated[List[Any], operator.add]
    evaluation_result: Dict[str, Any]
    final_answer: str


class ReviewerState(TypedDict):
    current_function: str
    user_prompt: str
    draft_content: str
    feedback: str
    retry_count: int
    max_retries: int
    status: str
    chat_history: str
    final_answer: str
