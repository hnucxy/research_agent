import operator
from typing import Annotated, Any, Dict, List, TypedDict


class AgentState(TypedDict):
    current_function: str  # 当前执行的功能模块名称
    current_chat_id: str  # 当前会话ID
    task_input: str  # 用户输入的内容
    resource_context: str  # 已选资源或检索资源的上下文内容
    selected_image_path: str  # 用户当前选中的图片路径
    retrieved_image_paths: List[str]  # 检索流程返回的相关图片路径列表
    chat_history: str  # 当前会话的历史对话内容

    search_source: str  # 文献搜索使用的数据源名称
    semantic_sort_by: str  # Semantic Scholar 搜索结果的排序方式
    semantic_year_filter: str  # Semantic Scholar 搜索结果的年份限制

    plan: List[str]  # Agent 生成的任务执行计划
    planned_tools: List[str]  # 计划中预计调用的工具名称列表
    retry_count: int  # 当前步骤或任务的重试次数
    replan_count: int  # 重新规划任务的次数
    current_step_index: int  # 当前正在执行的计划步骤索引
    step_history: Annotated[List[Any], operator.add]  # 各步骤执行结果的累积历史
    evaluation_result: Dict[str, Any]  # 对执行结果的评估信息
    final_answer: str  # 返回给用户的最终回答


class ReviewerState(TypedDict):
    current_function: str
    user_prompt: str
    draft_content: str
    original_draft_content: str
    feedback: str
    task_intent: str
    review_mode: str
    review_focus: str
    diff_content: str
    final_diff_content: str
    retry_count: int
    max_retries: int
    status: str
    chat_history: str
    final_answer: str
