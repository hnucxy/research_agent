from typing import TypedDict, List, Annotated, Dict, Any
import operator


class AgentState(TypedDict):
    """
    全局状态对象，在所有节点间传递。
    """
    # 当前所属的功能模块代码 (如 'a':检索, 'b':撰写, 'c':阅读, 'd':功能四)
    current_function: str

    # 原始用户输入
    task_input: str

    # 结构化的外部资源提示（如勾选文献、图表）
    resource_context: str

    # 用户选中的图表路径（如有）
    selected_image_path: str

    # 存放多轮对话历史，初始化时前端传入
    chat_history: str

    # [CoT] 任务规划列表 (结构化)
    plan: List[str]

    # Planner 为每个步骤分配的工具名称列表。长度必须与 plan 保持一致。
    planned_tools: List[str]

    #[Self-Refine] 当前步骤的重试次数，防止无限重试死循环
    retry_count: int

    # 全局重新规划次数
    replan_count: int

    # 当前正在执行的步骤索引
    current_step_index: int

    # [ReAct] 当前步骤的执行历史 (Tool calls & outputs)
    step_history: Annotated[List[Any], operator.add]

    # [Self-Refine] 评估结果
    evaluation_result: Dict[str, Any]  # e.g. {"passed": bool, "feedback": str}

    # 最终输出
    final_answer: str



class ReviewerState(TypedDict):
    """
    功能四 (Author-Reviewer) 专属状态对象
    """
    current_function: str
    user_prompt: str         # 用户的真实修改意图/提示词
    draft_content: str       # 当前的草稿/原稿内容
    feedback: str            # Reviewer 给出的修改意见
    retry_count: int         # 当前迭代次数
    max_retries: int         # 最大迭代次数上限
    status: str              # 状态："pass", "revise", "reject"
    chat_history: str
    final_answer: str        # 最终输出内容
