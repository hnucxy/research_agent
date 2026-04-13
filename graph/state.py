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

    # 存放上传文件的全文内容，供 Author 和 Reviewer 直接阅读
    document_context: str
    # 当前撰写的草稿
    current_draft: str
    # 审稿专家的反馈意见
    review_feedback: str

    # [Self-Refine] 评估结果
    evaluation_result: Dict[str, Any]  # e.g. {"passed": bool, "feedback": str}

    # 最终输出
    final_answer: str