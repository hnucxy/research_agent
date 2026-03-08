from typing import TypedDict, List, Annotated, Dict, Any
import operator


class AgentState(TypedDict):
    """
    全局状态对象，在所有节点间传递。
    """
    # 原始用户输入
    task_input: str

    # [Memory] 长期记忆上下文 (RAG 检索结果)
    long_term_context: str

    # [CoT] 任务规划列表 (结构化)
    plan: List[str]

    # [新增] Planner 为每个步骤分配的工具名称列表。长度必须与 plan 保持一致。
    planned_tools: List[str]

    #[Self-Refine] 当前步骤的重试次数，防止无限重试死循环
    retry_count: int

    # 当前正在执行的步骤索引
    current_step_index: int

    # [ReAct] 当前步骤的执行历史 (Tool calls & outputs)
    step_history: Annotated[List[Any], operator.add]

    # [Self-Refine] 评估结果
    evaluation_result: Dict[str, Any]  # e.g. {"passed": bool, "feedback": str}

    # 最终输出
    final_answer: str