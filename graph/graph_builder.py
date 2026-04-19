from langgraph.graph import END, StateGraph

from agents.evaluator import EvaluatorNode
from agents.executor import ExecutorNode
from agents.memory import MemoryNode
from agents.planner import PlannerNode
from agents.reviewer_author import AuthorNode, InputParserNode, ReviewerNode
from config.logger import get_logger
from graph.state import AgentState, ReviewerState

logger = get_logger()

SKIP_EVALUATOR_TOOLS = {"trigger_reviewer_loop"}


def give_up_node(state: AgentState):
    logger.info("--- [Give Up] Node ---")
    msg = (
        "经过多次检索与重新规划，仍未找到完全符合要求的文献或结果。"
        "建议放宽检索条件，或调整核心关键词后重试。"
    )
    return {
        "step_history": [f"Step: 强制兜底汇报\nTool: generate\nResult: {msg}"],
        "current_step_index": len(state.get("plan", [])),
    }


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", PlannerNode())
    workflow.add_node("executor", ExecutorNode())
    workflow.add_node("evaluator", EvaluatorNode())
    workflow.add_node("update_step", MemoryNode())
    workflow.add_node("give_up", give_up_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("give_up", "update_step")

    def check_post_execution(state: AgentState):
        current_index = state.get("current_step_index", 0)
        planned_tools = state.get("planned_tools", [])
        tool_name = planned_tools[current_index] if current_index < len(planned_tools) else ""
        if tool_name in SKIP_EVALUATOR_TOOLS:
            logger.info("    [System] 当前步骤由子图自审完成，跳过 Evaluator。")
            return "skip_evaluator"
        return "needs_evaluator"

    workflow.add_conditional_edges(
        "executor",
        check_post_execution,
        {
            "needs_evaluator": "evaluator",
            "skip_evaluator": "update_step",
        },
    )

    def check_evaluation(state: AgentState):
        result = state.get("evaluation_result", {})
        retry_count = state.get("retry_count", 0)
        replan_count = state.get("replan_count", 0)

        passed_val = result.get("passed")
        is_passed = str(passed_val).lower() == "true" or passed_val is True
        if is_passed:
            return "pass"

        needs_replan = False
        if retry_count >= 3:
            logger.warning("    [System] 局部重试达到上限，准备触发全局重规划。")
            needs_replan = True
        elif result.get("action") == "replan":
            logger.warning("    [System] Evaluator 主动要求重规划。")
            needs_replan = True

        if needs_replan:
            if replan_count >= 1:
                logger.warning("    [System] 全局重规划次数达到上限，转入兜底退出。")
                return "give_up"
            logger.warning("    [System] 触发新一轮规划。")
            return "replan"

        logger.warning("    [System] 当前步骤未通过，返回 Executor 重试。")
        return "retry"

    workflow.add_conditional_edges(
        "evaluator",
        check_evaluation,
        {
            "pass": "update_step",
            "retry": "executor",
            "replan": "planner",
            "give_up": "give_up",
        },
    )

    def check_loop(state: AgentState):
        current = state["current_step_index"]
        total_steps = len(state["plan"])
        return "continue" if current < total_steps else "end"

    workflow.add_conditional_edges(
        "update_step",
        check_loop,
        {
            "continue": "executor",
            "end": END,
        },
    )

    return workflow.compile()


def build_reviewer_graph():
    workflow = StateGraph(ReviewerState)

    workflow.add_node("input_parser", InputParserNode())
    workflow.add_node("reviewer", ReviewerNode())
    workflow.add_node("author", AuthorNode())

    workflow.set_entry_point("input_parser")
    workflow.add_edge("input_parser", "reviewer")
    workflow.add_edge("author", "reviewer")

    def check_reviewer_decision(state: ReviewerState):
        status = state.get("status")
        retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if status == "pass":
            logger.info("    [System] 审稿通过，结束循环。")
            return "end"
        if status == "reject":
            logger.info("    [System] 触发 reject，结束循环。")
            return "end"
        if retries >= max_retries:
            logger.warning("    [System] 达到最大修改次数，结束循环。")
            return "end"

        logger.info("    [System] 审稿未通过，交由 Author 修订。")
        return "revise"

    workflow.add_conditional_edges(
        "reviewer",
        check_reviewer_decision,
        {
            "end": END,
            "revise": "author",
        },
    )

    return workflow.compile()
