from langgraph.graph import StateGraph, END
from .state import AgentState
from agents.planner import PlannerNode
from agents.executor import ExecutorNode
from agents.evaluator import EvaluatorNode
from agents.memory import MemoryNode
from agents.author import AuthorNode
from agents.reviewer import ReviewerNode
from config.logger import get_logger

logger = get_logger()

def give_up_node(state: AgentState):
    logger.info("--- [Give Up] Node ---")
    msg = "经过多次检索或多轮反复重修，未能达到理想结果。建议您放宽条件或提供更详细的指引后重试。"
    return {
        "step_history": [f"Step: 强制兜底汇报\nTool: generate\nResult: {msg}"],
        "current_step_index": len(state.get("plan", []))
    }

def build_graph():
    workflow = StateGraph(AgentState)

    # 添加所有节点
    workflow.add_node("planner", PlannerNode())
    workflow.add_node("executor", ExecutorNode())
    workflow.add_node("evaluator", EvaluatorNode())
    workflow.add_node("update_step", MemoryNode())  
    workflow.add_node("give_up", give_up_node)  
    workflow.add_node("author", AuthorNode())
    workflow.add_node("reviewer", ReviewerNode())

    # 动态路由入口
    def route_entry(state: AgentState):
        if state.get("current_function") == "d":
            return "author"
        return "planner"

    workflow.set_conditional_entry_point(
        route_entry,
        {"author": "author", "planner": "planner"}
    )

    # 单Agent流转边 (功能 a,b,c)
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evaluator")
    workflow.add_edge("give_up", "update_step")

    def check_evaluation(state: AgentState):
        result = state.get("evaluation_result", {})
        retry_count = state.get("retry_count", 0)
        replan_count = state.get("replan_count", 0)
        passed_val = result.get("passed")
        is_passed = str(passed_val).lower() == "true" or passed_val is True

        if is_passed: return "pass"
        needs_replan = (retry_count >= 3) or (result.get("action") == "replan")
        if needs_replan:
            if replan_count >= 1: return "give_up"
            else: return "replan"
        return "retry"

    workflow.add_conditional_edges(
        "evaluator",
        check_evaluation,
        {"pass": "update_step", "retry": "executor", "replan": "planner", "give_up": "give_up"}
    )

    def check_loop(state: AgentState):
        if state["current_step_index"] < len(state["plan"]): return "continue"
        else: return "end"

    workflow.add_conditional_edges(
        "update_step", check_loop, {"continue": "executor", "end": END}
    )

    # 多Agent辩论流转边 (功能 d)
    workflow.add_edge("author", "reviewer")

    def check_reviewer(state: AgentState):
        res = state.get("evaluation_result", {})
        if res.get("passed"): return "end"
        if state.get("retry_count", 0) >= 3:
            logger.warning("    [System] 审稿驳回次数达到上限，强制结束！")
            return "give_up"
        return "author"

    workflow.add_conditional_edges(
        "reviewer",
        check_reviewer,
        {"end": END, "author": "author", "give_up": "give_up"}
    )

    return workflow.compile()