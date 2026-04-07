# graph/graph_builder.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from agents.planner import PlannerNode
from agents.executor import ExecutorNode
from agents.evaluator import EvaluatorNode
from config.logger import get_logger

logger = get_logger()


def give_up_node(state: AgentState):
    """强制兜底节点：当多次重规划都失败时，直接向用户汇报，切断循环。"""
    logger.info("--- [Give Up] Node ---")
    msg = "经过多次检索与重新规划，未能找到完全符合您要求的文献。这可能是因为相关领域的具体研究较少，或者关键词过于苛刻。建议您放宽检索条件或更换核心关键词后重试。"

    # 直接覆盖 step_history，前端会将其作为最终输出抓取
    return {
        "step_history": [f"Step: 强制兜底汇报\nTool: generate\nResult: {msg}"],
        # 将当前步骤索引推至最大值，确保接下来 check_loop 会直接返回 "end"
        "current_step_index": len(state.get("plan", []))
    }

# 定义一个简单的状态更新函数，用来让步骤 +1
def step_updater(state: AgentState):
    return {
        "current_step_index": state["current_step_index"] + 1,
        "retry_count": 0,
        "evaluation_result": {} #清空上一次的评估
    }


def build_graph():
    workflow = StateGraph(AgentState)

    # 1. 添加节点
    workflow.add_node("planner", PlannerNode())
    workflow.add_node("executor", ExecutorNode())
    workflow.add_node("evaluator", EvaluatorNode())
    workflow.add_node("update_step", step_updater)  # 负责翻页
    workflow.add_node("give_up", give_up_node)  # 兜底节点

    # 2. 定义入口
    workflow.set_entry_point("planner")

    # 3. 定义普通边 (流程流转)
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evaluator")
    workflow.add_edge("give_up", "update_step")  # 兜底汇报完直接去更新步骤(触发结束)

    def check_evaluation(state: AgentState):
        result = state.get("evaluation_result", {})
        retry_count = state.get("retry_count", 0)
        replan_count = state.get("replan_count", 0)  # 获取全局重规划次数

        # 1. 检查是否通过
        passed_val = result.get("passed")
        is_passed = str(passed_val).lower() == "true" or passed_val is True

        if is_passed:
            return "pass"

        # 2. 梳理是否需要触发重规划 (needs_replan)
        needs_replan = False

        if retry_count >= 3:
            logger.warning("    [System] 局部检索重试达上限，准备触发全局重规划 (Replan)！")
            needs_replan = True
        elif result.get("action") == "replan":
            logger.warning("    [System] 评估专家主动要求重规划。")
            needs_replan = True

        # 3. 集中检查全局重规划次数 (防死循环)
        if needs_replan:
            if replan_count >= 1:
                logger.warning("    [System] 全局重规划次数达上限(确认无匹配文献)，强制结束调研并汇报失败！")
                return "give_up"  # 指向我们上次新增的 give_up_node
            else:
                logger.warning(f"    [System] 触发第 {replan_count + 1} 次 Re-plan，回退到Planner！")
                return "replan"

        # 4. 如果既没通过，也不需要重规划，就乖乖回去重试
        logger.warning("    [System] 评估未通过，触发 Self-Refine 回退 Executor重试。")
        return "retry"

    workflow.add_conditional_edges(
        "evaluator",
        check_evaluation,
        {
            "pass": "update_step",  # 成功则更新索引
            "retry": "executor",  # 失败则回到 executor 执行重试
            "replan": "planner",
            "give_up": "give_up"
        }
    )
    # 4. 定义条件边 (循环逻辑)
    # 决定是 "继续下一步" 还是 "结束"
    def check_loop(state: AgentState):
        current = state["current_step_index"]
        total_steps = len(state["plan"])

        # 因为 update_step 刚刚已经把 index + 1 了
        # 所以如果 current < total_steps，说明还有任务
        if current < total_steps:
            return "continue"
        else:
            return "end"



    workflow.add_conditional_edges(
        "update_step",
        check_loop,
        {
            "continue": "executor",  # 回到执行器，执行下一个 Step
            "end": END  # 结束
        }
    )



    return workflow.compile()


