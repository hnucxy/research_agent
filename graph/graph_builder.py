# graph/graph_builder.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from agents.planner import PlannerNode
from agents.executor import ExecutorNode
from agents.evaluator import EvaluatorNode


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

    # 2. 定义入口
    workflow.set_entry_point("planner")

    # 3. 定义普通边 (流程流转)
    # 规划 -> 执行
    workflow.add_edge("planner", "executor")
    # 执行 -> 评估
    workflow.add_edge("executor", "evaluator")
    # 评估 -> 更新步骤索引
    # workflow.add_edge("evaluator", "update_step")


    def check_evaluation(state: AgentState):
        result = state.get("evaluation_result", {})
        retry_count = state.get("retry_count", 0)
        replan_count = state.get("replan_count", 0) # 获取全局重规划次数

        # 局部放死循环(原地重试超过三次)
        if retry_count >= 3:
            print("    [System] 重试次数达上限，强制放行。")
            return "pass"

        if result.get("passed") is True:
            return "pass"
        elif result.get("action") == "replan":
            # 全局防死循环(重规划超过2次)
            if replan_count >=2:
                print("    [System] 全局重规划次数达上限(可能无相关文献)，强制停止检索循环！")
                return "pass"  # 强制放行，让下一步去总结
            else:
                print(f"    [System] 评估发现方向性错误，触发第 {replan_count + 1} 次 Re-plan，回退到Planner！")
                return "replan"
        else:
            print("    [System] 评估未通过，触发 Self-Refine 回退 Executor重试。")
            return "retry"

    workflow.add_conditional_edges(
        "evaluator",
        check_evaluation,
        {
            "pass": "update_step",  # 成功则更新索引
            "retry": "executor",  # 失败则回到 executor 执行重试
            "replan": "planner"
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





# from langgraph.graph import StateGraph, END
# from .state import AgentState
# from agents.planner import PlannerNode
# from agents.executor import ExecutorNode
# from agents.evaluator import EvaluatorNode
# from agents.memory_agent import MemoryNode  # 假设已实现
#
#
# def build_graph():
#     workflow = StateGraph(AgentState)
#
#     # 1. 添加节点
#     workflow.add_node("planner", PlannerNode())
#     workflow.add_node("executor", ExecutorNode())
#     workflow.add_node("evaluator", EvaluatorNode())
#     workflow.add_node("memory_saver", MemoryNode())  # 用于写入长期记忆
#
#     # 2. 定义边 (Edges)
#     workflow.set_entry_point("planner")
#
#     workflow.add_edge("planner", "executor")
#     workflow.add_edge("executor", "evaluator")
#
#     # 3. 定义条件边 (Conditional Edges)
#     def check_evaluation(state: AgentState):
#         result = state["evaluation_result"]
#         if result["passed"]:
#             # 检查是否还有剩余步骤
#             if state["current_step_index"] < len(state["plan"]) - 1:
#                 return "next_step"
#             else:
#                 return "finalize"
#         else:
#             return "retry"
#
#     def route_decision(state: AgentState):
#         decision = check_evaluation(state)
#         if decision == "next_step":
#             # 更新索引的逻辑通常在节点内或通过单独的工具节点处理，
#             # 这里简化为指向 executor (实际需先更新 index)
#             # 为了骨架简单，假设 MemoryNode 处理索引增加
#             return "memory_saver"
#         elif decision == "finalize":
#             return "memory_saver"
#         else:
#             # 修正后重试
#             return "executor"
#
#     workflow.add_conditional_edges(
#         "evaluator",
#         route_decision,
#         {
#             "executor": "executor",  # 重试
#             "memory_saver": "memory_saver"  # 通过，去保存记忆/下一步
#         }
#     )
#
#     # Memory 节点决定是结束还是继续下一个步骤
#     def memory_router(state: AgentState):
#         if state["current_step_index"] >= len(state["plan"]):
#             return "end"
#         else:
#             return "continue"
#
#     workflow.add_conditional_edges(
#         "memory_saver",
#         memory_router,
#         {
#             "continue": "executor",
#             "end": END
#         }
#     )
#
#     return workflow.compile()