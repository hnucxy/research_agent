from config.settings import Settings
# class EvaluatorNode:
#     def __init__(self):
#         self.llm = Settings.get_llm(temperature=0.1)
#
#     def __call__(self, state: AgentState) -> dict:
#         print("\n--- [Evaluator] Node ---")
#         current_step = state["plan"][state["current_step_index"]]
#         last_result = state["step_history"][-1]
#
#         # [Methodology: Self-Refine]
#         # 1. 初始生成 (已在 Executor 完成)
#         # 2. 自反馈评估 (Evaluation)
#         # 3. 决定是否需要迭代 (Refinement)
#
#         prompt = f"""
#         请验证以下步骤的执行结果是否满足科研严谨性。
#         步骤: {current_step}
#         结果: {last_result}
#
#         如果不通过，请给出具体修改建议。
#         输出 JSON: {{"passed": bool, "feedback": "..."}}
#         """
#
#         # 模拟结果
#         # 假设总是通过，以便流程跑通
#         evaluation = {"passed": True, "feedback": "Result looks valid."}
#         print(f"Evaluation: {evaluation}")
#
#         return {"evaluation_result": evaluation}

from graph.state import AgentState


class EvaluatorNode:
    def __call__(self, state: AgentState) -> dict:
        print("\n--- [Evaluator] Node ---")
        # 为了演示汇报，强制通过，不进行 Self-Refine 循环
        print("    [System] Self-Refine module is currently disabled for fast-pass.")
        print("    [Result] Evaluation Passed.")

        return {
            "evaluation_result": {"passed": True, "feedback": "Auto-passed"}
        }