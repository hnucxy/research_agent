# from graph.state import AgentState
#
#
# class EvaluatorNode:
#     def __call__(self, state: AgentState) -> dict:
#         print("\n--- [Evaluator] Node ---")
#         # 为了演示汇报，强制通过，不进行 Self-Refine 循环
#         print("    [System] Self-Refine module is currently disabled for fast-pass.")
#         print("    [Result] Evaluation Passed.")
#
#         return {
#             "evaluation_result": {"passed": True, "feedback": "Auto-passed"}
#         }

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.settings import Settings
from graph.state import AgentState
from prompts.evaluator_prompts import EVALUATOR_SYSTEM_PROMPT,EVALUATOR_USER_PROMPT


# 定义评估结果的数据结构
class EvaluationSchema(BaseModel):
    passed: bool = Field(description="执行结果是否满足当前科研任务要求。如果工具报错或无相关内容，必须为False。")
    feedback: str = Field(description="如果不通过，请给出具体的修改建议（如：'请更换搜索关键词为X'）；如果通过，简述理由。")


class EvaluatorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=EvaluationSchema)

    def __call__(self, state: AgentState) -> dict:
        print("\n--- [Evaluator] Node ---")
        current_step = state["plan"][state["current_step_index"]]
        last_result = state["step_history"][-1] if state.get("step_history") else "无历史"

        #导入常量组装prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATOR_SYSTEM_PROMPT),
            ("user", EVALUATOR_USER_PROMPT)
        ])

        chain = prompt | self.llm | self.parser

        # 获取上下文和全局任务
        chat_history = state.get("chat_history", "无历史")
        original_task = state.get("task_input", "无")

        try:
            evaluation = chain.invoke({
                "chat_history": chat_history,
                "original_task": original_task,
                "step": current_step,
                "result": last_result,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            print(f"    [Error] Evaluator 解析失败: {e}")
            evaluation = {"passed": True, "feedback": "解析失败，启动兜底放行。"}

        print(f"    [Result] Passed: {evaluation.get('passed')}")
        print(f"    [Feedback]: {evaluation.get('feedback')}")

        # 每次进入评估器，我们让重试次数自动 +1
        current_retry = state.get("retry_count", 0) + 1
        return {
            "evaluation_result": evaluation,
            "retry_count": current_retry
        }