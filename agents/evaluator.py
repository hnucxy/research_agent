
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.settings import Settings
from config.logger import get_logger
from graph.state import AgentState
from prompts.evaluator_prompts import EVALUATOR_SYSTEM_PROMPT,EVALUATOR_USER_PROMPT

logger = get_logger()

# 定义评估结果的数据结构
class EvaluationSchema(BaseModel):
    passed: bool = Field(description="执行结果是否满足当前科研任务要求。如果工具报错或无相关内容，必须为False。")
    feedback: str = Field(description="如果不通过，请给出具体的修改建议（如：'请更换搜索关键词为X'）；如果通过，简述理由。")
    # action 字段
    action: Literal["retry_step", "replan"] = Field(
        default="retry_step",
        description="如果passed为False，决定下一步策略：若是工具参数格式错或只需微调，选 'retry_step'；若是因为前置检索步骤失败导致当前完全没上下文可用，选 'replan' 触发全局重规划。"
    )


class EvaluatorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=EvaluationSchema)

    def __call__(self, state: AgentState) -> dict:
        # print("\n--- [Evaluator] Node ---")
        logger.info("--- [Evaluator] Node ---")
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
            # print(f"    [Error] Evaluator 解析失败: {e}")
            logger.exception("[Error] Evaluator 解析失败: %s", e)
            evaluation = {"passed": True, "feedback": "解析失败，启动兜底放行。", "action": "retry_step"}

        is_passed = evaluation.get('passed', False)
        raw_action = evaluation.get('action', 'retry_step')

        # 针对终端显示的格式化逻辑
        display_action = "正常放行(Proceed)" if is_passed else raw_action

        # print(f"    [Result] Passed: {is_passed}")
        # print(f"    [Action]: {display_action}")
        # print(f"    [Feedback]: {evaluation.get('feedback')}")

        logger.info("    [Result] Passed: %s", is_passed)
        logger.info("    [Action]: %s", display_action)
        logger.info("    [Feedback]: %s", evaluation.get('feedback'))

        current_retry = state.get("retry_count", 0) + 1
        return {
            "evaluation_result": evaluation,
            "retry_count": current_retry
        }