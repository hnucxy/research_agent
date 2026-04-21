from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.logger import get_logger
from config.settings import Settings
from graph.state import AgentState
from prompts.evaluator_prompts import EVALUATOR_SYSTEM_PROMPT, EVALUATOR_USER_PROMPT
from utils.failure_experience import build_failure_record, store_failure_record

logger = get_logger()


class EvaluationSchema(BaseModel):
    passed: bool = Field(
        description="执行结果是否满足当前科研任务要求。如果工具报错或无相关内容，必须为 False。"
    )
    feedback: str = Field(
        description="如果不通过，给出具体修改建议；如果通过，简述理由。"
    )
    action: Literal["retry_step", "replan"] = Field(
        default="retry_step",
        description=(
            "如果 passed 为 False，决定下一步策略：若只是参数格式或小幅调整，选 retry_step；"
            "若是前置检索失败导致当前完全没上下文可用，选 replan。"
        ),
    )


class EvaluatorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=EvaluationSchema)
        self.embeddings = Settings.get_embeddings()

    def __call__(self, state: AgentState) -> dict:
        logger.info("--- [Evaluator] Node ---")

        current_step = state["plan"][state["current_step_index"]]
        last_result = state["step_history"][-1] if state.get("step_history") else "无历史"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EVALUATOR_SYSTEM_PROMPT),
                ("user", EVALUATOR_USER_PROMPT),
            ]
        )
        chain = prompt | self.llm | self.parser

        chat_history = state.get("chat_history", "无历史")
        original_task = state.get("task_input", "无")
        resource_context = state.get("resource_context", "无")

        try:
            evaluation = chain.invoke(
                {
                    "chat_history": chat_history,
                    "original_task": original_task,
                    "resource_context": resource_context,
                    "step": current_step,
                    "result": last_result,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
        except Exception as exc:
            logger.exception("[Error] Evaluator 解析失败: %s", exc)
            evaluation = {
                "passed": True,
                "feedback": "解析失败，启动兜底放行。",
                "action": "retry_step",
            }

        is_passed = evaluation.get("passed", False)
        raw_action = evaluation.get("action", "retry_step")
        display_action = "正常放行(Proceed)" if is_passed else raw_action

        logger.info("    [Result] Passed: %s", is_passed)
        logger.info("    [Action]: %s", display_action)
        logger.info("    [Feedback]: %s", evaluation.get("feedback"))

        current_retry = state.get("retry_count", 0) + 1
        if not is_passed:
            # 失败结果写入失败经验库
            self._store_failure_memory(
                state=state,
                current_step=current_step,
                last_result=last_result,
                evaluation=evaluation,
                retry_count=current_retry,
            )

        return {
            "evaluation_result": evaluation,
            "retry_count": current_retry,
        }

    def _store_failure_memory(
        self,
        state: AgentState,
        current_step: str,
        last_result: str,
        evaluation: dict,
        retry_count: int,
    ) -> None:
        tool_name = state["planned_tools"][state["current_step_index"]]
        record = build_failure_record(
            task_input=state.get("task_input", ""),
            current_step=current_step,
            tool_name=tool_name,
            step_result=last_result,
            feedback=evaluation.get("feedback", ""),
            retry_count=retry_count,
            evaluator_action=evaluation.get("action", "retry_step"),
            chat_id=state.get("current_chat_id", ""),
            current_function=state.get("current_function", ""),
        )
        if not record:
            return

        try:
            store_failure_record(record, embeddings=self.embeddings)
            logger.info("    [Evaluator] 已写入失败经验库。")
        except Exception as exc:
            logger.warning("    [Evaluator] 写入失败经验库失败: %s", exc)
