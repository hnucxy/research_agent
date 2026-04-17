from typing import List, Literal

from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.logger import get_logger
from config.settings import Settings
from prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from utils.exceptions import AgentPlanningError

logger = get_logger()

NonReadTools = Literal[
    "arxiv_search",
    "semantic_scholar_search",
    "academic_write",
    "generate",
]
ReadTools = Literal[
    "arxiv_search",
    "semantic_scholar_search",
    "academic_write",
    "generate",
    "literature_read",
    "literature_rag_search",
]


class NonReadPlanStep(BaseModel):
    task_description: str = Field(description="该步骤需要完成的具体任务描述。")
    tool_name: NonReadTools = Field(description="执行该步骤必须调用的工具。")


class ReadPlanStep(BaseModel):
    task_description: str = Field(description="该步骤需要完成的具体任务描述。")
    tool_name: ReadTools = Field(description="执行该步骤必须调用的工具。")


class NonReadExecutionPlan(BaseModel):
    steps: List[NonReadPlanStep] = Field(description="按执行顺序排列的步骤列表。")


class ReadExecutionPlan(BaseModel):
    steps: List[ReadPlanStep] = Field(description="按执行顺序排列的步骤列表。")


class PlannerNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.embeddings = Settings.get_embeddings()

    def __call__(self, state: dict) -> dict:
        logger.info("")
        logger.info("--- [Planner] Node ---")

        user_request = state.get("task_input", "")
        resource_context = state.get("resource_context", "无")
        current_func = state.get("current_function", "c")
        search_source = state.get("search_source", "arxiv")

        historical_experience = "无"
        if state.get("replan_count", 0) == 0:
            try:
                vectorstore = Chroma(
                    collection_name=Settings.get_collection_name("global_experience"),
                    embedding_function=self.embeddings,
                    persist_directory="./chroma_db",
                )
                docs_with_score = vectorstore.similarity_search_with_score(
                    user_request, k=1
                )
                if docs_with_score:
                    doc, score = docs_with_score[0]
                    if current_func != "a" and score < 0.3:
                        logger.info(
                            "    [Planner] 命中高相似度经验(score: %.2f), 分配 memo_output 直接输出。",
                            score,
                        )
                        return {
                            "plan": [f"【语义缓存直接输出】\n{doc.page_content}"],
                            "planned_tools": ["memo_output"],
                            "current_step_index": 0,
                            "retry_count": 0,
                            "replan_count": 0,
                        }
                    if score < 0.6:
                        historical_experience = doc.page_content
                        logger.info(
                            "    [Planner] 检索到相关历史经验(score: %.2f)。",
                            score,
                        )
            except Exception:
                logger.warning("    [Planner] 经验检索跳过或无相关经验。")

        strategy_hint = "无"
        if historical_experience != "无":
            if current_func == "a":
                strategy_hint = (
                    "文献检索具有时效性。历史经验只能辅助参考, 仍应优先规划当前外部检索步骤。"
                )
            elif current_func == "b":
                strategy_hint = (
                    "如果历史经验与当前写作任务高度相关, 可以复用其结构或结论, 再结合 `academic_write` 完成撰写。"
                )
            elif current_func == "c":
                strategy_hint = (
                    "如果历史阅读结论已经足够, 可直接使用 `generate`; 否则用 `literature_read` 或 `literature_rag_search` 进一步验证。"
                )
            else:
                strategy_hint = "只有在确认历史经验适用于当前任务后, 才可以使用。"

        if current_func in ["a", "b"]:
            parser = JsonOutputParser(pydantic_object=NonReadExecutionPlan)
            mode_prompt_addon = (
                "\n\n[动态约束] 当前为非阅读模式。"
                "禁止分配 `literature_read` 或 `literature_rag_search`。"
            )
        else:
            parser = JsonOutputParser(pydantic_object=ReadExecutionPlan)
            mode_prompt_addon = ""

        if current_func == "a":
            if search_source == "semantic_scholar":
                mode_prompt_addon += (
                    "\n[检索源约束] 前端选择了 Semantic Scholar。"
                    "当需要外部论文检索时, 优先使用 `semantic_scholar_search`, 不要切换到 `arxiv_search`。"
                )
            else:
                mode_prompt_addon += (
                    "\n[检索源约束] 前端选择了 arXiv。"
                    "当需要外部论文检索时, 使用 `arxiv_search`, 不要切换到 `semantic_scholar_search`。"
                )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PLANNER_SYSTEM_PROMPT + mode_prompt_addon),
                ("user", PLANNER_USER_PROMPT),
            ]
        )
        chain = prompt | self.llm | parser

        eval_res = state.get("evaluation_result", {})
        replan_count = state.get("replan_count", 0)
        if eval_res and not eval_res.get("passed", True):
            last_plan = state.get("plan", [])
            feedback = eval_res.get("feedback", "无反馈")
            step_history_str = (
                f"【注意】这是第 {replan_count} 次重新规划。\n"
                f"上一轮计划为: {last_plan}\n"
                f"执行失败, 评估反馈如下:\n\"{feedback}\"\n"
                "请严格吸收这条反馈, 调整检索词或策略, 不要重复处理无效结果。"
            )
        else:
            step_history_str = "无"

        try:
            plan_dict = chain.invoke(
                {
                    "chat_history": state.get("chat_history", "无"),
                    "task": user_request,
                    "resource_context": resource_context,
                    "historical_experience": historical_experience,
                    "strategy_hint": strategy_hint,
                    "step_history": step_history_str,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
        except Exception as exc:
            logger.error("[Error] Planner JSON 解析失败: %s", exc)
            raise AgentPlanningError(f"Planner 解析模型输出失败: {exc}")

        plan_descriptions = []
        assigned_tools = []
        for i, step in enumerate(plan_dict.get("steps", []), start=1):
            desc = step.get("task_description", "")
            tool = step.get("tool_name", "generate")
            logger.info("  步骤 %s: %s [工具: %s]", i, desc, tool)
            plan_descriptions.append(desc)
            assigned_tools.append(tool)

        current_replan_count = state.get("replan_count", 0)
        if step_history_str != "无":
            current_replan_count += 1

        return {
            "plan": plan_descriptions,
            "planned_tools": assigned_tools,
            "current_step_index": 0,
            "retry_count": 0,
            "replan_count": current_replan_count,
        }
