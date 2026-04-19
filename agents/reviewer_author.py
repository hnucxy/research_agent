import json
import re
from typing import Literal

import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.logger import get_logger
from config.settings import Settings
from graph.state import ReviewerState
from utils.reviewer_support import build_diff_markup, classify_review_intent

logger = get_logger()


class InputParseSchema(BaseModel):
    user_prompt: str = Field(description="仅保留用户对稿件的修改要求。")
    draft_content: str = Field(description="需要修改的原稿正文。")


class InputParserNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=InputParseSchema)

    def __call__(self, state: ReviewerState) -> dict:
        logger.info("--- [Input Parser] Node ---")

        if state.get("draft_content"):
            intent_meta = classify_review_intent(state.get("user_prompt", ""))
            original_draft = state.get("original_draft_content") or state["draft_content"]
            return {
                "original_draft_content": original_draft,
                **intent_meta,
            }

        prompt = ChatPromptTemplate.from_template(
            """
你负责从混合输入中拆出“修改要求”和“原稿正文”。
只输出 JSON，格式必须满足：
{format_instructions}

输入：
{input_text}
"""
        )
        chain = prompt | self.llm | self.parser

        try:
            parsed = chain.invoke(
                {
                    "input_text": state["user_prompt"],
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
            draft_content = parsed.get("draft_content", "")
            user_prompt = parsed.get("user_prompt", state["user_prompt"])
        except Exception as exc:
            logger.error("解析输入失败: %s", exc)
            draft_content = state["user_prompt"]
            user_prompt = "请将原稿修改为更符合学术写作规范的版本。"

        intent_meta = classify_review_intent(user_prompt)
        return {
            "user_prompt": user_prompt,
            "draft_content": draft_content,
            "original_draft_content": draft_content,
            **intent_meta,
        }


class ReviewerSchema(BaseModel):
    status: Literal["pass", "revise", "reject"] = Field(
        description="审查结论。pass 为通过，revise 为需要修改，reject 为严重问题直接驳回。"
    )
    feedback: str = Field(description="具体审查意见或驳回理由。")


class ReviewerNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=ReviewerSchema)

    def __call__(self, state: ReviewerState) -> dict:
        logger.info("--- [Reviewer] Node ---")

        system_prompt = """
你是学术审稿人，负责判断当前稿件是否满足用户的修改要求。

当前任务意图：{task_intent}
当前审查模式：{review_mode}
当前审查重点：{review_focus}

审查规则：
1. 始终先检查稿件是否真正响应了用户的本轮诉求。
2. 若 `review_mode` 为 `relaxed`，不得因为缺少新增定量数据、实验指标或新增引用而直接 reject。
3. 若 `review_mode` 为 `relaxed`，重点关注学术表达、逻辑衔接、结构清晰度、语气是否正式、是否消除口语化。
4. 若 `review_mode` 为 `strict`，必须重点检查事实支撑、定量证据、引用依据、实验结论是否得到原稿支持。
5. 只有在 strict 模式下，且用户明确要求补充论据、数据、实验结论或引用，而原稿无法提供足够事实支撑时，才可以给出 reject。
6. 若问题可通过继续改写解决，优先给出 revise，并指出应如何改。
7. 若稿件满足要求，输出 pass。

输出必须严格符合：
{format_instructions}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    "【用户的修改要求】\n{prompt}\n\n【当前稿件】\n{draft}",
                ),
            ]
        )

        chain = prompt | self.llm | self.parser

        try:
            result = chain.invoke(
                {
                    "prompt": state["user_prompt"],
                    "draft": state["draft_content"],
                    "task_intent": state.get("task_intent", "general_revision"),
                    "review_mode": state.get("review_mode", "relaxed"),
                    "review_focus": state.get("review_focus", ""),
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
        except Exception as exc:
            logger.error("Reviewer 解析失败: %s", exc)
            result = {
                "status": "revise",
                "feedback": "系统解析出错，请 Author 再次尝试按审稿意见修订原稿。",
            }

        status = result.get("status")
        feedback = result.get("feedback")
        logger.info("    [Reviewer Decision]: %s", status)
        logger.info("    [Reviewer Feedback]: %s", feedback)

        update = {
            "status": status,
            "feedback": feedback,
            "draft_content": state["draft_content"],
            "diff_content": state.get("diff_content", ""),
            "final_diff_content": state.get("final_diff_content", ""),
        }
        if status == "pass":
            update["final_answer"] = state["draft_content"]
        return update


class AuthorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.3, streaming=True)

    def __call__(self, state: ReviewerState) -> dict:
        logger.info("--- [Author] Node ---")

        system_prompt = """
你是学术编辑，负责根据审稿意见修订当前稿件。

任务意图：{task_intent}
审查模式：{review_mode}
修订重点：{review_focus}

修订规则：
1. 只能使用原稿中已经出现的事实、概念和论证，不得编造数据、实验结果、参考文献或不存在的结论。
2. 若当前任务是语言润色、逻辑重构、去口语化或学术表达优化，应优先改进措辞、结构和连贯性，而不是强行补数据。
3. 若当前任务是补充论据或完善实验结论，但原稿缺少足够事实支撑，只能重写为更谨慎、更符合证据边界的表达，不能虚构补充内容。
4. 直接输出修订后的完整正文，不要解释，不要加标题，不要输出代码块。
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    "【最初的修改要求】\n{prompt}\n\n"
                    "【审稿反馈】\n{feedback}\n\n"
                    "【待修订稿件】\n{draft}\n\n"
                    "请输出修订后的完整正文：",
                ),
            ]
        )

        chain = prompt | self.llm
        container = st.session_state.get("current_stream_container")
        output = ""

        for chunk in chain.stream(
            {
                "prompt": state["user_prompt"],
                "feedback": state["feedback"],
                "draft": state["draft_content"],
                "task_intent": state.get("task_intent", "general_revision"),
                "review_mode": state.get("review_mode", "relaxed"),
                "review_focus": state.get("review_focus", ""),
            }
        ):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not content:
                continue
            output += content
            if container:
                container.markdown(f"### ✍️ Author 正在重构文稿...\n\n{output} ▌")

        output = re.sub(r"^```[a-zA-Z]*\n", "", output.strip())
        output = re.sub(r"\n```$", "", output)

        if container:
            container.markdown(f"### ✍️ Author 重构完毕\n\n{output}")

        logger.info("    [Author] 修改完成，新文稿长度: %d", len(output))
        logger.info("    [Author Result Preview]: %s...", output.replace("\n", " "))

        previous_draft = state.get("draft_content", "")
        original_draft = state.get("original_draft_content") or previous_draft
        incremental_diff = build_diff_markup(previous_draft, output)
        cumulative_diff = build_diff_markup(original_draft, output)

        return {
            "draft_content": output,
            "diff_content": incremental_diff,
            "final_diff_content": cumulative_diff,
            "retry_count": state.get("retry_count", 0) + 1,
        }
