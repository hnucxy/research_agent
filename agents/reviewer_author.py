import json
import re
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.settings import Settings
from config.logger import get_logger
from graph.state import ReviewerState
import streamlit as st

logger = get_logger()

# 解析节点分离Prompt和原稿
class InputParserNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)

    def __call__(self, state: ReviewerState) -> dict:
        logger.info("--- [Input Parser] Node ---")
        # 如果已经通过前端文件上传读取到了原稿，则跳过分离
        if state.get("draft_content"):
            return {}

        # 否则，使用大模型从用户的纯文本输入中分离要求和原稿
        prompt = ChatPromptTemplate.from_template("""
        将下列混合输入拆成“修改要求”和“原稿正文”，只输出 JSON：
        {{"user_prompt": "修改要求", "draft_content": "原稿内容"}}

        输入：
        {input_text}
        """)
        chain = prompt | self.llm
        try:
            res = chain.invoke({"input_text": state["user_prompt"]})
            content = res.content.strip()
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content)
            parsed = json.loads(content)
            return {
                "user_prompt": parsed.get("user_prompt", state["user_prompt"]),
                "draft_content": parsed.get("draft_content", "")
            }
        except Exception as e:
            logger.error(f"解析输入失败: {e}")
            return {"draft_content": state["user_prompt"]} # 兜底

# 审稿人节点
class ReviewerSchema(BaseModel):
    status: Literal["pass", "revise", "reject"] = Field(description="审查结论。pass为通过，revise为需修改，reject为严重缺陷直接驳回。")
    feedback: str = Field(description="具体的审查意见或驳回理由。")

class ReviewerNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=ReviewerSchema)

    def __call__(self, state: ReviewerState) -> dict:
        logger.info("--- [Reviewer] Node ---")
        
        system_prompt = """你是学术审稿人，负责判断当前原稿是否满足用户修改要求。

        规则：
        1. 重点检查逻辑结构、学术规范、流畅度，以及是否响应了用户要求。
        2. 只有当用户明确要求补充数据、指标或引用，而原稿完全没有事实支撑时，才判定为 `reject`。
        3. 如果只是润色、重构、提升语气或消除口语化，不得因缺少量化数据直接驳回，应给出 `revise`。
        4. 完全满足要求时判定为 `pass`。

        输出必须符合：
        {format_instructions}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【用户的修改要求】: {prompt}\n\n【当前原稿】:\n{draft}")
        ])
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "prompt": state["user_prompt"],
                "draft": state["draft_content"],
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            logger.error(f"Reviewer解析失败: {e}")
            result = {"status": "revise", "feedback": "系统解析出错，请 Author 再次尝试润色原稿。"}

        logger.info(f"    [Reviewer Decision]: {result.get('status')}")
        logger.info(f"    [Reviewer Feedback]: {result.get('feedback')}")
        return {
            "status": result.get("status"), 
            "feedback": result.get("feedback"),
            # 显式传回方便 UI 抓取
            "draft_content": state["draft_content"] 
        }

# 作者节点
class AuthorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.3, streaming=True)

    def __call__(self, state: ReviewerState) -> dict:
        logger.info("--- [Author] Node ---")
        
        system_prompt = """你是学术编辑，负责按审稿意见重写当前原稿。

        规则：
        1. 只能使用原稿中已有的事实、概念和逻辑，不得捏造数据、实验或引用。
        2. 优先改善逻辑衔接、学术语气和表达严谨性。
        3. 直接输出修改后的正文，不要附加解释或代码块。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【最初的修改要求】: {prompt}\n\n【审稿人的反馈意见】: {feedback}\n\n【需要修改的当前原稿】:\n{draft}\n\n请输出重构修改后的全新正文版本：")
        ])
        
        chain = prompt | self.llm
        # 适配流式输出到前端容器
        container = st.session_state.get("current_stream_container")
        output = ""
        for chunk in chain.stream({
            "prompt": state["user_prompt"],
            "feedback": state["feedback"],
            "draft": state["draft_content"]
        }):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                output += content
                if container:
                    container.markdown(f"### ✍️ Author 正在重构文稿...\n\n{output} ▌")
        
        if container:
            container.markdown(f"### ✍️ Author 重构完毕\n\n{output}")

        logger.info("    [Author] 修改完成，新文稿长度: %d", len(output))
        logger.info("    [Author Result Preview]: %s...", output.replace("\n", " "))

        new_retry = state.get("retry_count", 0) + 1
        return {
            "draft_content": output,
            "retry_count": new_retry
        }
