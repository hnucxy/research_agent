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
        用户输入了一段文本，里面混合了“修改要求（Prompt）”和“论文原稿内容（Draft）”。
        请你将它们分离，并严格输出 JSON 格式：
        {{"user_prompt": "提取出的修改要求", "draft_content": "提取出的原稿内容"}}
        
        用户输入：
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
        
        system_prompt = """你是一个严苛的顶级学术审稿人与逻辑架构师。你的任务是根据【用户的修改要求】严格审查【当前原稿】。

        【审查重点与边界（差异化功能定位）】：
        1. 你的核心任务是评估文本的逻辑结构、学术规范性、流畅度以及是否满足用户的具体指令。你不是在盲目要求补充实验数据！
        2. 【防幻觉拒稿机制（精准触发）】：只有当用户的【修改要求】中**明确提出**需要添加具体的性能指标、实验数据或文献引用，而【当前原稿】中却完全没有提供这些支撑信息时，你才必须判定为 "reject"（驳回），告知用户缺乏事实基础无法凭空捏造。
        3. 【纯润色与重构放行】：如果用户的要求仅仅是“润色”、“重构逻辑”、“提升学术语气”、“消除口语化”等结构或风格调整，即便原稿只有定性的概念陈述而无定量数据，你也**绝对不能**因此驳回！你应该指出原稿在逻辑连贯性和学术用语上的不足，判定为 "revise"。
        4. 如果当前原稿已经完美契合了用户的要求，判定为 "pass"。

        按照以下格式输出：
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
        
        system_prompt = """你是一位资深的学术编辑与重写专家。你的任务是根据【审稿人的反馈意见】和【最初的修改要求】，对【当前原稿】进行深度的打磨与逻辑重构。

        【执行规范】：
        1. 忠于事实边界：你只能使用【当前原稿】中提供的事实、概念和逻辑思路。绝不能为了迎合审查而凭空捏造实验结果、数值或外部文献。
        2. 学术升华：如果原稿口语化严重或逻辑跳跃，请运用专业的学术连词、严密的推导逻辑（如“因果递进”、“对比分析”）将其转化为顶级学术期刊级别的表达。
        3. 纯净输出：请直接输出修改后的全新正文文本，绝对不要包含任何“好的”、“这是修改后的版本”、“根据意见修改如下”等解释性废话。不要输出多余的 Markdown 代码块标记。"""
        
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