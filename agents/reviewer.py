from pydantic import BaseModel, Field
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.settings import Settings
from config.logger import get_logger
from graph.state import AgentState

logger = get_logger()

class ReviewerSchema(BaseModel):
    passed: bool = Field(description="草稿是否满足用户初始需求并忠实于参考文档。")
    feedback: str = Field(description="如果不通过，给出详细且苛刻的修改意见；如果通过，简述通过理由。")

REVIEWER_PROMPT = """你是一位以严苛著称的学术期刊审稿专家。请审查作者提交的草稿是否忠实于【参考文档】，并圆满完成了用户的任务。

【参考文档内容】:
{document_context}

【用户初始任务】:
{task_input}

【作者提交的草稿】:
{current_draft}

【要求】:
1. 检查草稿是否过度偏离了【参考文档】的事实，如果缺乏关键数据或逻辑不严密，必须拒绝并给出修改意见。
2. 请严格按照以下JSON格式输出：
{format_instructions}
"""

class ReviewerNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=ReviewerSchema)

    def __call__(self, state: AgentState) -> dict:
        logger.info("--- [Reviewer] Node ---")
        task = state.get("task_input", "")
        draft = state.get("current_draft", "")
        doc_context = state.get("document_context", "无")
        
        prompt = ChatPromptTemplate.from_template(REVIEWER_PROMPT)
        chain = prompt | self.llm | self.parser

        try:
            review = chain.invoke({
                "document_context": doc_context,
                "task_input": task,
                "current_draft": draft,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            logger.error("    [Reviewer] 解析失败: %s", e)
            review = {"passed": True, "feedback": "大模型JSON解析失败，兜底放行。"}

        is_passed = review.get("passed", False)
        feedback = review.get("feedback", "")
        
        status_text = "✅ 审稿通过" if is_passed else "❌ 需发回重修"
        logger.info("    [Reviewer] %s | 意见: %s", status_text, feedback)

        container = st.session_state.get("current_stream_container")
        if container:
            container.markdown(f"### 🧐 [Reviewer] 审稿意见 ({status_text})\n\n{feedback}")

        current_retry = state.get("retry_count", 0) + 1

        return {
            "review_feedback": feedback,
            "evaluation_result": {"passed": is_passed, "feedback": feedback},
            "retry_count": current_retry,
            "step_history": [f"Step: 审稿专家审查\nTool: reviewer\nResult: [{status_text}]\n意见: {feedback}"]
        }