import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from config.settings import Settings
from config.logger import get_logger
from graph.state import AgentState

logger = get_logger()

AUTHOR_PROMPT = """你是一位严谨的学术论文作者。你的任务是基于【参考文档】的内容，结合用户的需求及审稿专家的反馈进行撰写。

【参考文档内容】:
{document_context}

【用户初始任务】:
{task_input}

【审稿专家反馈】(若为“无”说明是首次撰写):
{review_feedback}

【要求】:
1. 必须严格参考【参考文档】中的核心观点或事实。
2. 如果有审稿人反馈，请务必针对反馈中的意见进行逐条修改和提升。
3. 请直接输出学术文本，不要有任何寒暄、废话或解释性前缀。
"""

class AuthorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.3, streaming=True)

    def __call__(self, state: AgentState) -> dict:
        logger.info("--- [Author] Node ---")
        task = state.get("task_input", "")
        feedback = state.get("review_feedback", "无")
        doc_context = state.get("document_context", "无")

        prompt = ChatPromptTemplate.from_template(AUTHOR_PROMPT)
        chain = prompt | self.llm

        container = st.session_state.get("current_stream_container")
        output = ""

        try:
            for chunk in chain.stream({
                "document_context": doc_context,
                "task_input": task,
                "review_feedback": feedback
            }):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    output += content
                    if container:
                        container.markdown(f"### ✍️ [Author] 正在撰写/修改草稿...\n\n{output} ▌")
            
            if container:
                container.markdown(f"### ✍️ [Author] 草稿撰写完毕\n\n{output}")

            logger.info("Author 完成了草稿撰写:%s", output)
                
        except Exception as e:
            logger.error("    [Author] 执行失败: %s", e)
            output = f"【撰写失败】: {str(e)}"

        return {
            "current_draft": output,
            "step_history": [f"Step: 作者撰写草稿\nTool: author\nResult: {output}"]
        }