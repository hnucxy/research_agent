import json
import re
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from tools.base import BaseTool
from config.settings import Settings
# 引入集中管理的 Prompt
from prompts.executor_prompts import ACADEMIC_WRITER_PROMPT


class AcademicWriterTool(BaseTool):
    name = "academic_write"
    description = (
        "专门用于撰写、润色或扩写具有严谨学术风格的文本。输入参数必须是一个合法的 JSON 字符串，包含以下字段：\n"
        "- topic: (必填) 需要撰写的具体学术主题或段落核心论点。\n"
        "- section: (可选) 目标段落类型（如 Abstract, Introduction, Methodology, Conclusion）。\n"
        "- reference_context: (必填) 撰写该段落所必须依赖的文献和事实上下文。"
    )

    def __init__(self):
        # 学术写作需要稍微多一点的创造力，temperature可以比规划器(0.1)稍高，如0.3
        self.llm = Settings.get_llm(temperature=0.3, streaming=True)

    def run(self, params: str) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
            topic = args.get("topic", "")
            section = args.get("section", "General Academic Text")
            reference_context = args.get("reference_context", "无")

            # 直接使用导入的常量组装 prompt
            prompt_template = ChatPromptTemplate.from_template(ACADEMIC_WRITER_PROMPT)
            chain = prompt_template | self.llm
            stream_gen = chain.stream({
                "section": section,
                "topic": topic,
                "reference_context": reference_context
            })
            
            container = st.session_state.get("current_stream_container")
            output = ""
            for chunk in stream_gen:
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    output += content
                    if container:
                        container.markdown(f"### ✍️ 正在撰写学术内容...\n\n{output} ▌")
            
            if container:
                container.markdown(f"### ✍️ 学术内容撰写完毕\n\n{output}")

            return output

        except Exception as e:
            return f"学术写作工具出错: {str(e)}"