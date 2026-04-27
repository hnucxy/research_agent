import json
import re
import os
import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from prompts.executor_prompts import READING_PROMPT
from tools.base import BaseTool
from config.settings import Settings


class LiteratureReaderTool(BaseTool):
    name = "literature_read"
    description = (
        "用于阅读用户选中的本地 Markdown 文献，适合整篇总结、对比和核心贡献提炼。"
    )
    prompt_spec = (
        '输出 JSON：{"file_path":"文献路径","query":"阅读任务"}。'
        "file_path 必须来自已选文献清单。"
    )

    def __init__(self):
        # 阅读文献需要稍微高一点的理解力，temperature 设为 0.1
        self.llm = Settings.get_llm(temperature=0.1)

    def run(self, params: str, config: dict | None = None) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
            query = args.get("query", "")
            file_path = args.get("file_path", "")

            if not file_path or not os.path.exists(file_path):
                return f"执行失败：未找到文献文件（路径：{file_path}）。请检查 file_path 参数是否正确输入了清单中的路径。"

            with open(file_path, "r", encoding="utf-8") as f:
                doc_content = f.read()

            # 使用 cl100k_base 编码器
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(doc_content)

            # 设定最大 Token 限制为 115,000，为 Prompt 和输出留出缓冲空间
            max_tokens = 115000

            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                doc_content = enc.decode(truncated_tokens) + "\n...[文献过长，已基于 128k Token 限制自动截断后文]..."

            prompt_template = ChatPromptTemplate.from_template(READING_PROMPT)
            chain = prompt_template | self.llm
            stream_gen = chain.stream(
                {
                    "document_content": doc_content,
                    "user_query": query,
                },
                config=config,
            )
            
            container = st.session_state.get("current_stream_container")
            output = ""
            for chunk in stream_gen:
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    output += content
                    if container:
                        container.markdown(f"### 📖 正在深度阅读并分析文献...\n\n{output} ▌")
                        
            if container:
                container.markdown(f"### 📖 文献分析完毕\n\n{output}")

            return output

        except Exception as e:
            return f"文献阅读工具出错: {str(e)}"
