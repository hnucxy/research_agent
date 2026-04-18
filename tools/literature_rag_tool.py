import json
import os
import re

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings
from tools.base import BaseTool
from utils.exceptions import ToolExecutionError, VectorDBConnectionError
from utils.image_utils import build_markdown_image_gallery, encode_image_to_data_url


RAG_SYSTEM_PROMPT = """你是专业的学术问答助手。请严格基于检索到的文献片段和图片回答用户问题。
要求：
1. 所有结论都必须来自提供的文本或图片证据。
2. 如果现有材料不足以回答，就明确说明“文献中未提及”。
3. 回答保持客观、精炼，不要大段复述原文。"""


class LiteratureRagTool(BaseTool):
    name = "literature_rag_search"
    description = "在已上传文献中做细粒度检索，适合抽取指标、参数、实验现象和图表细节。"
    prompt_spec = (
        '输出 JSON，例如 {"query":"需要检索的具体问题","file_paths":["已选文献绝对路径"],"top_k":12}。'
        " `file_paths` 可选；如果提供，优先在这些文献内检索。"
    )

    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1, streaming=True)
        self.embeddings = Settings.get_embeddings()
        self.last_retrieved_images = []

    def run(self, params: str) -> str:
        self.last_retrieved_images = []

        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                "RAG 工具参数解析失败，请确保输入的是合法 JSON。"
            ) from exc

        query = (args.get("query") or "").strip()
        file_paths = args.get("file_paths") or []
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        top_k = args.get("top_k", 12)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 12
        top_k = max(5, min(top_k, 20))

        if not query:
            return "RAG 检索执行失败：query 参数不能为空。"

        try:
            vectorstore = Chroma(
                collection_name=Settings.get_collection_name("global_research_knowledge"),
                embedding_function=self.embeddings,
                persist_directory="./chroma_db",
            )
            docs = vectorstore.similarity_search(query, k=top_k)
        except Exception as db_err:
            raise VectorDBConnectionError(f"ChromaDB 连接或检索失败: {db_err}")

        docs = self._filter_docs_by_selected_files(docs, file_paths)
        if not docs:
            return "未检索到相关内容，可能是问题偏离现有文献，或文献尚未完成向量化。"

        content_blocks = []
        matched_image_paths = []
        text_context_blocks = []

        for doc in docs:
            metadata = doc.metadata or {}
            if metadata.get("type") == "image":
                image_path = metadata.get("image_path", "")
                context_chunk = metadata.get("context", "")
                if os.path.exists(image_path):
                    matched_image_paths.append(image_path)
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": f"【图像关联上下文】\n{context_chunk}",
                        }
                    )
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_image_to_data_url(image_path)},
                        }
                    )
                continue

            text_context_blocks.append(f"【文本片段】{doc.page_content}")

        if text_context_blocks:
            content_blocks.append(
                {
                    "type": "text",
                    "text": "【检索到的文本片段】\n" + "\n\n".join(text_context_blocks),
                }
            )

        content_blocks.append({"type": "text", "text": f"【用户问题】\n{query}"})

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=content_blocks),
        ]

        container = st.session_state.get("current_stream_container")
        output = ""
        for chunk in self.llm.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                output += content
                if container:
                    container.markdown(f"### 正在进行多模态片段分析...\n\n{output} ▌")

        self.last_retrieved_images = self._unique_paths(matched_image_paths)
        image_gallery = build_markdown_image_gallery(
            self.last_retrieved_images, title="**【检索命中的相关图表】**"
        )
        if image_gallery:
            output = f"{output}\n\n{image_gallery}"

        if container:
            container.markdown(f"### 片段分析完毕\n\n{output}")

        return output

    @staticmethod
    def _unique_paths(image_paths):
        unique_paths = []
        seen = set()
        for image_path in image_paths:
            if image_path and image_path not in seen and os.path.exists(image_path):
                seen.add(image_path)
                unique_paths.append(image_path)
        return unique_paths

    @staticmethod
    def _filter_docs_by_selected_files(docs, file_paths):
        if not docs or not file_paths:
            return docs

        normalized_paths = {os.path.normpath(path) for path in file_paths if path}
        selected_file_names = {
            os.path.basename(path) for path in normalized_paths if path
        }

        filtered_docs = []
        for doc in docs:
            metadata = doc.metadata or {}
            doc_file_path = os.path.normpath(metadata.get("file_path", ""))
            doc_file_name = metadata.get("file_name", "")
            if (
                doc_file_path in normalized_paths
                or doc_file_name in selected_file_names
            ):
                filtered_docs.append(doc)

        return filtered_docs or docs
