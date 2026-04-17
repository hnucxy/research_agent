import json
import re
import os
import base64
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from tools.base import BaseTool
from config.settings import Settings
from utils.exceptions import VectorDBConnectionError, ToolExecutionError

def encode_image(image_path):
    """将本地图片文件转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

RAG_SYSTEM_PROMPT = """你是一个专业的学术问答助手。请基于以下检索到的文献片段和图片，准确回答用户的问题。
【严格要求】：
1. 你的所有回答必须以提供的文献片段和图片为事实依据。
2. 如果提供的片段或图片中找不到答案，请明确说明“文献中未提及”，切勿自行捏造。
3. 请保持客观、严谨的学术风格。直接输出结论或提取出的具体细节，不要大段重复原文。"""

class LiteratureRagTool(BaseTool):
    name = "literature_rag_search"
    description = (
        "用于在已上传文献中做细粒度检索，适合指标、参数、实验现象和图表细节提取。"
    )
    prompt_spec = '输出 JSON：{"query":"需要检索的具体问题或细节"}。'

    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1, streaming=True)
        self.embeddings = Settings.get_embeddings()

    def run(self, params: str) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
            query = args.get("query", "")

            if not query:
                return "RAG 检索执行失败：query 参数不能为空。"

            # 连接全局 Chroma 数据库
            try:
                vectorstore = Chroma(
                    collection_name=Settings.get_collection_name("global_research_knowledge"),
                    embedding_function=self.embeddings,
                    persist_directory="./chroma_db"
                )
                docs = vectorstore.similarity_search(query, k=5)
            except Exception as db_err:
                # 包装为自定义的向量库异常抛出
                raise VectorDBConnectionError(f"ChromaDB 连接或检索失败: {db_err}")

            # 执行相似度检索
            docs = vectorstore.similarity_search(query, k=5)
            if not docs:
                return "未检索到相关内容，可能是因为用户的问题偏离了现有文献，或文献尚未完成向量化。"

            content_blocks = []
            image_markdowns = []
            text_context = ""

            for doc in docs:
                if doc.metadata.get("type") == "image":
                    img_path = doc.metadata.get("image_path", "")
                    context_chunk = doc.metadata.get("context", "")
                    
                    if os.path.exists(img_path):
                        # 1. 编码给大模型使用
                        base64_image = encode_image(img_path)
                        img_ext = img_path.split('.')[-1].lower()
                        mime_type = "image/png"
                        if img_ext in ['jpg', 'jpeg']: mime_type = "image/jpeg"
                        elif img_ext == 'webp': mime_type = "image/webp"
                        
                        # 添加到用户的多模态 Message
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                        })
                        content_blocks.append({
                            "type": "text", 
                            "text": f"【图文匹配项】关联图片对应的文献上下文：\n{context_chunk}"
                        })
                        
                        # 2. 收集图片本地路径供前端显示（避免直接返回 base64 导致系统记忆 Token 爆炸）
                        md_img = f"![检索到的文献图表]({img_path})"
                        if md_img not in image_markdowns:
                            image_markdowns.append(md_img)
                else:
                    text_context += f"【文本片段】：{doc.page_content}\n\n"

            # 组装纯文本部分
            if text_context:
                content_blocks.append({"type": "text", "text": f"【检索到的文本片段】:\n{text_context}"})
            
            # 最后加入用户问题
            content_blocks.append({"type": "text", "text": f"【用户问题】:\n{query}"})

            # 构建完整的对话消息
            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=content_blocks)
            ]

            container = st.session_state.get("current_stream_container")
            output = ""
            # 交给具备视觉能力的 LLM 进行推理
            for chunk in self.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    output += content
                    if container:
                        container.markdown(f"### 🔎 正在进行多模态片段分析...\n\n{output} ▌")

            if image_markdowns:
                output += "\n\n**【检索到的相关图表】**:\n"
                output += "\n".join(image_markdowns)
                
            if container:
                container.markdown(f"### 🔎 片段分析完毕\n\n{output}")

            return output

        except json.JSONDecodeError:
            raise ToolExecutionError("RAG 工具出错: 参数解析失败，请确保输入的是合法的 JSON 字符串。")
        except VectorDBConnectionError as ve:
            return str(ve) # 让外层 Executor 捕获或直接返回给大模型
        except Exception as e:
            raise ToolExecutionError(f"RAG 工具内部逻辑出错: {str(e)}")
