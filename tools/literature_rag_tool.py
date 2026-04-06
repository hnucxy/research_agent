import json
import re
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from tools.base import BaseTool
from config.settings import Settings

RAG_QA_PROMPT = """你是一个专业的学术问答助手。请基于以下检索到的文献片段，准确回答用户的问题。

【检索到的文献片段】：
{context}

【用户问题】：
{user_query}

【严格要求】：
1. 你的所有回答必须以提供的【文献片段】为事实依据。
2. 如果提供的片段中找不到答案，请明确说明“文献中未提及”，切勿自行捏造。
3. 请保持客观、严谨的学术风格。直接输出结论或提取出的具体细节，不要大段重复原文。"""

class LiteratureRagTool(BaseTool):
    name = "literature_rag_search"
    description = (
        "用于在已上传的文献中进行精准的事实检索、指标提取或特定方法查询。\n"
        "当用户询问文献里的具体细节（如参数、算法某一步骤、某个实验结果）时使用。\n"
        "输入参数必须是一个合法的 JSON 字符串，包含以下字段：\n"
        "- query: (必填) 用户的检索问题或需要提取的具体细节描述。"
    )

    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
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

            # 获取当前会话 ID 作为 collection_name
            chat_id = st.session_state.current_chat_id
            
            # 连接 Chroma 数据库
            vectorstore = Chroma(
                collection_name=chat_id,
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )

            # 执行相似度检索
            docs = vectorstore.similarity_search(query, k=5)
            if not docs:
                return "未检索到相关内容，可能是因为用户的问题偏离了现有文献，或文献尚未完成向量化。"

            # 组装 Context
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])

            # 交给 LLM 进行抽取总结
            prompt_template = ChatPromptTemplate.from_template(RAG_QA_PROMPT)
            chain = prompt_template | self.llm
            res = chain.invoke({
                "context": context,
                "user_query": query
            })

            return res.content

        except json.JSONDecodeError:
            return "RAG 工具出错: 参数解析失败，请确保输入的是合法的 JSON 字符串。"
        except Exception as e:
            return f"RAG 工具出错: {str(e)}"