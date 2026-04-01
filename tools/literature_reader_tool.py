import json
import re
import os
import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from tools.base import BaseTool
from config.settings import Settings

READING_PROMPT = """你是一个专业的学术文献阅读助手。请基于以下提供的文献内容，精准回答用户的问题。

【文献内容】：
{document_content}

【当前用户问题/任务】：
{user_query}

【严格要求】：
1. 你的所有回答必须以提供的【文献内容】为核心事实依据。
2. 如果用户的提问超出了该文献的范围，请直接说明“文献中未提及”，切勿自行捏造。
3. 提取关键信息时，语言保持客观、严谨的学术风格。"""


class LiteratureReaderTool(BaseTool):
    name = "literature_read"
    description = (
        "用于阅读和分析用户上传的本地文献（Markdown格式）。\n"
        "当用户要求总结文献、提取文献中的特定信息或基于文献进行分析时，必须调用此工具。\n"
        "输入参数必须是一个合法的 JSON 字符串，包含以下字段：\n"
        "- query: (必填) 针对文献提出的具体问题、提取要求或总结指令。"
    )

    def __init__(self, doc_path="uploaded_doc.md"):
        # 阅读文献需要稍微高一点的理解力，temperature 设为 0.1
        self.llm = Settings.get_llm(temperature=0.1)
        self.doc_path = doc_path

    def run(self, params: str) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
            query = args.get("query", "")

            if not os.path.exists(self.doc_path):
                return "执行失败：未找到用户上传的文献文件，请在聊天中提醒用户先在侧边栏上传 Markdown 文献。"

            with open(self.doc_path, "r", encoding="utf-8") as f:
                doc_content = f.read()

            # 使用 cl100k_base 编码器（大多数现代大模型的通用标准参考）
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(doc_content)

            # 设定最大 Token 限制为 115,000，为 Prompt 和输出留出 13k 的缓冲空间
            max_tokens = 115000

            if len(tokens) > max_tokens:
                # 截断 Token 并解码回字符串
                truncated_tokens = tokens[:max_tokens]
                doc_content = enc.decode(truncated_tokens) + "\n...[文献过长，已基于 128k Token 限制自动截断后文]..."

            prompt_template = ChatPromptTemplate.from_template(READING_PROMPT)
            chain = prompt_template | self.llm
            res = chain.invoke({
                "document_content": doc_content,
                "user_query": query
            })

            return res.content

        except json.JSONDecodeError:
            return "文献阅读工具出错: 参数解析失败，请确保输入的是合法的 JSON 字符串。"
        except Exception as e:
            return f"文献阅读工具出错: {str(e)}"