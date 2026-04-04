import json
import re
import os
import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from prompts.executor_prompts import READING_PROMPT
from tools.base import BaseTool
from config.settings import Settings


class LiteratureReaderTool(BaseTool):
    name = "literature_read"
    description = (
        "用于阅读和分析用户上传的本地文献（Markdown格式）。\n"
        "当用户要求总结特定文献、提取指定文献中的信息或比较多篇文献时，必须调用此工具。\n"
        "输入参数必须是一个合法的 JSON 字符串，包含以下字段：\n"
        "- file_path: (必填) 要阅读的文献在系统中的绝对或相对路径。\n"
        "- query: (必填) 针对该文献提出的具体问题、提取要求或总结指令。"
    )

    def __init__(self):
        # 阅读文献需要稍微高一点的理解力，temperature 设为 0.1
        self.llm = Settings.get_llm(temperature=0.1)

    def run(self, params: str) -> str:
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
            res = chain.invoke({
                "document_content": doc_content,
                "user_query": query
            })

            return res.content

        except json.JSONDecodeError:
            return "文献阅读工具出错: 参数解析失败，请确保输入的是合法的 JSON 字符串。"
        except Exception as e:
            return f"文献阅读工具出错: {str(e)}"