import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

class Settings:
    # 配置大语言模型API
    API_KEY = os.getenv("API_KEY")
    BASE_URL = "https://api.deepseek.com"
    # BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    # BASE_URL = "https://ai.zhansi.top/v1"
    MODEL_NAME = "deepseek-chat" # 或 gpt-5.2  或doubao-seed-2-0-lite-260215

    # 配置向量模型API
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
    EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL_NAME = "text-embedding-v4"

    @classmethod
    def get_llm(cls, temperature=0.0):
        """获取统一的 LLM 实例"""
        return ChatOpenAI(
            model=cls.MODEL_NAME,
            api_key=cls.API_KEY,
            base_url=cls.BASE_URL,
            temperature=temperature,
            # default_headers = {
            #     "User-Agent": "curl/7.68.0",
            #     "Connection": "close"
            # }
        )
    
    # 获取向量模型
    @classmethod
    def get_embeddings(cls):
        return OpenAIEmbeddings(
            model=cls.EMBEDDING_MODEL_NAME,
            api_key=cls.EMBEDDING_API_KEY,
            base_url=cls.EMBEDDING_BASE_URL,
            # 绕过本地检查
            check_embedding_ctx_length=False,
            # 遵守阿里云单词请求数量上限
            chunk_size=10,
            max_retries=3
        )