import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.multimodal_embedding import DashScopeMultiModalEmbeddings

load_dotenv()

class Settings:
    # 配置大语言模型API
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME")

    # 配置向量模型API
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

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

            # 关闭千问模型深度思考
            extra_body={
                "enable_thinking": False
            }
        )
    
    # # 获取向量模型
    # @classmethod
    # def get_embeddings(cls):
    #     return OpenAIEmbeddings(
    #         model=cls.EMBEDDING_MODEL_NAME,
    #         api_key=cls.EMBEDDING_API_KEY,
    #         base_url=cls.EMBEDDING_BASE_URL,
    #         # 绕过本地检查
    #         check_embedding_ctx_length=False,
    #         # 遵守阿里云单词请求数量上限
    #         chunk_size=10,
    #         max_retries=3
    #     )

    # 阿里多模态向量模型
    @classmethod
    def get_embeddings(cls):
        # 切换为阿里多模态向量模型
        return DashScopeMultiModalEmbeddings(
            api_key=cls.EMBEDDING_API_KEY,
            model=cls.EMBEDDING_MODEL_NAME
            )