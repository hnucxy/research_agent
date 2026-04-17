import os
import re
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
    def get_llm(cls, temperature=0.0, streaming=False):
        """获取统一的 LLM 实例"""
        
        # 基础配置参数
        model_kwargs = {
            "model": cls.MODEL_NAME,
            "api_key": cls.API_KEY,
            "base_url": cls.BASE_URL,
            "temperature": temperature,
            "streaming": streaming,
            "extra_body": {
                "enable_thinking": False
            }
        }
        
        # 核心修复：如果开启了流式输出，强制要求 API 在最后一个 Chunk 中附带 Token 使用量
        if streaming:
            model_kwargs["stream_options"] = {"include_usage": True}

        return ChatOpenAI(**model_kwargs)
    
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

    @classmethod
    def get_collection_name(cls, base_name: str) -> str:
        """Namespace Chroma collections by embedding model to avoid dimension conflicts."""
        model_name = cls.EMBEDDING_MODEL_NAME or "default_embedding"
        suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name).strip("_").lower()
        if not suffix:
            suffix = "default_embedding"
        return f"{base_name}_{suffix}"
