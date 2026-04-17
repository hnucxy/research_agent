import os
import re

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return False

load_dotenv()


class Settings:
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME")

    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    SEMANTIC_SCHOLAR_BASE_URL = os.getenv(
        "SEMANTIC_SCHOLAR_BASE_URL",
        "https://s2api.ominiai.cn/s2/graph/v1/paper/search",
    )

    @classmethod
    def get_llm(cls, temperature=0.0, streaming=False):
        """返回统一的 LLM 客户端。"""
        from langchain_openai import ChatOpenAI

        model_kwargs = {
            "model": cls.MODEL_NAME,
            "api_key": cls.API_KEY,
            "base_url": cls.BASE_URL,
            "temperature": temperature,
            "streaming": streaming,
            "extra_body": {
                "enable_thinking": False,
            },
        }

        if streaming:
            model_kwargs["stream_options"] = {"include_usage": True}

        return ChatOpenAI(**model_kwargs)

    @classmethod
    def get_embeddings(cls):
        from utils.multimodal_embedding import DashScopeMultiModalEmbeddings

        return DashScopeMultiModalEmbeddings(
            api_key=cls.EMBEDDING_API_KEY,
            model=cls.EMBEDDING_MODEL_NAME,
        )

    @classmethod
    def get_collection_name(cls, base_name: str) -> str:
        """按 embedding 模型为 Chroma collection 添加命名空间, 避免维度冲突。"""
        model_name = cls.EMBEDDING_MODEL_NAME or "default_embedding"
        suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name).strip("_").lower()
        if not suffix:
            suffix = "default_embedding"
        return f"{base_name}_{suffix}"
