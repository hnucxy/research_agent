import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Settings:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com" # 假设的 Base URL
    MODEL_NAME = "deepseek-chat" # 或 deepseek-coder

    @classmethod
    def get_llm(cls, temperature=0.0):
        """获取统一的 LLM 实例"""
        return ChatOpenAI(
            model=cls.MODEL_NAME,
            api_key=cls.DEEPSEEK_API_KEY,
            base_url=cls.DEEPSEEK_BASE_URL,
            temperature=temperature
        )