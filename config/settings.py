import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Settings:
    API_KEY = os.getenv("API_KEY")

    BASE_URL = "https://api.deepseek.com"
    # BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    # BASE_URL = "https://ai.zhansi.top/v1"
    MODEL_NAME = "deepseek-chat" # 或 gpt-5.2  或doubao-seed-2-0-lite-260215


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