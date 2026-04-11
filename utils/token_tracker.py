from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class TokenTracker(BaseCallbackHandler):
    """
    自定义 LangChain 回调处理器：
    拦截 LLM 输出，将 Token 消耗直接写入传入的 session_state 字典中。
    """
    def __init__(self, usage_dict: dict):
        super().__init__()
        # 接收并绑定 Streamlit 的 session_state 字典
        self.usage_dict = usage_dict

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """当任何一个 LLM 节点运行结束并返回结果时自动触发"""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            
            # 直接更新绑定的字典
            self.usage_dict["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.usage_dict["completion_tokens"] += usage.get("completion_tokens", 0)
            self.usage_dict["successful_requests"] += 1