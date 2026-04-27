from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _normalize_usage(usage: dict | None) -> tuple[int, int]:
    if not usage:
        return 0, 0
    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_token_count")
        or 0
    )
    return int(prompt_tokens or 0), int(completion_tokens or 0)


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
        prompt_tokens = 0
        completion_tokens = 0

        if response.llm_output and "token_usage" in response.llm_output:
            prompt_tokens, completion_tokens = _normalize_usage(
                response.llm_output["token_usage"]
            )

        if not (prompt_tokens or completion_tokens):
            for generation_group in response.generations or []:
                for generation in generation_group:
                    message = getattr(generation, "message", None)
                    usage_metadata = getattr(message, "usage_metadata", None)
                    response_metadata = getattr(message, "response_metadata", None) or {}
                    usage = usage_metadata or response_metadata.get("token_usage")
                    gen_prompt, gen_completion = _normalize_usage(usage)
                    prompt_tokens += gen_prompt
                    completion_tokens += gen_completion

        if prompt_tokens or completion_tokens:
            self.usage_dict["prompt_tokens"] += prompt_tokens
            self.usage_dict["completion_tokens"] += completion_tokens
            self.usage_dict["successful_requests"] += 1
