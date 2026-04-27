from abc import ABC, abstractmethod

class BaseTool(ABC):
    name: str
    description: str
    prompt_spec: str = ""

    @abstractmethod
    def run(self, params: str, config: dict | None = None) -> str:
        """执行工具逻辑"""
        pass
