from abc import ABC, abstractmethod

class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, params: str) -> str:
        """执行工具逻辑"""
        pass