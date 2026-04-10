class BaseAgentException(Exception):
    """Agent 系统的基础异常类"""
    pass

class AgentPlanningError(BaseAgentException):
    """规划器(Planner)解析或生成计划失败异常"""
    pass

class AgentEvaluationError(BaseAgentException):
    """评估器(Evaluator)执行或解析失败异常"""
    pass

class ToolExecutionError(BaseAgentException):
    """工具执行的基础异常"""
    pass

class ToolExecutionTimeout(ToolExecutionError):
    """工具执行超时异常"""
    pass

class VectorDBConnectionError(BaseAgentException):
    """向量数据库连接或检索失败异常"""
    pass

class LLMConnectionError(BaseAgentException):
    """大语言模型API调用失败异常"""
    pass

class DocumentParseError(BaseAgentException):
    """文献解析或转换失败异常"""
    pass