from config.settings import Settings
from graph.state import AgentState
from tools.arxiv_tool import ArxivSearchTool


class ExecutorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)
        # 初始化真实工具
        self.tools = {"arxiv_search": ArxivSearchTool()}

    def __call__(self, state: AgentState) -> dict:
        print("\n--- [Executor] Node ---")
        current_step = state["plan"][state["current_step_index"]]
        print(f"正在执行步骤: {current_step}")

        output = ""

        # 简单的意图识别逻辑
        if "Search" in current_step or "search" in current_step or "检索" in current_step:
            # 提取查询词（简单处理：把整个步骤作为查询，或者提取关键词）
            # 为了演示，我们硬编码提取关键词逻辑，或者让 LLM 提取
            # 这里简化：直接用 "LLM medical diagnosis" 加上时间限制的概念
            # 实际项目中应由 LLM 生成工具参数
            query = "LLM medical diagnosis"
            print(f"    检测到工具调用需求，Query: {query}")

            tool = self.tools["arxiv_search"]
            tool_result = tool.run(query)
            output = f"【检索结果】:\n{tool_result}"

        else:
            # 纯生成任务 (Summarize)
            # 将历史步骤（包含检索到的论文）作为上下文
            context = "\n".join(state["step_history"])
            prompt = f"""
            你是一个科研助手。基于以下检索到的文献内容，完成步骤：{current_step}。

            已有的文献信息：
            {context}

            要求：
            1. 总结主要方法。
            2. 语言学术且精炼。
            """
            res = self.llm.invoke(prompt)
            output = res.content

        print(f"    步骤输出预览: {output[:100]}...")

        # 将结果追加到历史记录
        return {
            "step_history": [f"Step: {current_step}\nResult: {output}"]
        }