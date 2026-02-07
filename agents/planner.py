import json
import re
from config.settings import Settings
from graph.state import AgentState


class PlannerNode:
    def __init__(self):
        # 适当调高温度，让 CoT 更丰富
        self.llm = Settings.get_llm(temperature=0.3)

    def _parse_json(self, text: str) -> dict:
        """清洗并解析 LLM 返回的 JSON"""
        try:
            # 尝试直接解析
            return json.loads(text)
        except:
            # 如果包含 markdown 代码块，提取出来
            match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # 兜底：尝试找大括号
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"无法解析 JSON: {text}")

    def __call__(self, state: AgentState) -> dict:
        print("\n--- [Planner] Node: Generating Chain-of-Thought ---")
        task = state["task_input"]

        # 修改后的通用 Prompt
        prompt = f"""
        你是一名经验丰富的科研任务规划专家。你的目标是将用户的模糊需求拆解为可执行的科研步骤。

        用户任务: {task}

        请遵循以下原则进行规划（Chain-of-Thought）：
        1. **需求分析**：判断任务类型（综述、对比、溯源等）。
        2. **工具感知**：你拥有一个 arXiv 搜索引擎。
           - 如果任务需要外部知识或最新论文，**必须** 生成包含 "Search"、"Retrieve" 或 "检索" 关键词的步骤。
           - 搜索步骤应明确具体的搜索查询词（Query）。
        3. **逻辑闭环**：检索后必须有步骤对结果进行处理（如 "Summarize"、"Analyze" 或 "总结"）。

        请输出纯 JSON 格式（不要使用 Markdown 代码块）：
        {{
            "reasoning": "简要分析任务难点及规划思路...",
            "plan": [
                "Step 1: Search for [具体关键词]...",
                "Step 2: Summarize the retrieved papers..."
            ]
        }}
        """

        response = self.llm.invoke(prompt).content

        # 解析输出
        data = self._parse_json(response)

        print(f"\033[94m[CoT Reasoning]: {data['reasoning']}\033[0m")
        print(f"[Plan]: {data['plan']}")

        return {
            "plan": data["plan"],
            "current_step_index": 0,
            "step_history": []
        }