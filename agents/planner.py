import json
from typing import List, Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config.settings import Settings
from langchain_core.output_parsers import JsonOutputParser

# ==========================================
# 1. 依然保留严格的工具池和数据结构约束
# ==========================================
AvailableTools = Literal[
    "arxiv_search",
    "generate",
    # "web_search",
    # "data_analysis"
]


class PlanStep(BaseModel):
    task_description: str = Field(description="该步骤需要完成的具体任务描述")
    tool_name: AvailableTools = Field(description="执行此步骤必须调用的工具。如果不需外部工具请选 'generate'")


class ExecutionPlan(BaseModel):
    steps: List[PlanStep] = Field(description="按照执行顺序排列的步骤列表")


# ==========================================
# 2. 规划器节点实现
# ==========================================
class PlannerNode:
    def __init__(self):
        # 规划节点，严谨起见 temperature 依然设为较低值
        self.llm = Settings.get_llm(temperature=0.1)

        # 核心改动：使用 JsonOutputParser 替代 with_structured_output
        self.parser = JsonOutputParser(pydantic_object=ExecutionPlan)

    def __call__(self, state: dict) -> dict:
        print("\n--- [Planner] Node ---")
        user_request = state.get("task_input", "")

        # 核心改动：在 System Prompt 中注入 parser 自动生成的格式化指令
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一名经验丰富的科研任务规划专家。你的目标是将用户的模糊需求拆解为可执行的科研步骤。\n"
             "【严禁上下文丢失】：每个步骤的描述（task_description）必须包含具体的研究对象、实体名称或专业术语，绝不能使用'相关领域'、'目标文献'等模糊代词。例如：不要写'搜索相关文献'，必须写'在arXiv上搜索强化学习(RL)在自动驾驶(Autonomous Driving)领域的文献'。\n\n"
             "【禁止预判与虚构】（极其重要）：绝不能在计划中虚构、假设或举例具体的未来数据（如特定的论文标题、作者名）。\n\n"
             "严格遵守以下输出格式指南：\n{format_instructions}"),
            ("user", "用户任务：{task}")
        ])

        # 构建处理链：Prompt -> LLM -> JSON解析器
        chain = prompt | self.llm | self.parser

        try:
            # 传入 task 和 parser 自动生成的格式要求
            plan_dict = chain.invoke({
                "task": user_request,
                "format_instructions": self.parser.get_format_instructions()
            })

            plan_descriptions = []
            assigned_tools = []

            # 因为 parser 解析后返回的是 dict，我们需要按字典方式读取
            for i, step in enumerate(plan_dict.get("steps", [])):
                desc = step.get("task_description", "")
                tool = step.get("tool_name", "generate")
                print(f"  步骤 {i + 1}: {desc} [分配工具: {tool}]")
                plan_descriptions.append(desc)
                assigned_tools.append(tool)

            return {
                "plan": plan_descriptions,
                "planned_tools": assigned_tools,  # 传递给 Executor 使用
                "current_step_index": 0
            }

        except Exception as e:
            print(f"[Error] Planner 解析 JSON 失败: {e}")
            # 基础兜底逻辑，防止图直接崩溃
            return {
                "plan": [f"直接处理用户任务: {user_request}"],
                "planned_tools": ["generate"],
                "current_step_index": 0
            }