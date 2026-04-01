from typing import List, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import Settings
from prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT

# ==========================================
# 1. 依然保留严格的工具池和数据结构约束
# ==========================================
AvailableTools = Literal[
    "arxiv_search",
    "generate",
    "academic_write",
    "literature_read"
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

        # 使用导入的常量组装 prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("user", PLANNER_USER_PROMPT)
        ])

        # 构建处理链：Prompt -> LLM -> JSON解析器
        chain = prompt | self.llm | self.parser
        # 提取历史教训：不再使用原始的冗长执行日志，而是直接提取 Evaluator 的评估反馈
        eval_res = state.get("evaluation_result", {})
        replan_count = state.get("replan_count", 0)

        # 只有在存在评估结果且未通过时，才构造失败教训
        if eval_res and not eval_res.get("passed", True):
            last_plan = state.get("plan", [])
            feedback = eval_res.get("feedback", "无评估反馈")
            step_history_str = (
                f"【注意：这是第 {replan_count} 次重新规划】\n"
                f"你上一次制定的计划步骤是：{last_plan}\n"
                f"但该计划执行失败。评估专家的拒绝原因与修改建议如下：\n"
                f"\"{feedback}\"\n"
                f"要求：请务必吸取上述教训，严格按照专家的建议更换搜索关键词或调整策略，切勿尝试去总结或处理上一次检索到的无效内容！"
            )
        else:
            step_history_str = "无"
        try:
            # 传入 task 和 parser 自动生成的格式要求
            plan_dict = chain.invoke({
                "chat_history": state.get("chat_history", "无"),
                "task": user_request,
                "step_history": step_history_str,  # 传入失败教训
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


            # 计算全局重规划次数
            current_replan_count = state.get("replan_count", 0)
            if step_history_str != "无":
                current_replan_count += 1

            return {
                "plan": plan_descriptions,
                "planned_tools": assigned_tools,  # 传递给 Executor 使用
                "current_step_index": 0,
                "retry_count": 0,  # 重置局部重试次数
                "replan_count": current_replan_count  # 更新全局重规划次数
            }

        except Exception as e:
            print(f"[Error] Planner 解析 JSON 失败: {e}")
            # 基础兜底逻辑，防止图直接崩溃
            return {
                "plan": [f"直接处理用户任务: {user_request}"],
                "planned_tools": ["generate"],
                "current_step_index": 0,
                "retry_count": 0,
                "replan_count": state.get("replan_count", 0)
            }