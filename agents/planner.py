from typing import List, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from config.settings import Settings
from config.logger import get_logger
from prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from utils.exceptions import AgentPlanningError


# 1. 动态工具池与数据结构约束 根据当前对话种类进行限定
# 针对检索、撰写等非阅读功能的工具池（剔除 reading 和 rag）
NonReadTools = Literal["arxiv_search", "academic_write", "generate"]
# 包含所有工具的全量工具池
ReadTools = Literal["arxiv_search", "academic_write", "generate", "literature_read", "literature_rag_search"]

class NonReadPlanStep(BaseModel):
    task_description: str = Field(description="该步骤需要完成的具体任务描述")
    tool_name: NonReadTools = Field(description="执行此步骤必须调用的工具。如果不需外部工具请选 'generate'")

class ReadPlanStep(BaseModel):
    task_description: str = Field(description="该步骤需要完成的具体任务描述")
    tool_name: ReadTools = Field(description="执行此步骤必须调用的工具。如果不需外部工具请选 'generate'")

class NonReadExecutionPlan(BaseModel):
    steps: List[NonReadPlanStep] = Field(description="按照执行顺序排列的步骤列表")

class ReadExecutionPlan(BaseModel):
    steps: List[ReadPlanStep] = Field(description="按照执行顺序排列的步骤列表")


# 初始化logger
logger = get_logger()

# 2. 规划器节点实现

class PlannerNode:
    def __init__(self):
        # 规划节点，严谨起见 temperature 依然设为较低值
        self.llm = Settings.get_llm(temperature=0.1)

        self.embeddings = Settings.get_embeddings()
        # self.parser在运行时动态实例化

    def __call__(self, state: dict) -> dict:
        logger.info("")
        logger.info("--- [Planner] Node ---")
        user_request = state.get("task_input", "")

        # 获取传入的当前功能类型，默认 fallback 为 'c'
        current_func = state.get("current_function", "c")

        # 检索并融合记忆/经验
        historical_experience = "无"
        if state.get("replan_count", 0) == 0:
            try:
                vectorstore = Chroma(
                    collection_name="global_experience",
                    embedding_function=self.embeddings,
                    persist_directory="./chroma_db"
                )
                # 检索最相关的 1 条经验
                docs_with_score = vectorstore.similarity_search_with_score(user_request, k=1)
                if docs_with_score:
                    doc, score = docs_with_score[0]
                    
                    if current_func != "a" and score < 0.3:
                        logger.info(f"    [Planner] 命中高度相似经验(score:{score:.2f})，分配 memo_output 工具短路 LLM。")
                        return {
                            "plan": [f"【语义缓存直接输出】\n{doc.page_content}"],
                            "planned_tools": ["memo_output"], # <--- 修改此处，使用特殊标识
                            "current_step_index": 0,
                            "retry_count": 0,
                            "replan_count": 0
                        }
                    elif score < 0.6: 
                        # 2. 检索模块(无论分数多高) 或 非检索模块但相似度一般(需要推敲)
                        # 将经验提取出来，交由大模型二次确认
                        historical_experience = doc.page_content
                        logger.info(f"    [Planner] 检索到相关历史经验(score:{score:.2f})，将交由大模型进行二次确认与融合规划...")
            except Exception as e:
                logger.warning("    [Planner] 检索经验库跳过或无相关经验。")

        # 动态注入提示，定制二次确认的具体策略
        enhanced_task = user_request
        if historical_experience != "无":
            if current_func == "a":  # 文献检索：强制双轨
                strategy_prompt = "科研文献具有强时效性，绝不能仅依赖历史经验。你必须在规划中同时包含调用文献检索工具（如 arxiv_search）的步骤，以获取最新进展，实现“吸收历史经验 + 检索最新文献”的双轨处理。"
            elif current_func == "b": # 学术撰写：整合润色
                strategy_prompt = "请评估该历史撰写经验/结论是否适用于当前写作任务。如果可以复用，请结合 `academic_write` 工具进行内容整合、扩写或润色。"
            elif current_func == "c": # 文献阅读：补充验证
                strategy_prompt = "请评估该历史阅读结论是否能直接回答当前问题。如果能，你可以直接将任务分配给 `generate` 生成结论；或者规划 `literature_read` / `literature_rag_search` 进行补充验证。"
            else:
                strategy_prompt = "请结合当前任务判断是否复用历史经验，并合理分配工具来完成任务。"

            enhanced_task = (
                f"原始任务：{user_request}\n\n"
                f"【系统补充：检索到的历史成功经验】\n{historical_experience}\n\n"
                f"【规划要求】：请对上述历史经验进行二次确认。如果该经验与当前任务强相关，请在接下来的规划中吸收其高价值结论或思路。\n"
                f"【执行策略】：{strategy_prompt}"
            )
        # 
        # 根据对话种类动态选择解析器，从根源限制大模型能看到的 JSON Schema 工具枚举
        if current_func in ["a", "b"]:
            parser = JsonOutputParser(pydantic_object=NonReadExecutionPlan)
            mode_prompt_addon = "\n\n【动态模式约束】：当前为非文献阅读模式，绝对禁止分配 `literature_read` 或 `literature_rag_search` 工具，请仅从输出格式要求的 Enum 工具列表中进行选择。"
        else:
            parser = JsonOutputParser(pydantic_object=ReadExecutionPlan)
            mode_prompt_addon = ""

        # 动态拼接最终的 System Prompt
        dynamic_system_prompt = PLANNER_SYSTEM_PROMPT + mode_prompt_addon

        # 使用导入的常量组装 prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", dynamic_system_prompt),
            ("user", PLANNER_USER_PROMPT)
        ])

        # 构建处理链：Prompt -> LLM -> JSON解析器
        chain = prompt | self.llm | parser
        # 提取历史教训：直接提取 Evaluator 的评估反馈
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
                "task": enhanced_task,
                "step_history": step_history_str,  # 传入失败教训
                "format_instructions": parser.get_format_instructions()
            })

            plan_descriptions = []
            assigned_tools = []

            # 因为 parser 解析后返回的是 dict，我们需要按字典方式读取
            for i, step in enumerate(plan_dict.get("steps", [])):
                desc = step.get("task_description", "")
                tool = step.get("tool_name", "generate")
                # print(f"  步骤 {i + 1}: {desc} [分配工具: {tool}]")
                logger.info("  步骤 %s : %s [分配工具: %s]", i+1, desc, tool)
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
            
            logger.error("[Error] Planner 解析 JSON 失败: %s", e)
            raise AgentPlanningError(f"规划器解析大模型输出失败: {e}")
            # 基础兜底逻辑，防止图直接崩溃
            # return {
            #     "plan": [f"直接处理用户任务: {user_request}"],
            #     "planned_tools": ["generate"],
            #     "current_step_index": 0,
            #     "retry_count": 0,
            #     "replan_count": state.get("replan_count", 0)
            # }