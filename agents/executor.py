from langchain_core.prompts import ChatPromptTemplate

from config.settings import Settings
from graph.state import AgentState
from tools.arxiv_tool import ArxivSearchTool
from prompts.executor_prompts import TOOL_EXTRACTION_PROMPT,TEXT_GENERATION_PROMPT


class ExecutorNode:
    def __init__(self):
        self.llm = Settings.get_llm(temperature=0.1)

        # 注册工具池（未来随意扩充，Executor 逻辑一字都不用改）
        self.tools = {
            "arxiv_search": ArxivSearchTool(),
            # "calculator": CalculatorTool(),      # 假设未来加了计算器
            # "data_plot": DataPlottingTool(),     # 假设未来加了画图工具
        }

    def __call__(self, state: AgentState) -> dict:
        print("\n--- [Executor] Node ---")
        current_index = state["current_step_index"]
        current_step = state["plan"][current_index]
        tool_name = state["planned_tools"][current_index]

        #处理反馈信息
        feedback = ""
        # 检查是否是重试循环
        eval_res = state.get("evaluation_result")
        if eval_res and not  eval_res.get("passed"):
            feedback = f"\n【⚠️上一次执行未通过，评估专家给出的修正建议】: {eval_res.get('feedback')}\n请务必参考此建议调整本次的工具参数或生成逻辑！"


        print(f"正在执行步骤: {current_step}")
        context_raw = "\n".join(state.get("step_history", []))
        # 预处理上下文
        context_str = context_raw if context_raw else "无"
        # output = ""

        # ==========================================
        # 通用路由与执行逻辑
        # ==========================================
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            print(f"    🛠️ 准备调用外部工具: [{tool_name}]")

            # 获取全局原始任务
            original_task = state.get("task_input", "")

            # 动态获取工具的描述。
            # 工具类有 description 属性（这是编写 Agent 工具的标准规范）
            tool_desc = getattr(tool, "description", "执行特定任务的工具")

            # [工程规范修改点]：使用 ChatPromptTemplate 组装链 (Chain)
            prompt_template = ChatPromptTemplate.from_template(TOOL_EXTRACTION_PROMPT)
            chain = prompt_template | self.llm

            # 动态传入变量字典（参数）
            query_res = chain.invoke({
                "original_task": original_task,
                "tool_name": tool_name,
                "tool_desc": tool_desc,
                "current_step": current_step,
                "context": context_str,
                "feedback": feedback
            })
            tool_input = query_res.content.strip()
            print(f"    解析出的工具参数: [{tool_input}]")

            # 统一执行接口
            try:
                tool_result = tool.run(tool_input)
                output = f"【{tool_name} 执行结果】:\n{tool_result}"
            except Exception as e:
                output = f"【{tool_name} 执行失败】: {str(e)}"

        elif tool_name == "generate":
            print(f"    ✍️ 执行文本生成/总结/推理任务...")
            #生成任务提示词的解耦
            prompt_template = ChatPromptTemplate.from_template(TEXT_GENERATION_PROMPT)
            chain = prompt_template | self.llm

            # 传入模板需要的变量
            res = chain.invoke({
                "chat_history": state.get("chat_history", "无"),
                "current_step": current_step,
                "context": context_str
            })
            output = res.content

        else:
            print(f"    ⚠️ 警告: 未知工具 '{tool_name}'")
            output = f"【系统提示】: 无法执行。Planner 分配了未注册的工具 '{tool_name}'。"

        # print(f"    步骤输出预览: {output[:100]}...\n")
        print(f"     步骤输出：\n{output}")

        return {
            "step_history": [f"Step: {current_step}\nTool: {tool_name}\nResult: {output}"]
        }