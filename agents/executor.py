from config.settings import Settings
from graph.state import AgentState
from tools.arxiv_tool import ArxivSearchTool


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
        feedback = ""
        # 检查是否是重试循环
        eval_res = state.get("evaluation_result")
        if eval_res and not  eval_res.get("passed"):
            feedback = f"\n【⚠️上一次执行未通过，评估专家给出的修正建议】: {eval_res.get('feedback')}\n请务必参考此建议调整本次的工具参数或生成逻辑！"
        tool_name = state["planned_tools"][current_index]

        print(f"正在执行步骤: {current_step}")
        context = "\n".join(state.get("step_history", []))
        output = ""

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

            # 这是一个通用的参数提取 Prompt
            extract_prompt = f"""
            你是一个工具参数提取助手。请根据【全局用户任务】和【当前执行步骤】，为指定的工具提取或生成合适的输入参数。

            【全局用户任务】: {original_task}
            【目标工具名称】: {tool_name}
            【目标工具功能】: {tool_desc}
            【当前任务描述】: {current_step}
            【已有上下文信息】: {context if context else "无"}
            {feedback} # <---注入反馈

            要求：
            1. 仔细阅读工具功能，理解该工具需要什么类型的输入（如：搜索关键词、数学表达式、Python代码等）。
            2. 根据任务描述和全局用户任务，推导出该工具需要的具体参数。
            3. 如果是调用文献检索工具（如arxiv_search），提取的关键词应尽量精简并使用双引号确保精确匹配（例如：\"reinforcement learning\" AND \"autonomous driving\"），不要包含年份或“最近3个月”等宽泛的自然语言时间词，这会导致检索引擎混乱。
            4. 请直接输出该参数字符串，绝对不要包含任何解释文字、前缀或 Markdown 代码块标记。
            """

            query_res = self.llm.invoke(extract_prompt)
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
            prompt = f"""
            你是一个专业的科研助手。请基于以下已有信息，完成指定的任务。
            
            【当前任务指令】：
            {current_step}

            【已有上下文信息】：
            {context if context else "无"}

            【严格事实约束】（极其重要）：
            1. 你必须且只能基于【已有上下文信息】进行回答。
            2. 仔细审查上下文信息是否与任务指令真实相关。如果上下文提供的文献与任务（如自动驾驶、强化学习）毫无关联，你必须明确指出“检索到的文献与任务无关”，并拒绝执行总结或筛选。
            3. 绝不能为了完成任务而强行解释、捏造关联或过度联想。

            要求：语言保持学术严谨且精炼，直接输出结果。
            """
            res = self.llm.invoke(prompt)
            output = res.content

        else:
            print(f"    ⚠️ 警告: 未知工具 '{tool_name}'")
            output = f"【系统提示】: 无法执行。Planner 分配了未注册的工具 '{tool_name}'。"

        # print(f"    步骤输出预览: {output[:100]}...\n")
        print(f"     步骤输出：\n{output}")

        return {
            "step_history": [f"Step: {current_step}\nTool: {tool_name}\nResult: {output}"]
        }