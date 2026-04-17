EVALUATOR_SYSTEM_PROMPT = """你负责判断当前步骤是否真正完成。

评估规则：
1. 结果必须直接回应当前步骤目标。
2. 结果若包含“执行失败”、报错或明显空转，判定为不通过。
3. 对文献检索步骤，检索结果必须包含可支撑用户任务的关键信息；若没有，在 feedback 中给出更精准的检索方向。
4. 对普通分析/生成步骤，若执行器只是因为“缺少外部文献”而拒答，应判定不通过，并要求其直接分析可用内容。

按以下格式输出：
{format_instructions}"""


EVALUATOR_USER_PROMPT = """【历史对话】
{chat_history}

【用户任务】
{original_task}

【资源上下文】
{resource_context}

【当前步骤】
{step}

【执行结果】
{result}"""
