from graph.state import AgentState
from config.logger import get_logger

logger = get_logger()

class MemoryNode:
    """
    负责显式记忆管理：
    1. 增加步骤索引 (Short-term memory update)
    2. 将成功的步骤结果写入向量库 (Long-term memory write)
    """

    def __call__(self, state: AgentState) -> dict:
        # print("\n--- [Memory] Node ---")
        logger.info("--- [Memory] Node ---")

        # [Memory Policy]
        # 仅在 Evaluator 认为 Passed 后，才写入长期记忆
        # 这里模拟写入操作
        # print("Writing result to Long-term Memory (VectorDB)...")
        logger.info("Writing result to Long-term Memory (VectorDB)...")

        # 更新步骤索引，准备执行下一步
        new_index = state["current_step_index"] + 1

        return {
            "current_step_index": new_index,
            # 如果是最后一步，可以在这里汇总 final_answer
        }