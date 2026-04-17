from graph.state import AgentState
from config.logger import get_logger
from langchain_chroma import Chroma
from config.settings import Settings

logger = get_logger()

class MemoryNode:
    """
    负责显式记忆管理：
    1. 将成功的步骤结果写入向量库 (Long-term memory write)
    2. 增加步骤索引，推进流程 (Short-term memory update)
    """
    def __init__(self):
        # 初始化词嵌入模型
        self.embeddings = Settings.get_embeddings()

    def __call__(self, state: AgentState) -> dict:
        logger.info("--- [Memory] Node ---")

        eval_res = state.get("evaluation_result", {})
        
        # [Memory Policy] 仅在 Evaluator 认为 Passed 后，才写入长期记忆
        if eval_res.get("passed"):
            logger.info("Writing result to Long-term Memory (VectorDB)...")
            
            # 提取记忆所需的元素
            task = state.get("task_input", "未知任务")
            plan = state.get("plan", [])
            curr_idx = state.get("current_step_index", 0)
            step_desc = plan[curr_idx] if curr_idx < len(plan) else "未知步骤"
            history = state.get("step_history", [])
            result = history[-1] if history else "无结论"

            # 重新组合成高价值文本块
            experience_text = (
                f"【用户原始问题/任务】: {task}\n"
                f"【成功执行路径/步骤】: {step_desc}\n"
                f"【高价值结论】: {result}"
            )
            
            try:
                # 写入到单独的 global_experience Collection
                vectorstore = Chroma(
                    collection_name=Settings.get_collection_name("global_experience"),
                    embedding_function=self.embeddings,
                    persist_directory="./chroma_db"
                )
                # 写入前进行相似度去重检测
                is_duplicate = False
                # 检索是否存在极其相似的经验
                existing_docs = vectorstore.similarity_search_with_score(experience_text, k=1)
                if existing_docs:
                    _, score = existing_docs[0]
                    # 分数域值设定（L2距离下越小越相似，0.15代表文本几乎完全一致）
                    if score < 0.15: 
                        is_duplicate = True
                        
                if is_duplicate:
                    logger.info("    [Memory] 经验库中已存在高度相似的记录，跳过重复写入避免冗余。")
                else:
                    vectorstore.add_texts(texts=[experience_text])
                    logger.info("    [Memory] 成功写入全局经验库！")
                
            except Exception as e:
                logger.error("    [Memory] 写入经验库失败: %s", e)

        # 更新步骤索引，准备执行下一步
        new_index = state["current_step_index"] + 1

        return {
            "current_step_index": new_index,
            "retry_count": 0,
            "evaluation_result": {} # 清空上一次的评估状态
        }
