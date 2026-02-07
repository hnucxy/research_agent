from graph.graph_builder import build_graph
from config.settings import Settings
import sys

# 简单测试 API key 是否存在
if not Settings.DEEPSEEK_API_KEY:
    print("错误：未检测到 DEEPSEEK_API_KEY，请在 .env 文件中配置。")
    sys.exit(1)


def main():
    print(">>> 初始化科研 Agent (DeepSeek Kernel)...")
    app = build_graph()

    # 你指定的 User Query
    user_query = "请帮我调研最近3个月关于 LLM 在医学诊断领域应用的论文，并总结主要方法。"

    initial_state = {
        "task_input": user_query,
        "long_term_context": "",
        "plan": [],
        "current_step_index": 0,
        "step_history": [],
        "evaluation_result": {},
        "final_answer": ""
    }

    print(f">>> 任务: {user_query}")
    print(">>> 开始执行图...\n")

    try:
        # 使用 invoke 而不是 stream，或者简单遍历 stream
        for output in app.stream(initial_state):
            pass  # 具体的打印逻辑已在各个 Node 内部实现

        print("\n>>> 任务执行完毕。")
    except Exception as e:
        print(f"\n[Error] 执行过程中发生错误: {e}")


if __name__ == "__main__":
    main()