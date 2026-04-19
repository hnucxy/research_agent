import os
import re

import streamlit as st

from graph.graph_builder import build_graph
from utils.prompt_utils import build_resource_context
from utils.token_tracker import TokenTracker

def build_chat_history_string(messages) -> str:
    history_msgs = []
    for msg in messages[:-1]:
        role_name = "用户" if msg["role"] == "user" else "助手"
        history_msgs.append(f"[{role_name}]: {msg.get('content', '')}")
    return "\n".join(history_msgs) if history_msgs else "无历史对话"


def load_reviewer_draft_content(selected_files_for_agent) -> str:
    draft_content = ""
    if not selected_files_for_agent:
        return draft_content

    file_path = selected_files_for_agent[0]["path"]
    if file_path.endswith(".md") and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            draft_content = file.read()
        st.info(
            f"已识别勾选文件为原稿：{selected_files_for_agent[0]['name']}"
        )
    return draft_content


def handle_reviewer_flow(
    prompt: str,
    draft_content: str,
    chat_history_str: str,
    run_config: dict,
    process_logs: list[str],
):
    from graph.graph_builder import build_reviewer_graph

    final_output = ""
    agent_app = build_reviewer_graph()
    initial_state = {
        "current_function": "d",
        "user_prompt": prompt,
        "draft_content": draft_content,
        "original_draft_content": draft_content,
        "feedback": "",
        "task_intent": "",
        "review_mode": "",
        "review_focus": "",
        "diff_content": "",
        "final_diff_content": "",
        "retry_count": 0,
        "max_retries": 3,
        "status": "",
        "chat_history": chat_history_str,
        "final_answer": "",
    }

    for output in agent_app.stream(initial_state, config=run_config):
        for node_name, state_update in output.items():
            if node_name == "input_parser":
                if "draft_content" in state_update:
                    draft_content = state_update["draft_content"]
                st.write(
                    "🧩 **解析器** 已完成任务识别。"
                    f" `intent={state_update.get('task_intent', 'general_revision')}`"
                    f" `mode={state_update.get('review_mode', 'relaxed')}`"
                )

            elif node_name == "reviewer":
                status_flag = state_update.get("status")
                feedback = state_update.get("feedback", "")
                current_manuscript = state_update.get("draft_content", draft_content)
                final_diff = state_update.get("final_diff_content", "")

                if status_flag == "pass":
                    log_msg = f"✅ **Reviewer (通过)**: {feedback}"
                    st.success(log_msg)
                    final_output = f"### ✅ 最终审定文稿\n\n{current_manuscript}"
                    if final_diff:
                        final_output += f"\n\n---\n#### 增量修改视图\n\n{final_diff}"
                    if feedback:
                        final_output += f"\n\n---\n**Reviewer 意见**：{feedback}"
                elif status_flag == "reject":
                    log_msg = f"⛔ **Reviewer (驳回)**: {feedback}"
                    st.error(log_msg)
                    final_output = f"### ⛔ 任务被驳回\n\n**原因**：{feedback}"
                else:
                    log_msg = f"⚠️ **Reviewer (修改建议)**: {feedback}"
                    st.warning(log_msg)

                process_logs.append(log_msg)

            elif node_name == "author":
                retry_count = state_update.get("retry_count", 0)
                draft_content = state_update.get("draft_content", "")
                diff_content = state_update.get("diff_content", "")

                log_msg = f"✍️ **Author**: 完成第 {retry_count} 次修改。"
                st.info(log_msg)
                process_logs.append(
                    f"{log_msg}\n\n**增量修改视图**：\n\n{diff_content or draft_content}"
                )

                if retry_count >= 3:
                    final_output = f"### ⚠️ 达到最大修改次数\n\n{draft_content}"

    return final_output


def handle_general_flow(
    prompt: str,
    chat_history_str: str,
    selected_files_for_agent,
    selected_image_for_chat,
    search_source: str,
    semantic_sort_by: str,
    run_config: dict,
    process_logs: list[str],
):
    resource_context = build_resource_context(
        selected_files=selected_files_for_agent,
        selected_image_path=selected_image_for_chat,
        search_source=search_source if st.session_state.current_function == "a" else None,
        semantic_sort_by=semantic_sort_by
        if st.session_state.current_function == "a"
        else None,
    )

    initial_state = {
        "current_function": st.session_state.current_function,
        "task_input": prompt,
        "resource_context": resource_context,
        "selected_image_path": selected_image_for_chat or "",
        "retrieved_image_paths": [],
        "chat_history": chat_history_str,
        "search_source": search_source,
        "semantic_sort_by": semantic_sort_by,
        "plan": [],
        "planned_tools": [],
        "current_step_index": 0,
        "retry_count": 0,
        "replan_count": 0,
        "step_history": [],
        "evaluation_result": {},
        "final_answer": "",
    }

    final_output = ""
    auto_image_paths = []
    agent_app = build_graph()

    for output in agent_app.stream(initial_state, config=run_config):
        for node_name, state_update in output.items():
            if node_name == "planner":
                st.write("🧠 **规划器 (Planner)** 制定了新计划：")
                process_logs.append("🧠 **规划器 (Planner)** 制定了新计划：")
                plans = state_update.get("plan", [])
                tools = state_update.get("planned_tools", [])
                for idx, (plan, tool) in enumerate(zip(plans, tools), start=1):
                    log_str = f"**Step {idx}**: {plan} `[Tool: {tool}]`"
                    st.info(log_str)
                    process_logs.append(f"- {log_str}")

            elif node_name == "executor":
                if state_update.get("retrieved_image_paths"):
                    auto_image_paths = state_update.get("retrieved_image_paths", [])
                step_history = state_update.get("step_history", [])
                if step_history:
                    st.write("🛠️ **执行器 (Executor)** 完成操作：")
                    st.code(step_history[-1], language="text")
                    process_logs.append(
                        f"🛠️ **执行器操作**:\n```text\n{step_history[-1]}\n```"
                    )

                    last_log = step_history[-1]
                    match = re.search(r"Result:\s*(.*)", last_log, flags=re.DOTALL)
                    parsed_text = match.group(1).strip() if match else last_log.strip()
                    parsed_text = re.sub(r"^【[^】]+】\n?", "", parsed_text).strip()
                    if parsed_text:
                        final_output = parsed_text

            elif node_name == "evaluator":
                eval_res = state_update.get("evaluation_result", {})
                passed = eval_res.get("passed", False)
                feedback = eval_res.get("feedback", "")
                if passed:
                    log_str = f"✅ **评估 (Evaluator)**: 步骤通过。(反馈: {feedback})"
                    st.success(log_str)
                else:
                    log_str = f"⚠️ **评估 (Evaluator)**: 未通过，触发修正重试。(反馈: {feedback})"
                    st.warning(log_str)
                process_logs.append(log_str)

            elif node_name == "give_up":
                final_output = ""

    return final_output, auto_image_paths


def run_chat_turn(
    prompt: str,
    selected_files_for_agent,
    selected_image_for_chat,
    search_source: str,
    semantic_sort_by: str,
    token_usage: dict,
    messages,
):
    chat_history_str = build_chat_history_string(messages)
    process_logs = []
    tracker = TokenTracker(token_usage)
    run_config = {"callbacks": [tracker]}
    final_output = ""
    auto_image_paths = []

    if st.session_state.current_function == "d":
        draft_content = load_reviewer_draft_content(selected_files_for_agent)
        final_output = handle_reviewer_flow(
            prompt=prompt,
            draft_content=draft_content,
            chat_history_str=chat_history_str,
            run_config=run_config,
            process_logs=process_logs,
        )
    else:
        final_output, auto_image_paths = handle_general_flow(
            prompt=prompt,
            chat_history_str=chat_history_str,
            selected_files_for_agent=selected_files_for_agent,
            selected_image_for_chat=selected_image_for_chat,
            search_source=search_source,
            semantic_sort_by=semantic_sort_by,
            run_config=run_config,
            process_logs=process_logs,
        )

    return final_output, auto_image_paths, process_logs
