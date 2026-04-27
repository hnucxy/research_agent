from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from graph.graph_builder import build_graph
from utils.prompt_utils import build_resource_context


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    message: str = ""
    log: str = ""
    code: str = ""
    final_output: str | None = None
    auto_image_paths: tuple[str, ...] = ()


def build_chat_history_string(messages: Iterable[dict]) -> str:
    history_msgs = []
    for msg in list(messages)[:-1]:
        role_name = "用户" if msg["role"] == "user" else "助手"
        history_msgs.append(f"[{role_name}]: {msg.get('content', '')}")
    return "\n".join(history_msgs) if history_msgs else "无历史对话"


def load_reviewer_draft_content(selected_files_for_agent) -> tuple[str, str]:
    if not selected_files_for_agent:
        return "", ""

    file_path = selected_files_for_agent[0]["path"]
    if file_path.endswith(".md") and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read(), selected_files_for_agent[0]["name"]
    return "", ""


def stream_reviewer_flow(
    prompt: str,
    draft_content: str,
    chat_history_str: str,
    run_config: dict,
):
    from graph.graph_builder import build_reviewer_graph

    final_output = ""
    current_draft = draft_content
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
                    current_draft = state_update["draft_content"]
                yield AgentEvent(
                    kind="write",
                    message=(
                        "🧩 **解析器** 已完成任务识别。"
                        f" `intent={state_update.get('task_intent', 'general_revision')}`"
                        f" `mode={state_update.get('review_mode', 'relaxed')}`"
                    ),
                )

            elif node_name == "reviewer":
                status_flag = state_update.get("status")
                feedback = state_update.get("feedback", "")
                manuscript = state_update.get("draft_content", current_draft)
                final_diff = state_update.get("final_diff_content", "")

                if status_flag == "pass":
                    log_msg = f"✅ **Reviewer (通过)**: {feedback}"
                    final_output = f"### ✅ 最终审定文稿\n\n{manuscript}"
                    if final_diff:
                        final_output += f"\n\n---\n#### 增量修改视图\n\n{final_diff}"
                    if feedback:
                        final_output += f"\n\n---\n**Reviewer 意见**：{feedback}"
                    yield AgentEvent(
                        kind="success",
                        message=log_msg,
                        log=log_msg,
                        final_output=final_output,
                    )
                elif status_flag == "reject":
                    log_msg = f"⛔ **Reviewer (驳回)**: {feedback}"
                    final_output = f"### ⛔ 任务被驳回\n\n**原因**：{feedback}"
                    yield AgentEvent(
                        kind="error",
                        message=log_msg,
                        log=log_msg,
                        final_output=final_output,
                    )
                else:
                    log_msg = f"⚠️ **Reviewer (修改建议)**: {feedback}"
                    yield AgentEvent(kind="warning", message=log_msg, log=log_msg)

            elif node_name == "author":
                retry_count = state_update.get("retry_count", 0)
                current_draft = state_update.get("draft_content", "")
                diff_content = state_update.get("diff_content", "")

                log_msg = f"✍️ **Author**: 完成第 {retry_count} 次修改。"
                yield AgentEvent(
                    kind="info",
                    message=log_msg,
                    log=(
                        f"{log_msg}\n\n**增量修改视图**：\n\n"
                        f"{diff_content or current_draft}"
                    ),
                )

                if retry_count >= 3:
                    final_output = f"### ⚠️ 达到最大修改次数\n\n{current_draft}"
                    yield AgentEvent(kind="final", final_output=final_output)


def stream_general_flow(
    prompt: str,
    chat_history_str: str,
    selected_files_for_agent,
    selected_image_for_chat,
    search_source: str,
    semantic_sort_by: str,
    semantic_year_filter: str,
    current_function: str,
    current_chat_id: str,
    run_config: dict,
):
    resource_context = build_resource_context(
        selected_files=selected_files_for_agent,
        selected_image_path=selected_image_for_chat,
        search_source=search_source if current_function == "a" else None,
        semantic_sort_by=semantic_sort_by if current_function == "a" else None,
        semantic_year_filter=semantic_year_filter if current_function == "a" else None,
        user_task=prompt,
    )

    initial_state: dict[str, Any] = {
        "current_function": current_function,
        "current_chat_id": current_chat_id,
        "task_input": prompt,
        "resource_context": resource_context,
        "selected_image_path": selected_image_for_chat or "",
        "retrieved_image_paths": [],
        "chat_history": chat_history_str,
        "search_source": search_source,
        "semantic_sort_by": semantic_sort_by,
        "semantic_year_filter": semantic_year_filter,
        "plan": [],
        "planned_tools": [],
        "current_step_index": 0,
        "retry_count": 0,
        "replan_count": 0,
        "step_history": [],
        "evaluation_result": {},
        "final_answer": "",
    }

    agent_app = build_graph()

    for output in agent_app.stream(initial_state, config=run_config):
        for node_name, state_update in output.items():
            if node_name == "planner":
                yield AgentEvent(
                    kind="write",
                    message="🧠 **规划器 (Planner)** 制定了新计划：",
                    log="🧠 **规划器 (Planner)** 制定了新计划：",
                )
                plans = state_update.get("plan", [])
                tools = state_update.get("planned_tools", [])
                for idx, (plan, tool) in enumerate(zip(plans, tools), start=1):
                    log_str = f"**Step {idx}**: {plan} `[Tool: {tool}]`"
                    yield AgentEvent(kind="info", message=log_str, log=f"- {log_str}")

            elif node_name == "executor":
                auto_image_paths = tuple(state_update.get("retrieved_image_paths") or ())
                if auto_image_paths:
                    yield AgentEvent(kind="images", auto_image_paths=auto_image_paths)

                step_history = state_update.get("step_history", [])
                if step_history:
                    last_log = step_history[-1]
                    yield AgentEvent(
                        kind="write",
                        message="🛠️ **执行器 (Executor)** 完成操作：",
                    )
                    yield AgentEvent(
                        kind="code",
                        code=last_log,
                        log=f"🛠️ **执行器操作**:\n```text\n{last_log}\n```",
                    )

                    match = re.search(r"Result:\s*(.*)", last_log, flags=re.DOTALL)
                    parsed_text = match.group(1).strip() if match else last_log.strip()
                    parsed_text = re.sub(r"^【[^】]+】\n?", "", parsed_text).strip()
                    if parsed_text:
                        yield AgentEvent(kind="final", final_output=parsed_text)

            elif node_name == "evaluator":
                eval_res = state_update.get("evaluation_result", {})
                passed = eval_res.get("passed", False)
                feedback = eval_res.get("feedback", "")
                if passed:
                    log_str = f"✅ **评估 (Evaluator)**: 步骤通过。(反馈: {feedback})"
                    yield AgentEvent(kind="success", message=log_str, log=log_str)
                else:
                    log_str = f"⚠️ **评估 (Evaluator)**: 未通过，触发修正重试。(反馈: {feedback})"
                    yield AgentEvent(kind="warning", message=log_str, log=log_str)

            elif node_name == "give_up":
                yield AgentEvent(kind="final", final_output="")

