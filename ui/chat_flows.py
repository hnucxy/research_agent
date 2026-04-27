import streamlit as st

from services.agent_service import (
    build_chat_history_string,
    load_reviewer_draft_content,
    stream_general_flow,
    stream_reviewer_flow,
)
from utils.token_tracker import TokenTracker


class EventRenderer:
    def __init__(self, process_logs: list[str]):
        self.process_logs = process_logs
        self.final_output = ""
        self.auto_image_paths: list[str] = []

    def render(self, event) -> None:
        if event.kind == "write":
            st.write(event.message)
        elif event.kind == "info":
            st.info(event.message)
        elif event.kind == "success":
            st.success(event.message)
        elif event.kind == "warning":
            st.warning(event.message)
        elif event.kind == "error":
            st.error(event.message)
        elif event.kind == "code":
            st.code(event.code, language="text")

        if event.log:
            self.process_logs.append(event.log)
        if event.final_output is not None:
            self.final_output = event.final_output
        if event.auto_image_paths:
            self.auto_image_paths = list(event.auto_image_paths)


def run_chat_turn(
    prompt: str,
    selected_files_for_agent,
    selected_image_for_chat,
    search_source: str,
    semantic_sort_by: str,
    semantic_year_filter: str,
    token_usage: dict,
    messages,
):
    chat_history_str = build_chat_history_string(messages)
    process_logs: list[str] = []
    renderer = EventRenderer(process_logs)
    tracker = TokenTracker(token_usage)
    run_config = {"callbacks": [tracker]}
    current_function = st.session_state.current_function
    current_chat_id = st.session_state.get("current_chat_id", "")

    if current_function == "d":
        draft_content, draft_name = load_reviewer_draft_content(selected_files_for_agent)
        if draft_name:
            st.info(f"已识别勾选文件为原稿：{draft_name}")
        events = stream_reviewer_flow(
            prompt=prompt,
            draft_content=draft_content,
            chat_history_str=chat_history_str,
            run_config=run_config,
        )
    else:
        events = stream_general_flow(
            prompt=prompt,
            chat_history_str=chat_history_str,
            selected_files_for_agent=selected_files_for_agent,
            selected_image_for_chat=selected_image_for_chat,
            search_source=search_source,
            semantic_sort_by=semantic_sort_by,
            semantic_year_filter=semantic_year_filter,
            current_function=current_function,
            current_chat_id=current_chat_id,
            run_config=run_config,
        )

    for event in events:
        renderer.render(event)

    return renderer.final_output, renderer.auto_image_paths, process_logs

