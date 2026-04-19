import os

import streamlit as st

from ui.chat_flows import run_chat_turn
from ui.chat_panels import (
    render_document_management_panel,
    render_memory_governance_panel,
    render_search_settings,
)
from ui.chat_renderers import (
    render_chat_messages,
    render_markdown_with_images,
)
from ui.config import FUNC_MAP, UPLOAD_DIR
from ui.session import save_chat
from utils.image_utils import append_image_gallery_to_markdown


def render_chat_page():
    func_name = FUNC_MAP.get(st.session_state.current_function, "未知")
    current_function = st.session_state.current_function
    use_side_panel = current_function == "a"
    use_document_panel = current_function in ["c", "d"]
    selected_files_for_agent = []
    selected_image_for_chat = None
    search_source = st.session_state.get("search_source", "arxiv")
    semantic_sort_by = st.session_state.get("semantic_sort_by", "relevance")

    if use_side_panel:
        chat_col, side_col = st.columns([3, 1], gap="large")
    else:
        chat_col = st.container()
        side_col = None

    with chat_col:
        st.title(f"智能科研助手 - {func_name}")
        st.caption(f"当前会话 ID: `{st.session_state.current_chat_id}`")
        render_chat_messages(st.session_state.messages)

        if use_document_panel:
            chat_upload_dir = os.path.join(UPLOAD_DIR, st.session_state.current_chat_id)
            (
                selected_files_for_agent,
                selected_image_for_chat,
            ) = render_document_management_panel(chat_upload_dir)

        render_memory_governance_panel()

    if side_col is not None:
        with side_col:
            search_source, semantic_sort_by = render_search_settings(
                search_source, semantic_sort_by
            )

    placeholder_text = "请输入你的科研需求..."
    if prompt := st.chat_input(placeholder_text):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat(st.session_state.current_chat_id, st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_container = st.empty()
            st.session_state.current_stream_container = stream_container

            with st.status("Agent 正在思考与执行...", expanded=True) as status:
                process_logs = []
                try:
                    (
                        final_output,
                        auto_image_paths,
                        process_logs,
                    ) = run_chat_turn(
                        prompt=prompt,
                        selected_files_for_agent=selected_files_for_agent,
                        selected_image_for_chat=selected_image_for_chat,
                        search_source=search_source,
                        semantic_sort_by=semantic_sort_by,
                        token_usage=st.session_state.token_usage,
                        messages=st.session_state.messages,
                    )

                    status.update(
                        label="任务执行完毕，点击查看执行详情",
                        state="complete",
                        expanded=False,
                    )
                    final_answer_display = (
                        f"### 执行结果\n{final_output}" if final_output else "未获取到有效结果。"
                    )
                    final_answer_display = append_image_gallery_to_markdown(
                        final_answer_display,
                        auto_image_paths,
                        title="**文献自动命中的相关图表**",
                    )
                    stream_container.markdown(
                        render_markdown_with_images(final_answer_display),
                        unsafe_allow_html=True,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": final_answer_display,
                            "process_logs": process_logs,
                        }
                    )
                    save_chat(st.session_state.current_chat_id, st.session_state.messages)

                except Exception as exc:
                    status.update(
                        label="执行过程中发生系统级错误",
                        state="error",
                        expanded=True,
                    )
                    error_msg = (
                        f"**系统异常终止**: {str(exc)}\n\n"
                        "*请调整提示词或检查配置后重新输入。*"
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "process_logs": process_logs,
                        }
                    )
                    save_chat(st.session_state.current_chat_id, st.session_state.messages)
