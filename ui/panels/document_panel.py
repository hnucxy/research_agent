import os

import streamlit as st

from config.logger import get_logger
from services.document_service import (
    delete_global_document,
    list_completed_vectorization_notices,
    list_global_documents,
    list_images_for_documents,
    process_uploaded_documents,
)

logger = get_logger()


def _build_memory_management_actor() -> str:
    current_function = st.session_state.get("current_function") or "none"
    current_chat_id = st.session_state.get("current_chat_id") or "none"
    return f"function:{current_function}|chat:{current_chat_id}"


def _mark_files_for_current_chat(file_hashes) -> None:
    active_chat_id = st.session_state.get("current_chat_id")
    if not active_chat_id:
        return
    chat_file_hashes = st.session_state.setdefault("chat_file_hashes", {})
    current_hashes = set(chat_file_hashes.get(active_chat_id, []))
    current_hashes.update(file_hashes)
    chat_file_hashes[active_chat_id] = sorted(current_hashes)


def _show_vectorization_notifications() -> None:
    notified_keys = st.session_state.setdefault("vectorization_notified_keys", [])
    notified_set = set(notified_keys)
    for task in list_completed_vectorization_notices():
        notice_key = (
            f"{task.get('file_hash')}:{task.get('status')}:{task.get('finished_at')}"
        )
        if notice_key in notified_set:
            continue

        file_name = task.get("file_name") or "文献"
        toast = getattr(st, "toast", None)
        if task.get("status") == "completed":
            message = f"`{file_name}` 向量化完成，现可勾选使用。"
            toast(message) if callable(toast) else st.success(message)
        else:
            message = f"`{file_name}` 向量化失败，请删除后重新上传。"
            toast(message) if callable(toast) else st.warning(message)
        notified_keys.append(notice_key)


def _render_progress_notice(placeholder, level: str, message: str) -> None:
    if level == "warning":
        placeholder.warning(message)
    elif level == "error":
        placeholder.error(message)
    else:
        placeholder.info(message)


def _handle_document_upload(chat_upload_dir: str):
    active_chat_id = st.session_state.get("current_chat_id") or "home"
    st.write("**上传文献**")
    uploaded_files = st.file_uploader(
        "上传您的文献",
        type=["md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        return

    progress_bars = {}
    stage_notices = {}

    def on_progress(event):
        if event.file_name not in progress_bars:
            progress_bars[event.file_name] = st.progress(
                event.progress or 0,
                text=event.message,
            )
            stage_notices[event.file_name] = st.empty()
        if event.progress is not None:
            progress_bars[event.file_name].progress(event.progress, text=event.message)
        _render_progress_notice(
            stage_notices[event.file_name],
            event.level,
            event.message,
        )

    with st.spinner("正在处理上传文献..."):
        summary = process_uploaded_documents(
            uploaded_files=uploaded_files,
            chat_upload_dir=chat_upload_dir,
            active_chat_id=active_chat_id,
            progress_callback=on_progress,
        )

    _mark_files_for_current_chat(summary.processed_hashes)
    for progress_bar in progress_bars.values():
        progress_bar.empty()
    for notice in stage_notices.values():
        notice.empty()

    if summary.success_count > 0:
        st.success(f"成功处理 {summary.success_count} 份文献。")


def _remove_document_checkbox_state(file_path: str, file_hash: str) -> None:
    for key in (
        f"cb_global_{file_path}",
        f"cb_processing_{file_hash}",
        f"cb_failed_{file_hash}",
    ):
        st.session_state.pop(key, None)


def _get_selected_global_documents():
    selected_files_for_agent = []
    for document in list_global_documents():
        checkbox_key = f"cb_global_{document.file_path}"
        if (
            not document.is_processing
            and not document.is_failed
            and st.session_state.get(checkbox_key)
        ):
            selected_files_for_agent.append(
                {"name": document.file_name, "path": document.file_path}
            )

    selected_image_for_chat = None
    all_images = list_images_for_documents(selected_files_for_agent)
    selected_img_opt = st.session_state.get("selected_document_image", "none")
    if selected_img_opt in all_images:
        selected_image_for_chat = selected_img_opt
    return selected_files_for_agent, selected_image_for_chat


@st.fragment
def _render_global_document_selector_fragment():
    _show_vectorization_notifications()
    documents = list_global_documents()
    selected_files_for_agent = []
    active_chat_id = st.session_state.get("current_chat_id") or ""

    if not documents:
        return

    st.write("**全局文献库**")
    current_chat_hashes = set(
        st.session_state.get("chat_file_hashes", {}).get(active_chat_id, [])
    )
    for document in documents:
        checkbox_key = f"cb_global_{document.file_path}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = bool(
                document.file_hash in current_chat_hashes
                or (active_chat_id and active_chat_id in document.file_path)
            )

        col1, col2 = st.columns([4, 1])
        if document.is_processing:
            status_label = "排队中" if document.vector_status == "queued" else "向量化中"
            is_checked = col1.checkbox(
                f"`{document.file_name}`（{status_label}，暂不可选）",
                value=False,
                key=f"cb_processing_{document.file_hash}",
                disabled=True,
            )
        elif document.is_failed:
            is_checked = col1.checkbox(
                f"`{document.file_name}`（向量化失败，请删除后重新上传）",
                value=False,
                key=f"cb_failed_{document.file_hash}",
                disabled=True,
            )
        else:
            is_checked = col1.checkbox(
                f"`{document.file_name}`",
                value=st.session_state[checkbox_key],
                key=checkbox_key,
            )
        if is_checked:
            selected_files_for_agent.append(
                {"name": document.file_name, "path": document.file_path}
            )

        if col2.button(
            "删除",
            key=f"del_global_{document.file_path}",
            help="处理中时不可删除" if document.is_processing else "删除此物理文献",
            disabled=document.is_processing,
        ):
            delete_global_document(
                file_hash=document.file_hash,
                actor=_build_memory_management_actor(),
            )
            _remove_document_checkbox_state(document.file_path, document.file_hash)
            logger.info(
                "删除全局文献及向量记录 | file_name=%s | file_hash=%s",
                document.file_name,
                document.file_hash,
            )
            st.rerun(scope="fragment")

    if selected_files_for_agent:
        all_images = list_images_for_documents(selected_files_for_agent)

        if all_images:
            st.divider()
            st.write("**文献图表多模态分析（可选）**")

            def format_img_func(img_path):
                return "不使用图表" if img_path == "none" else os.path.basename(img_path)

            img_options = ["none"] + all_images
            if st.session_state.get("selected_document_image", "none") not in img_options:
                st.session_state.selected_document_image = "none"
            selected_img_opt = st.selectbox(
                "选择一张图表结合提问：",
                img_options,
                format_func=format_img_func,
                key="selected_document_image",
            )

            if selected_img_opt != "none":
                st.image(
                    selected_img_opt,
                    caption=f"选中图表: {os.path.basename(selected_img_opt)}",
                    width="stretch",
                )


def render_document_management_panel(chat_upload_dir: str):
    with st.expander("文献上传与管理", expanded=False):
        upload_col, library_col = st.columns(2, gap="large")

        with upload_col:
            _handle_document_upload(chat_upload_dir)

        with library_col:
            _render_global_document_selector_fragment()
            return _get_selected_global_documents()

    return [], None
