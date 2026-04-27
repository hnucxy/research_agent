import streamlit as st

from utils.memory_management import (
    MEMORY_COLLECTION_OPTIONS,
    delete_memory_entries,
    get_memory_audit_log_path,
    get_memory_stats,
    list_memory_entries,
)


def _build_memory_management_actor() -> str:
    current_function = st.session_state.get("current_function") or "none"
    current_chat_id = st.session_state.get("current_chat_id") or "none"
    return f"function:{current_function}|chat:{current_chat_id}"


def _format_memory_entry_label(collection_key: str, row: dict) -> str:
    entry_id = row.get("entry_id", "")
    if collection_key == "research":
        file_name = row.get("文件名", "")
        return f"{file_name or '未命名文献'} | {entry_id}"
    if collection_key == "failure":
        failure_type = row.get("失败类型", "")
        preview = row.get("内容预览", "")
        return f"{failure_type or 'unknown'} | {entry_id} | {preview}"
    preview = row.get("内容预览", "")
    return f"{entry_id} | {preview}"


def _render_memory_overview_tab():
    collection_key = st.selectbox(
        "选择记忆库",
        options=list(MEMORY_COLLECTION_OPTIONS.keys()),
        format_func=lambda key: MEMORY_COLLECTION_OPTIONS[key],
        key="memory_view_collection",
    )
    rows = list_memory_entries(collection_key)
    st.caption("最多展示 50 条记录。")
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("当前记忆库暂无记录。")


def _save_cleanup_result(result: dict) -> None:
    st.session_state.memory_cleanup_last_message = (
        f"已删除 {result['deleted_entries']} 条记录，"
        f"影响 {result['deleted_vectors']} 条向量。"
    )


def _delete_single_cleanup_target(collection_key: str, actor: str) -> None:
    single_target = st.session_state.get("memory_single_delete_target")
    option_to_id = st.session_state.get("memory_cleanup_option_to_id", {})
    if not single_target:
        st.session_state.memory_cleanup_last_warning = "请先选择一条记录。"
        return

    result = delete_memory_entries(
        collection_key=collection_key,
        entry_ids=[option_to_id[single_target]],
        actor=actor,
        action="single_delete",
    )
    _save_cleanup_result(result)
    st.session_state.memory_single_delete_target = ""


def _delete_batch_cleanup_targets(collection_key: str, actor: str) -> None:
    batch_targets = st.session_state.get("memory_batch_delete_targets", [])
    option_to_id = st.session_state.get("memory_cleanup_option_to_id", {})
    if not batch_targets:
        st.session_state.memory_cleanup_last_warning = "请至少选择一条记录。"
        return

    result = delete_memory_entries(
        collection_key=collection_key,
        entry_ids=[option_to_id[label] for label in batch_targets],
        actor=actor,
        action="batch_delete",
    )
    _save_cleanup_result(result)
    st.session_state.memory_batch_delete_targets = []


@st.fragment
def _render_memory_cleanup_tab():
    message = st.session_state.pop("memory_cleanup_last_message", "")
    warning = st.session_state.pop("memory_cleanup_last_warning", "")
    if message:
        st.success(message)
    if warning:
        st.warning(warning)

    collection_key = st.selectbox(
        "选择要清理的记忆库",
        options=list(MEMORY_COLLECTION_OPTIONS.keys()),
        format_func=lambda key: MEMORY_COLLECTION_OPTIONS[key],
        key="memory_cleanup_collection",
    )
    rows = list_memory_entries(collection_key)
    if not rows:
        st.info("当前记忆库暂无可清理记录。")
        return

    st.dataframe(rows, width="stretch", hide_index=True)
    option_labels = [_format_memory_entry_label(collection_key, row) for row in rows]
    option_to_id = {
        _format_memory_entry_label(collection_key, row): row["entry_id"] for row in rows
    }
    st.session_state.memory_cleanup_option_to_id = option_to_id
    actor = _build_memory_management_actor()

    st.write("**单条清理**")
    st.selectbox(
        "选择单条记录",
        options=[""] + option_labels,
        key="memory_single_delete_target",
    )
    st.button(
        "删除该条记录",
        key="memory_single_delete_button",
        on_click=_delete_single_cleanup_target,
        args=(collection_key, actor),
    )

    st.write("**批量清理**")
    st.multiselect(
        "选择多条记录",
        options=option_labels,
        key="memory_batch_delete_targets",
    )
    st.button(
        "批量删除选中记录",
        key="memory_batch_delete_button",
        type="primary",
        on_click=_delete_batch_cleanup_targets,
        args=(collection_key, actor),
    )


def _render_memory_stats_tab():
    stats = get_memory_stats()
    collection_stats = stats.get("collections", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("向量总数", stats.get("overall_total_vectors", 0))
    col2.metric(
        "成功经验数",
        collection_stats.get("experience", {}).get("total_vectors", 0),
    )
    col3.metric(
        "失败经验数",
        collection_stats.get("failure", {}).get("total_vectors", 0),
    )
    col4.metric(
        "文献文件数",
        collection_stats.get("research", {}).get("document_count", 0),
    )

    experience_stats = collection_stats.get("experience", {})
    failure_stats = collection_stats.get("failure", {})
    research_stats = collection_stats.get("research", {})

    with st.container(border=True):
        st.write("**成功经验库**")
        st.write(f"记录数：{experience_stats.get('total_vectors', 0)}")

    with st.container(border=True):
        st.write("**失败经验库**")
        st.write(f"记录数：{failure_stats.get('total_vectors', 0)}")
        breakdown = failure_stats.get("failure_breakdown", [])
        if breakdown:
            st.dataframe(breakdown, width="stretch", hide_index=True)

    with st.container(border=True):
        st.write("**文献知识库**")
        st.write(f"文件数：{research_stats.get('document_count', 0)}")
        st.write(f"文本块数：{research_stats.get('text_chunk_count', 0)}")
        st.write(f"图片块数：{research_stats.get('image_chunk_count', 0)}")
        st.write(f"向量总数：{research_stats.get('total_vectors', 0)}")

    st.caption(f"审计日志写入路径：`{get_memory_audit_log_path()}`")


def render_memory_management_panel():
    with st.expander("记忆管理", expanded=False):
        overview_tab, cleanup_tab, stats_tab = st.tabs(
            ["可视化查看", "清理", "统计面板"]
        )
        with overview_tab:
            _render_memory_overview_tab()
        with cleanup_tab:
            _render_memory_cleanup_tab()
        with stats_tab:
            _render_memory_stats_tab()

