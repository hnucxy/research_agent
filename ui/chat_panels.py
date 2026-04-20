import os
import shutil

import streamlit as st
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.logger import get_logger
from config.settings import Settings
from utils.document_parser import parse_pdf_to_markdown
from utils.exceptions import DocumentParseError
from utils.file_utils import (
    get_all_registered_files,
    get_file_hash,
    get_file_path_from_hash,
    is_file_duplicate,
    register_file,
)
from utils.image_utils import extract_markdown_image_paths
from utils.memory_management import (
    MEMORY_COLLECTION_OPTIONS,
    delete_memory_entries,
    get_memory_audit_log_path,
    get_memory_stats,
    list_memory_entries,
)

logger = get_logger()


def render_search_settings(search_source: str, semantic_sort_by: str) -> tuple[str, str]:
    with st.expander("搜索设置", expanded=True):
        source_options = {
            "arXiv API": "arxiv",
            "Semantic Scholar API": "semantic_scholar",
        }
        selected_source_label = st.radio(
            "选择文献检索数据源",
            options=list(source_options.keys()),
            index=0 if search_source == "arxiv" else 1,
        )
        search_source = source_options[selected_source_label]
        st.session_state.search_source = search_source

        if search_source == "semantic_scholar":
            semantic_sort_options = {
                "Sort by relevance": "relevance",
                "Sort by citation count": "citation_count",
                "Sort by most influential papers": "most_influential",
                "Sort by recency": "recency",
            }
            current_sort_index = (
                list(semantic_sort_options.values()).index(semantic_sort_by)
                if semantic_sort_by in semantic_sort_options.values()
                else 0
            )
            selected_sort_label = st.selectbox(
                "Semantic Scholar 排序方式",
                options=list(semantic_sort_options.keys()),
                index=current_sort_index,
            )
            semantic_sort_by = semantic_sort_options[selected_sort_label]
            st.session_state.semantic_sort_by = semantic_sort_by
        else:
            semantic_sort_by = st.session_state.get("semantic_sort_by", "relevance")

    return search_source, semantic_sort_by


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

    os.makedirs(chat_upload_dir, exist_ok=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = Settings.get_embeddings()
    research_collection = Settings.get_collection_name("global_research_knowledge")
    vectorstore = Chroma(
        collection_name=research_collection,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    success_count = 0

    for uploaded_file in uploaded_files:
        progress_bar = st.progress(0, text=f"准备处理 `{uploaded_file.name}`")
        stage_notice = st.empty()
        file_bytes = uploaded_file.getvalue()
        file_hash = get_file_hash(file_bytes)
        file_ext = uploaded_file.name.split(".")[-1].lower()
        base_name = os.path.splitext(uploaded_file.name)[0]
        file_path = os.path.join(chat_upload_dir, f"{base_name}.md")

        try:
            logger.info(
                "开始处理上传文献 | file_name=%s | file_hash=%s",
                uploaded_file.name,
                file_hash,
            )

            if is_file_duplicate(file_hash):
                logger.info(
                    "检测到重复文献，跳过重复解析 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )
                st.warning(f"文件 `{uploaded_file.name}` 已存在于全局库中，跳过重复解析。")
                old_path = get_file_path_from_hash(file_hash)
                if not old_path or not os.path.exists(old_path):
                    logger.warning(
                        "重复文献缺少已注册文件，无法继续处理 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    continue

                progress_bar.progress(
                    20, text=f"检测到重复文献，正在复用 `{uploaded_file.name}` 的解析结果"
                )
                stage_notice.warning(
                    f"正在复用 `{uploaded_file.name}` 的解析结果，并检查是否需要补做向量化，请勿刷新页面。"
                )
                old_path_abs = os.path.abspath(old_path)
                file_path_abs = os.path.abspath(file_path)
                if old_path_abs != file_path_abs:
                    shutil.copy2(old_path, file_path)
                with open(old_path, "r", encoding="utf-8") as file:
                    text_content = file.read()

                existing = vectorstore._collection.get(
                    where={"file_hash": file_hash}, limit=1
                )
                if existing and existing.get("ids"):
                    logger.info(
                        "检测到文献已存在向量记录，跳过向量化 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    continue
            elif file_ext == "pdf":
                progress_bar.progress(15, text=f"正在解析 `{uploaded_file.name}` 并提取图片")
                stage_notice.warning(
                    f"正在解析 `{uploaded_file.name}` 并提取图片，请勿刷新或切换页面。"
                )
                with st.spinner(f"正在解析 `{uploaded_file.name}` 并提取图片..."):
                    try:
                        text_content = parse_pdf_to_markdown(
                            pdf_bytes=file_bytes,
                            output_dir=chat_upload_dir,
                            base_name=base_name,
                            file_name=uploaded_file.name,
                            file_hash=file_hash,
                        )
                    except DocumentParseError as exc:
                        logger.warning(
                            "PDF解析失败 | file_name=%s | file_hash=%s",
                            uploaded_file.name,
                            file_hash,
                        )
                        st.error(f"解析被跳过: {str(exc)}")
                        continue
                    except Exception as exc:
                        logger.exception(
                            "PDF解析出现未知错误 | file_name=%s | file_hash=%s",
                            uploaded_file.name,
                            file_hash,
                        )
                        st.error(f"未知错误导致解析失败: {str(exc)}")
                        continue
            else:
                progress_bar.progress(15, text=f"正在读取 `{uploaded_file.name}`")
                stage_notice.info(f"正在读取 `{uploaded_file.name}` 并准备向量化，请稍候。")
                logger.info(
                    "开始处理 Markdown 文献 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )
                text_content = file_bytes.decode("utf-8", errors="ignore")

            progress_bar.progress(45, text=f"正在整理 `{uploaded_file.name}` 的文本内容")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_content)

            chunks = text_splitter.split_text(text_content)
            docs_to_insert = []
            metadatas = []

            for chunk in chunks:
                docs_to_insert.append(chunk)
                metadatas.append(
                    {
                        "chat_id": active_chat_id,
                        "file_name": uploaded_file.name,
                        "file_path": os.path.abspath(file_path),
                        "file_hash": file_hash,
                        "type": "text",
                    }
                )
                img_paths = extract_markdown_image_paths(chunk, document_path=file_path)
                for img_path in img_paths:
                    if os.path.exists(img_path):
                        docs_to_insert.append(f"image://{img_path}")
                        metadatas.append(
                            {
                                "chat_id": active_chat_id,
                                "file_name": uploaded_file.name,
                                "file_path": os.path.abspath(file_path),
                                "file_hash": file_hash,
                                "type": "image",
                                "image_path": img_path,
                                "context": chunk,
                            }
                        )

            if docs_to_insert:
                progress_bar.progress(70, text=f"正在向量化 `{uploaded_file.name}`")
                stage_notice.warning(
                    f"已完成解析，正在向量化 `{uploaded_file.name}`。这一步可能耗时较长，界面短暂空转属于正常现象。"
                )
                logger.info(
                    "开始向量化文献 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )
                try:
                    vectorstore.add_texts(texts=docs_to_insert, metadatas=metadatas)
                except Exception as exc:
                    logger.exception(
                        "文献向量化失败 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    st.error(f"向量化失败: {str(exc)}")
                    continue
                logger.info(
                    "完成向量化文献 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )

            register_file(file_hash, file_path)
            logger.info(
                "完成上传文献处理 | file_name=%s | file_hash=%s",
                uploaded_file.name,
                file_hash,
            )
            progress_bar.progress(100, text=f"`{uploaded_file.name}` 处理完成")
            success_count += 1
        finally:
            progress_bar.empty()
            stage_notice.empty()

    if success_count > 0:
        st.success(f"成功上传、解析并向量化 {success_count} 份文献。")


def _build_memory_management_actor() -> str:
    current_function = st.session_state.get("current_function") or "none"
    current_chat_id = st.session_state.get("current_chat_id") or "none"
    return f"function:{current_function}|chat:{current_chat_id}"


def _render_global_document_selector():
    global_files = get_all_registered_files()
    selected_files_for_agent = []
    active_chat_id = st.session_state.get("current_chat_id") or ""
    unique_md_files = {}
    for file_hash, file_path in global_files.items():
        if file_path.endswith(".md") and os.path.exists(file_path):
            unique_md_files[file_path] = {
                "file_name": os.path.basename(file_path),
                "file_hash": file_hash,
            }

    if not unique_md_files:
        return selected_files_for_agent, None

    st.write("**全局文献库**")
    for file_path, file_meta in unique_md_files.items():
        file_name = file_meta["file_name"]
        file_hash = file_meta["file_hash"]
        col1, col2 = st.columns([4, 1])
        checkbox_key = f"cb_global_{file_path}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = bool(
                active_chat_id and active_chat_id in file_path
            )

        is_checked = col1.checkbox(
            f"`{file_name}`", value=st.session_state[checkbox_key], key=checkbox_key
        )
        if is_checked:
            selected_files_for_agent.append({"name": file_name, "path": file_path})

        delete_key = f"del_global_{file_path}"
        if col2.button("删除", key=delete_key, help="删除此物理文献"):
            delete_memory_entries(
                collection_key="research",
                entry_ids=[file_hash],
                actor=_build_memory_management_actor(),
                action="single_delete_document_selector",
            )
            logger.info(
                "删除全局文献及向量记录 | file_name=%s | file_hash=%s",
                file_name,
                file_hash,
            )
            st.rerun()

    selected_image_for_chat = None
    if selected_files_for_agent:
        all_images = []
        for file_meta in selected_files_for_agent:
            try:
                with open(file_meta["path"], "r", encoding="utf-8") as file:
                    md_content = file.read()
                img_paths = extract_markdown_image_paths(
                    md_content, document_path=file_meta["path"]
                )
                for img in img_paths:
                    if os.path.exists(img) and img not in all_images:
                        all_images.append(img)
            except Exception:
                continue

        if all_images:
            st.divider()
            st.write("**文献图表多模态分析（可选）**")

            def format_img_func(img_path):
                return "不使用图表" if img_path == "none" else os.path.basename(img_path)

            img_options = ["none"] + all_images
            selected_img_opt = st.selectbox(
                "选择一张图表结合提问：",
                img_options,
                format_func=format_img_func,
            )

            if selected_img_opt != "none":
                st.image(
                    selected_img_opt,
                    caption=f"选中图表: {os.path.basename(selected_img_opt)}",
                    width="stretch",
                )
                selected_image_for_chat = selected_img_opt

    return selected_files_for_agent, selected_image_for_chat


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


def _render_memory_cleanup_tab():
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
    actor = _build_memory_management_actor()

    st.write("**单条清理**")
    single_target = st.selectbox(
        "选择单条记录",
        options=[""] + option_labels,
        key="memory_single_delete_target",
    )
    if st.button("删除该条记录", key="memory_single_delete_button"):
        if not single_target:
            st.warning("请先选择一条记录。")
        else:
            result = delete_memory_entries(
                collection_key=collection_key,
                entry_ids=[option_to_id[single_target]],
                actor=actor,
                action="single_delete",
            )
            st.success(
                f"已删除 {result['deleted_entries']} 条记录，影响 {result['deleted_vectors']} 条向量。"
            )
            st.rerun()

    st.write("**批量清理**")
    batch_targets = st.multiselect(
        "选择多条记录",
        options=option_labels,
        key="memory_batch_delete_targets",
    )
    if st.button(
        "批量删除选中记录",
        key="memory_batch_delete_button",
        type="primary",
    ):
        if not batch_targets:
            st.warning("请至少选择一条记录。")
        else:
            result = delete_memory_entries(
                collection_key=collection_key,
                entry_ids=[option_to_id[label] for label in batch_targets],
                actor=actor,
                action="batch_delete",
            )
            st.success(
                f"已删除 {result['deleted_entries']} 条记录，影响 {result['deleted_vectors']} 条向量。"
            )
            st.rerun()


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


def render_document_management_panel(chat_upload_dir: str):
    with st.expander("文献上传与管理", expanded=False):
        upload_col, library_col = st.columns(2, gap="large")

        with upload_col:
            _handle_document_upload(chat_upload_dir)

        with library_col:
            return _render_global_document_selector()

    return [], None
