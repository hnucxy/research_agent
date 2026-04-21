import os
import shutil

import streamlit as st
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.logger import get_logger
from config.settings import Settings
from ui.config import UPLOAD_DIR
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
from utils.vectorization_tasks import (
    get_vectorization_task,
    is_vectorization_active,
    list_vectorization_tasks,
    start_vectorization_task,
)

logger = get_logger()
GLOBAL_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "_global")


def _get_library_file_path(file_hash: str, file_name: str) -> str:
    base_name = os.path.splitext(file_name)[0]
    return os.path.join(GLOBAL_UPLOAD_DIR, file_hash, f"{base_name}.md")


def _get_vector_snapshot(vectorstore: Chroma, file_hash: str) -> dict:
    try:
        return vectorstore._collection.get(
            where={"file_hash": file_hash},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.warning("检查文献向量记录失败 | file_hash=%s | error=%s", file_hash, exc)
        return {}


def _has_vector_record(snapshot: dict) -> bool:
    return bool(snapshot and snapshot.get("ids"))


def _rebuild_markdown_from_vectors(snapshot: dict) -> str:
    documents = snapshot.get("documents", []) or []
    metadatas = snapshot.get("metadatas", []) or []
    text_chunks = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if (metadata or {}).get("type") == "image":
            continue
        if document and not str(document).startswith("image://"):
            text_chunks.append(str(document).strip())
    return "\n\n".join(chunk for chunk in text_chunks if chunk)


def _copy_markdown_bundle_to_library(source_path: str, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    source_dir = os.path.dirname(os.path.abspath(source_path))
    target_dir = os.path.dirname(os.path.abspath(target_path))
    source_image_dir = os.path.join(source_dir, "images")
    target_image_dir = os.path.join(target_dir, "images")
    if os.path.isdir(source_image_dir) and source_image_dir != target_image_dir:
        shutil.copytree(source_image_dir, target_image_dir, dirs_exist_ok=True)

    with open(source_path, "r", encoding="utf-8") as file:
        text_content = file.read()

    source_rel = os.path.relpath(source_dir, os.getcwd()).replace("\\", "/")
    target_rel = os.path.relpath(target_dir, os.getcwd()).replace("\\", "/")
    source_abs = source_dir.replace("\\", "/")
    target_abs = target_dir.replace("\\", "/")
    text_content = text_content.replace(
        f"{source_rel}/images/", f"{target_rel}/images/"
    )
    text_content = text_content.replace(
        f"{source_abs}/images/",
        f"{target_abs}/images/",
    )
    with open(target_path, "w", encoding="utf-8") as file:
        file.write(text_content)
    return text_content


def _write_text_file(file_path: str, text_content: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_content)


def _mark_file_for_current_chat(file_hash: str) -> None:
    active_chat_id = st.session_state.get("current_chat_id")
    if not active_chat_id:
        return
    chat_file_hashes = st.session_state.setdefault("chat_file_hashes", {})
    current_hashes = set(chat_file_hashes.get(active_chat_id, []))
    current_hashes.add(file_hash)
    chat_file_hashes[active_chat_id] = sorted(current_hashes)


def _show_vectorization_notifications() -> None:
    notified_keys = st.session_state.setdefault("vectorization_notified_keys", [])
    notified_set = set(notified_keys)
    for task in list_vectorization_tasks():
        if not task or task.get("status") not in {"completed", "failed"}:
            continue
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

    # 为当前聊天面板中的所有上传文件准备共享资源。
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
        library_file_path = _get_library_file_path(file_hash, uploaded_file.name)

        try:
            logger.info(
                "开始处理上传文献 | file_name=%s | file_hash=%s",
                uploaded_file.name,
                file_hash,
            )

            vector_snapshot = _get_vector_snapshot(vectorstore, file_hash)
            old_path = (
                get_file_path_from_hash(file_hash)
                if is_file_duplicate(file_hash)
                else None
            )
            can_reuse_duplicate = bool(
                (old_path and os.path.exists(old_path))
                or _has_vector_record(vector_snapshot)
            )
            if can_reuse_duplicate:
                # 复用已解析的Markdown同时确认向量记录是否已经存在
                logger.info(
                    "检测到重复文献，跳过重复解析 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )
                st.warning(f"文件 `{uploaded_file.name}` 已存在于全局库中，跳过重复解析。")
                if old_path and os.path.exists(old_path):
                    text_content = _copy_markdown_bundle_to_library(
                        old_path, library_file_path
                    )
                    register_file(file_hash, library_file_path)
                elif _has_vector_record(vector_snapshot):
                    text_content = _rebuild_markdown_from_vectors(vector_snapshot)
                    if text_content:
                        _write_text_file(library_file_path, text_content)
                        register_file(file_hash, library_file_path)
                    else:
                        logger.warning(
                            "重复文献只有向量记录但无法还原文本 | file_name=%s | file_hash=%s",
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
                with open(library_file_path, "r", encoding="utf-8") as file:
                    text_content = file.read()
                _write_text_file(file_path, text_content)

                if _has_vector_record(vector_snapshot):
                    logger.info(
                        "检测到文献已存在向量记录，跳过向量化 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    progress_bar.progress(100, text=f"`{uploaded_file.name}` 处理完成")
                    _mark_file_for_current_chat(file_hash)
                    success_count += 1
                    continue
            elif file_ext == "pdf":
                progress_bar.progress(15, text=f"正在解析 `{uploaded_file.name}` 并提取图片")
                stage_notice.warning(
                    f"正在解析 `{uploaded_file.name}` 并提取图片，请勿刷新或切换页面。"
                )
                os.makedirs(os.path.dirname(library_file_path), exist_ok=True)
                with st.spinner(f"正在解析 `{uploaded_file.name}` 并提取图片..."):
                    try:
                        text_content = parse_pdf_to_markdown(
                            pdf_bytes=file_bytes,
                            output_dir=os.path.dirname(library_file_path),
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
            _write_text_file(library_file_path, text_content)
            _write_text_file(file_path, text_content)

            # PDF和Markdown都走同一套切分与入库流程
            chunks = text_splitter.split_text(text_content)
            docs_to_insert = []
            metadatas = []

            for chunk in chunks:
                docs_to_insert.append(chunk)
                metadatas.append(
                    {
                        "chat_id": active_chat_id,
                        "file_name": uploaded_file.name,
                        "file_path": os.path.abspath(library_file_path),
                        "file_hash": file_hash,
                        "type": "text",
                    }
                )
                # 将图片引用和原文片段一起入库便于后续多模态检索
                img_paths = extract_markdown_image_paths(
                    chunk, document_path=library_file_path
                )
                for img_path in img_paths:
                    if os.path.exists(img_path):
                        docs_to_insert.append(f"image://{img_path}")
                        metadatas.append(
                            {
                                "chat_id": active_chat_id,
                                "file_name": uploaded_file.name,
                                "file_path": os.path.abspath(library_file_path),
                                "file_hash": file_hash,
                                "type": "image",
                                "image_path": img_path,
                                "context": chunk,
                            }
                        )

            if docs_to_insert:
                progress_bar.progress(70, text=f"已提交 `{uploaded_file.name}` 后台向量化")
                stage_notice.info(
                    f"`{uploaded_file.name}` 已加入后台向量化队列，可继续使用当前页面。"
                )
                logger.info(
                    "开始向量化文献 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )
                try:
                    register_file(file_hash, library_file_path)
                    _mark_file_for_current_chat(file_hash)
                    if not is_vectorization_active(file_hash):
                        start_vectorization_task(
                            file_hash=file_hash,
                            file_name=uploaded_file.name,
                            texts=docs_to_insert,
                            metadatas=metadatas,
                        )
                except Exception as exc:
                    logger.exception(
                        "文献向量化失败 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    st.error(f"向量化失败: {str(exc)}")
                    continue
                logger.info(
                    "已提交后台向量化文献 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )

            register_file(file_hash, library_file_path)
            _mark_file_for_current_chat(file_hash)
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
        st.success(f"成功处理 {success_count} 份文献。")


def _build_memory_management_actor() -> str:
    current_function = st.session_state.get("current_function") or "none"
    current_chat_id = st.session_state.get("current_chat_id") or "none"
    return f"function:{current_function}|chat:{current_chat_id}"


def _render_global_document_selector():
    _show_vectorization_notifications()
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
    current_chat_hashes = set(
        st.session_state.get("chat_file_hashes", {}).get(active_chat_id, [])
    )
    for file_path, file_meta in unique_md_files.items():
        file_name = file_meta["file_name"]
        file_hash = file_meta["file_hash"]
        vector_task = get_vectorization_task(file_hash)
        vector_status = (vector_task or {}).get("status", "")
        is_processing = vector_status in {"queued", "running"}
        is_failed = vector_status == "failed"
        col1, col2 = st.columns([4, 1])
        checkbox_key = f"cb_global_{file_path}"
        if checkbox_key not in st.session_state:
            # 新会话默认勾选路径中包含当前chat_id的已上传文件
            st.session_state[checkbox_key] = bool(
                file_hash in current_chat_hashes
                or (active_chat_id and active_chat_id in file_path)
            )

        if is_processing:
            status_label = "排队中" if vector_status == "queued" else "向量化中"
            is_checked = col1.checkbox(
                f"`{file_name}`（{status_label}，暂不可选）",
                value=False,
                key=f"cb_processing_{file_hash}",
                disabled=True,
            )
        elif is_failed:
            is_checked = col1.checkbox(
                f"`{file_name}`（向量化失败，请删除后重新上传）",
                value=False,
                key=f"cb_failed_{file_hash}",
                disabled=True,
            )
        else:
            is_checked = col1.checkbox(
                f"`{file_name}`",
                value=st.session_state[checkbox_key],
                key=checkbox_key,
            )
        if is_checked:
            selected_files_for_agent.append({"name": file_name, "path": file_path})

        delete_key = f"del_global_{file_path}"
        if col2.button(
            "删除",
            key=delete_key,
            help="处理中时不可删除" if is_processing else "删除此物理文献",
            disabled=is_processing,
        ):
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
        # 只把已选Markdown文档中的图片暴露给聊天提问使用
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
    # UI 标签可能较长，这里保留到稳定entry_id的直接映射
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
