import base64
import os
import re
import shutil

import streamlit as st
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.logger import get_logger
from config.settings import Settings
from graph.graph_builder import build_graph
from ui.config import FUNC_MAP, UPLOAD_DIR
from ui.session import save_chat
from utils.document_parser import parse_pdf_to_markdown
from utils.exceptions import DocumentParseError
from utils.file_utils import (
    get_all_registered_files,
    get_file_hash,
    get_file_path_from_hash,
    is_file_duplicate,
    register_file,
)
from utils.image_utils import (
    append_image_gallery_to_markdown,
    extract_markdown_image_paths,
    resolve_local_image_path,
)
from utils.memory_governance import (
    MEMORY_COLLECTION_OPTIONS,
    delete_memory_entries,
    get_memory_audit_log_path,
    get_memory_stats,
    list_memory_entries,
)
from utils.prompt_utils import build_resource_context
from utils.token_tracker import TokenTracker

logger = get_logger()


def render_markdown_with_images(text: str) -> str:
    if not text:
        return text

    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

    def replace_image(match):
        alt_text = match.group(1)
        img_path = resolve_local_image_path(match.group(2))
        if img_path and not img_path.startswith("http") and os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = img_path.split(".")[-1].lower()
                mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
                return f"![{alt_text}](data:{mime};base64,{b64})"
            except Exception:
                return match.group(0)
        return match.group(0)

    return re.sub(pattern, replace_image, text)


def render_rich_markdown(text: str):
    st.markdown(render_markdown_with_images(text), unsafe_allow_html=True)


def _render_search_settings(search_source: str, semantic_sort_by: str) -> tuple[str, str]:
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
                with open(old_path, "r", encoding="utf-8") as f:
                    text_content = f.read()

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
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            chunks = text_splitter.split_text(text_content)
            docs_to_insert = []
            metadatas = []

            for chunk in chunks:
                docs_to_insert.append(chunk)
                metadatas.append(
                    {
                        "chat_id": st.session_state.current_chat_id,
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
                                "chat_id": st.session_state.current_chat_id,
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


def _render_global_document_selector():
    global_files = get_all_registered_files()
    selected_files_for_agent = []
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
            st.session_state[checkbox_key] = (
                st.session_state.current_chat_id in file_path
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
                actor=_build_memory_governance_actor(),
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
                with open(file_meta["path"], "r", encoding="utf-8") as f:
                    md_content = f.read()
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


def _build_memory_governance_actor() -> str:
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
        st.dataframe(rows, use_container_width=True, hide_index=True)
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

    st.dataframe(rows, use_container_width=True, hide_index=True)
    option_labels = [_format_memory_entry_label(collection_key, row) for row in rows]
    option_to_id = {
        _format_memory_entry_label(collection_key, row): row["entry_id"] for row in rows
    }
    actor = _build_memory_governance_actor()

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
    if st.button("批量删除选中记录", key="memory_batch_delete_button", type="primary"):
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
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.write("**文献知识库**")
        st.write(f"文件数：{research_stats.get('document_count', 0)}")
        st.write(f"文本块数：{research_stats.get('text_chunk_count', 0)}")
        st.write(f"图片块数：{research_stats.get('image_chunk_count', 0)}")
        st.write(f"向量总数：{research_stats.get('total_vectors', 0)}")

    st.caption(f"审计日志写入路径：`{get_memory_audit_log_path()}`")


def _render_memory_governance_panel():
    with st.expander("记忆治理", expanded=False):
        overview_tab, cleanup_tab, stats_tab = st.tabs(
            ["可视化查看", "清理", "统计面板"]
        )
        with overview_tab:
            _render_memory_overview_tab()
        with cleanup_tab:
            _render_memory_cleanup_tab()
        with stats_tab:
            _render_memory_stats_tab()


def _handle_reviewer_flow(
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


def _handle_general_flow(
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


def _render_document_management_panel(chat_upload_dir: str):
    with st.expander("文献上传与管理", expanded=False):
        upload_col, library_col = st.columns(2, gap="large")

        with upload_col:
            _handle_document_upload(chat_upload_dir)

        with library_col:
            return _render_global_document_selector()

    return [], None


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
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "process_logs" in msg:
                    with st.expander("查看历史执行过程"):
                        for log in msg["process_logs"]:
                            render_rich_markdown(log)
                render_rich_markdown(msg["content"])

        if use_document_panel:
            chat_upload_dir = os.path.join(UPLOAD_DIR, st.session_state.current_chat_id)
            (
                selected_files_for_agent,
                selected_image_for_chat,
            ) = _render_document_management_panel(chat_upload_dir)

        _render_memory_governance_panel()

    if side_col is not None:
        with side_col:
            search_source, semantic_sort_by = _render_search_settings(
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
                    history_msgs = []
                    for msg in st.session_state.messages[:-1]:
                        role_name = "用户" if msg["role"] == "user" else "助手"
                        history_msgs.append(f"[{role_name}]: {msg.get('content', '')}")
                    chat_history_str = "\n".join(history_msgs) if history_msgs else "无历史对话"

                    tracker = TokenTracker(st.session_state.token_usage)
                    run_config = {"callbacks": [tracker]}
                    final_output = ""
                    auto_image_paths = []

                    if st.session_state.current_function == "d":
                        draft_content = ""
                        if selected_files_for_agent:
                            file_path = selected_files_for_agent[0]["path"]
                            if file_path.endswith(".md") and os.path.exists(file_path):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    draft_content = f.read()
                                st.info(
                                    f"已识别勾选文件为原稿：{selected_files_for_agent[0]['name']}"
                                )

                        final_output = _handle_reviewer_flow(
                            prompt=prompt,
                            draft_content=draft_content,
                            chat_history_str=chat_history_str,
                            run_config=run_config,
                            process_logs=process_logs,
                        )
                    else:
                        final_output, auto_image_paths = _handle_general_flow(
                            prompt=prompt,
                            chat_history_str=chat_history_str,
                            selected_files_for_agent=selected_files_for_agent,
                            selected_image_for_chat=selected_image_for_chat,
                            search_source=search_source,
                            semantic_sort_by=semantic_sort_by,
                            run_config=run_config,
                            process_logs=process_logs,
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
