import streamlit as st
import os
import re
import base64
import shutil  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config.settings import Settings
from graph.graph_builder import build_graph
from ui.config import UPLOAD_DIR, FUNC_MAP
from ui.session import save_chat
from utils.file_utils import get_file_hash, is_file_duplicate, register_file, get_file_path_from_hash, get_all_registered_files
from utils.document_parser import parse_pdf_to_markdown
from utils.exceptions import DocumentParseError
from utils.token_tracker import TokenTracker
from utils.prompt_utils import build_resource_context

def render_markdown_with_images(text: str) -> str:
    """提取 Markdown 中的本地路径并将其替换为 Base64 以供 Streamlit 渲染"""
    if not text:
        return text
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        if not img_path.startswith("http") and os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = img_path.split('.')[-1].lower()
                mime = "image/jpeg" if ext in ['jpg', 'jpeg'] else f"image/{ext}"
                return f"![{alt_text}](data:{mime};base64,{b64})"
            except Exception:
                pass
        return match.group(0)
    return re.sub(pattern, replace_image, text)

def render_chat_page():
    func_name = FUNC_MAP.get(st.session_state.current_function, "未知")
    chat_col, doc_col = st.columns([3, 1], gap="large")
    selected_files_for_agent = []
    selected_image_for_chat = None
    search_source = st.session_state.get("search_source", "arxiv")
    semantic_sort_by = st.session_state.get("semantic_sort_by", "relevance")

    # 左侧：聊天主界面
    with chat_col:
        st.title(f"🤖 智能科研助手 - {func_name}")
        st.caption(f"当前会话 ID: `{st.session_state.current_chat_id}`")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "process_logs" in msg:
                    with st.expander("查看历史执行过程"):
                        for log in msg["process_logs"]:
                            st.markdown(render_markdown_with_images(log))
                st.markdown(render_markdown_with_images(msg["content"]))

    # 右侧：文献库与管理
    with doc_col:
        if st.session_state.current_function == "a":
            with st.expander("🔎 检索设置", expanded=True):
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
                    semantic_sort_by = st.session_state.get(
                        "semantic_sort_by", "relevance"
                    )

        elif st.session_state.current_function in ["c", "d"]:
            chat_upload_dir = os.path.join(UPLOAD_DIR, st.session_state.current_chat_id)
            
            with st.expander("📁 文献上传与管理", expanded=False):
                st.write("📄 **上传文献**")
                uploaded_files = st.file_uploader(
                    "上传您的文献", 
                    type=["md", "pdf"], 
                    accept_multiple_files=True, 
                    label_visibility="collapsed"
                )
                
                if uploaded_files:
                    os.makedirs(chat_upload_dir, exist_ok=True)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    embeddings = Settings.get_embeddings()
                    research_collection = Settings.get_collection_name("global_research_knowledge")
                    vectorstore = Chroma(
                        collection_name=research_collection,
                        embedding_function=embeddings,
                        persist_directory="./chroma_db"
                    )
                    success_count = 0

                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.getvalue()
                        file_hash = get_file_hash(file_bytes)
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        file_path = os.path.join(chat_upload_dir, f"{base_name}.md")
                        
                        # 防溢出复用机制
                        if is_file_duplicate(file_hash):
                            st.warning(f"⚠️ 文件 `{uploaded_file.name}` 已存在于全局库中，跳过重复解析。")
                            # 获取过往的真实路径并复制到当前会话
                            old_path = get_file_path_from_hash(file_hash)
                            if old_path and os.path.exists(old_path):
                                shutil.copy2(old_path, file_path)
                                with open(old_path, "r", encoding="utf-8") as f:
                                    text_content = f.read()

                                existing = vectorstore._collection.get(
                                    where={"file_hash": file_hash},
                                    limit=1
                                )
                                if existing and existing.get("ids"):
                                    continue
                            else:
                                continue

                        # 新文件解析与存入 Chroma 逻辑保持不变...
                        elif file_ext == "pdf":
                            with st.spinner(f"正在解析 `{uploaded_file.name}` 并提取图片..."):
                                try:
                                    text_content = parse_pdf_to_markdown(file_bytes, chat_upload_dir, base_name)
                                except DocumentParseError as e:
                                    st.error(f"解析被跳过: {str(e)}")
                                    continue  # 抛出异常后跳过当前文件，继续处理用户上传的下一个文件
                                except Exception as e:
                                    st.error(f"未知错误导致解析失败: {str(e)}")
                                    continue
                        else:
                            text_content = file_bytes.decode("utf-8", errors="ignore")
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(text_content)
                        
                        chunks = text_splitter.split_text(text_content)
                        docs_to_insert = []
                        metadatas = []

                        for chunk in chunks:
                            docs_to_insert.append(chunk)
                            metadatas.append({
                                "chat_id": st.session_state.current_chat_id, 
                                "file_name": uploaded_file.name,
                                "file_hash": file_hash,
                                "type": "text"
                            })
                            img_paths = re.findall(r'!\[.*?\]\((.*?)\)', chunk)
                            for img_path in img_paths:
                                if os.path.exists(img_path):
                                    docs_to_insert.append(f"image://{img_path}")
                                    metadatas.append({
                                        "chat_id": st.session_state.current_chat_id,
                                        "file_name": uploaded_file.name,
                                        "file_hash": file_hash,
                                        "type": "image",
                                        "image_path": img_path,
                                        "context": chunk
                                    })
                        
                        if docs_to_insert:
                            vectorstore.add_texts(
                                texts=docs_to_insert,
                                metadatas=metadatas,
                            )
                        register_file(file_hash, file_path)
                        success_count += 1
                        
                    if success_count > 0:
                        st.success(f"成功上传、解析并向量化 {success_count} 份文献！")
                
                st.divider()
                
                # 渲染带有 Checkbox 的文献库
                global_files = get_all_registered_files()
                selected_files_for_agent = []  # 收集当前被勾选的文献信息 [{"name": "xxx", "path": "xxx"}]
                
                # 提取所有存在的 md 文件路径 (使用字典去重，防止 hash 不同但路径相同的极端情况)
                unique_md_files = {} 
                for f_hash, f_path in global_files.items():
                    if f_path.endswith('.md') and os.path.exists(f_path):
                        unique_md_files[f_path] = os.path.basename(f_path)
                
                if unique_md_files:
                    st.write("📚 **全局文献库 (请勾选需要分析的文献)**")
                    for f_path, f_name in unique_md_files.items():
                        col1, col2 = st.columns([4, 1])
                        
                        # 利用 session_state 保存每个复选框的勾选状态，采用文件路径作为唯一 key
                        cb_key = f"cb_global_{f_path}"
                        if cb_key not in st.session_state:
                            # 如果该文献属于当前会话上传的，默认勾选；否则默认不勾选，避免全局文件过多污染上下文
                            is_in_current_chat = st.session_state.current_chat_id in f_path
                            st.session_state[cb_key] = is_in_current_chat
                        
                        is_checked = col1.checkbox(f"`{f_name}`", value=st.session_state[cb_key], key=cb_key)
                        if is_checked:
                            selected_files_for_agent.append({"name": f_name, "path": f_path})
                        
                        btn_key = f"del_global_{f_path}"
                        if col2.button("🗑️", key=btn_key, help="删除此物理文献"):
                            # 从物理硬盘中删除文献
                            if os.path.exists(f_path):
                                os.remove(f_path)
                            st.rerun()
                    selected_image_for_chat = None
                if selected_files_for_agent:
                    all_images = []
                    for f in selected_files_for_agent:
                        try:
                            with open(f["path"], "r", encoding="utf-8") as file:
                                md_content = file.read()
                                # 正则提取文献中通过 pymupdf 解析出的图片绝对路径
                                imgs = re.findall(r'!\[.*?\]\((.*?)\)', md_content)
                                for img in imgs:
                                    if os.path.exists(img) and img not in all_images:
                                        all_images.append(img)
                        except Exception:
                            pass

                    if all_images:
                        st.divider()
                        st.write("🖼️ **文献图表多模态分析 (可选)**")
                        
                        # 格式化函数：仅显示文件名，不显示冗长路径
                        def format_img_func(img_path):
                            if img_path == "无": return "不使用图表"
                            return os.path.basename(img_path)
                            
                        img_options = ["无"] + all_images
                        selected_img_opt = st.selectbox("选择一张图表结合提问：", img_options, format_func=format_img_func)
                        
                        if selected_img_opt != "无":
                            st.image(selected_img_opt, caption=f"选中图表: {os.path.basename(selected_img_opt)}", width="stretch")
                            selected_image_for_chat = selected_img_opt
    # 底部聊天输入框
    placeholder_text = "请输入你的科研需求..."
    if prompt := st.chat_input(placeholder_text):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat(st.session_state.current_chat_id, st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 在系统执行日志外，提供一个用于全局捕获流式输出的容器
            stream_container = st.empty()
            st.session_state.current_stream_container = stream_container

            with st.status("Agent 正在思考与执行...", expanded=True) as status:
                try:
                    history_msgs = []
                    for m in st.session_state.messages[:-1]:
                        role_name = "用户" if m["role"] == "user" else "助手"
                        content = m.get("content", "")
                        history_msgs.append(f"[{role_name}]: {content}")

                    chat_history_str = "\n".join(history_msgs) if history_msgs else "无历史对话"
                    
                    tracker = TokenTracker(st.session_state.token_usage)
                    run_config = {"callbacks": [tracker]}
                    process_logs = []
                    final_output = ""

                    # 功能四 (Author-Reviewer
                    if st.session_state.current_function == "d":
                        # 复用文件上传机制提取原稿内容
                        draft_content = ""
                        if selected_files_for_agent:
                            # 假设只取勾选的第一个文件作为原稿
                            file_path = selected_files_for_agent[0]["path"]
                            if file_path.endswith('.md') and os.path.exists(file_path):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    draft_content = f.read()
                                st.info(f"已识别到作为原稿的勾选文件：{selected_files_for_agent[0]['name']}")

                        from graph.graph_builder import build_reviewer_graph
                        agent_app = build_reviewer_graph()
                        initial_state = {
                            "current_function": "d",
                            "user_prompt": prompt,
                            "draft_content": draft_content,
                            "feedback": "",
                            "retry_count": 0,
                            "max_retries": 3,
                            "status": "",
                            "chat_history": chat_history_str,
                            "final_answer": ""
                        }

                        # 循环并适配界面输出
                        for output in agent_app.stream(initial_state, config=run_config):
                            for node_name, state_update in output.items():
                                if node_name == "input_parser":
                                    # 更新当前草稿内容，防止后续 Reviewer 拿不到
                                    if "draft_content" in state_update:
                                        draft_content = state_update["draft_content"]
                                    st.write("🧩 **解析器** 准备就绪。")

                                elif node_name == "reviewer":
                                    status_flag = state_update.get('status')
                                    fb = state_update.get('feedback', '')
                                    # 确保从 state_update 或闭包变量中获取最新的稿件
                                    current_manuscript = state_update.get('draft_content', draft_content)
                                    
                                    if status_flag == "pass":
                                        log_msg = f"✅ **Reviewer (通过)**: {fb}"
                                        st.success(log_msg)
                                        # 最终文稿输出 bug 修复：组装最终显示内容
                                        final_output = f"### ✨ 最终审定文稿\n\n{current_manuscript}\n\n---\n**审稿专家意见**：{fb}"
                                    elif status_flag == "reject":
                                        log_msg = f"⛔ **Reviewer (驳回)**: {fb}"
                                        st.error(log_msg)
                                        final_output = f"### ❌ 任务被驳回\n\n**理由**：{fb}"
                                    else:
                                        log_msg = f"⚠️ **Reviewer (反馈建议)**: {fb}"
                                        st.warning(log_msg)
                                    
                                    process_logs.append(log_msg)

                                elif node_name == "author":
                                    rc = state_update.get("retry_count", 0)
                                    # 更新闭包变量中的稿件内容，供下一轮 Reviewer 使用
                                    draft_content = state_update.get("draft_content", "")
                                    
                                    log_msg = f"✍️ **Author**: 完成第 {rc} 次修改。"
                                    st.info(log_msg)
                                    
                                    # 在日志中记录修改结果
                                    process_logs.append(f"{log_msg}\n\n**修改预览**：\n{draft_content}...")
                                    
                                    if rc >= 3:
                                        final_output = f"### ⚠️ 达到最大修改次数\n\n{draft_content}"

                    # 功能 a/b/c
                    else:
                        resource_context = build_resource_context(
                            selected_files=selected_files_for_agent,
                            selected_image_path=selected_image_for_chat,
                            search_source=search_source if st.session_state.current_function == "a" else None,
                            semantic_sort_by=semantic_sort_by if st.session_state.current_function == "a" else None,
                        )

                        initial_state = {
                            "current_function": st.session_state.current_function,
                            "task_input": prompt,
                            "resource_context": resource_context,
                            "selected_image_path": selected_image_for_chat or "",
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
                            "final_answer": ""
                        }
                        
                        agent_app = build_graph()

                        for output in agent_app.stream(initial_state, config=run_config):
                            for node_name, state_update in output.items():
                                
                                if node_name == "planner":
                                    st.write("🧠 **规划器 (Planner)** 制定了新计划：")
                                    process_logs.append("🧠 **规划器 (Planner)** 制定了新计划：")
                                    plans = state_update.get('plan', [])
                                    tools = state_update.get('planned_tools', [])
                                    for i, (p, t) in enumerate(zip(plans, tools)):
                                        log_str = f"**Step {i + 1}**: {p} `[Tool: {t}]`"
                                        st.info(log_str)
                                        process_logs.append(f"  - {log_str}")

                                elif node_name == "executor":
                                    step_history = state_update.get('step_history', [])
                                    if step_history:
                                        st.write("🛠️ **执行器 (Executor)** 完成操作：")
                                        st.code(step_history[-1], language="text")
                                        process_logs.append(f"🛠️ **执行器操作**:\n```text\n{step_history[-1]}\n```")

                                        last_log = step_history[-1]
                                        match = re.search(r"Result:\s*(.*)", last_log, flags=re.DOTALL)
                                        parsed_text = match.group(1).strip() if match else last_log.strip()
                                        parsed_text = re.sub(r"^【.*?执行结果】:\s*", "", parsed_text).strip()
                                        if parsed_text:
                                            final_output = parsed_text

                                elif node_name == "evaluator":
                                    eval_res = state_update.get('evaluation_result', {})
                                    passed = eval_res.get('passed', False)
                                    fb = eval_res.get('feedback', '')
                                    if passed:
                                        log_str = f"✅ **评估 (Evaluator)**: 步骤通过！(反馈: {fb})"
                                        st.success(log_str)
                                    else:
                                        log_str = f"⚠️ **评估 (Evaluator)**: 未通过，触发修正重试。(反馈: {fb})"
                                        st.warning(log_str)
                                    process_logs.append(log_str)

                                elif node_name == "give_up":
                                    final_output = None

                    # 共同结束处理逻辑
                    status.update(label="任务执行完毕！点击查看执行详情", state="complete", expanded=False)
                    final_answer_display = f"### 执行结果\n{final_output}" if final_output else "未获取到有效结果。"
                    stream_container.markdown(render_markdown_with_images(final_answer_display))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer_display,
                        "process_logs": process_logs
                    })
                    save_chat(st.session_state.current_chat_id, st.session_state.messages)

                except Exception as e:
                    status.update(label="执行过程中发生系统级错误", state="error", expanded=True)
                    error_msg = f"**系统异常终止**: {str(e)}\n\n*请调整您的提示词或检查配置后重新输入。*"
                    st.error(error_msg)
                    
                    # 将异常信息作为助手的回复强制存入会话历史，这样刷新或进行下一次对话时，上文不会断裂，前端依然能正常处理下一次输入
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "process_logs": process_logs  # 保留崩溃前已经打印出来的中间步骤日志
                    })
                    save_chat(st.session_state.current_chat_id, st.session_state.messages)
