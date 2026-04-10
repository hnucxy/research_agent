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
        if st.session_state.current_function in ["c", "d"]:
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
                    success_count = 0

                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.getvalue()
                        file_hash = get_file_hash(file_bytes)
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        file_path = os.path.join(chat_upload_dir, f"{base_name}.md")
                        
                        # 防溢出复用机制
                        if is_file_duplicate(file_hash):
                            st.warning(f"⚠️ 文件 `{uploaded_file.name}` 已存在于全局库中，跳过解析与向量化。")
                            # 获取过往的真实路径并复制到当前会话
                            old_path = get_file_path_from_hash(file_hash)
                            if old_path and os.path.exists(old_path):
                                shutil.copy2(old_path, file_path)
                            continue

                        # 新文件解析与存入 Chroma 逻辑保持不变...
                        if file_ext == "pdf":
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
                                "type": "text"
                            })
                            img_paths = re.findall(r'!\[.*?\]\((.*?)\)', chunk)
                            for img_path in img_paths:
                                if os.path.exists(img_path):
                                    docs_to_insert.append(f"image://{img_path}")
                                    metadatas.append({
                                        "chat_id": st.session_state.current_chat_id,
                                        "file_name": uploaded_file.name,
                                        "type": "image",
                                        "image_path": img_path,
                                        "context": chunk
                                    })
                        
                        if docs_to_insert:
                            Chroma.from_texts(
                                texts=docs_to_insert,
                                metadatas=metadatas,
                                embedding=embeddings,
                                collection_name="global_research_knowledge",
                                persist_directory="./chroma_db"
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
    # 底部聊天输入框
    placeholder_text = "请输入你的科研需求..."
    if prompt := st.chat_input(placeholder_text):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat(st.session_state.current_chat_id, st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Agent 正在思考与执行...", expanded=True) as status:
                try:
                    history_msgs = []
                    for m in st.session_state.messages[:-1]:
                        role_name = "用户" if m["role"] == "user" else "助手"
                        content = m.get("content", "")
                        history_msgs.append(f"[{role_name}]: {content}")

                    chat_history_str = "\n".join(history_msgs) if history_msgs else "无历史对话"
                    enhanced_prompt = prompt
                    
                    # 只向Agent暴露被勾选的文献
                    if st.session_state.current_function in ["c", "d"]:
                        if selected_files_for_agent:
                            # 提取选中的文献的真实绝对路径，并转义反斜杠防止 JSON 解析出错
                            file_list_str = "\n".join([f"- 文件名: {f['name']}, 绝对路径: {os.path.abspath(f['path']).replace(chr(92), '/')}" for f in selected_files_for_agent])
                            
                            enhanced_prompt += (
                                f"\n\n【系统隐式提示】：用户已在全局文献库中**勾选**了以下文献。请根据用户需求在以下两个工具中选择：\n"
                                f"1. 如果用户要求对文献进行全局性概括、总结核心贡献等，请规划并调用 `literature_read`（全文阅读工具）。必须将下方的“绝对路径”填入 file_path 参数中。\n"
                                f"2. 如果用户询问文献中的具体细节、特定指标、或者定位某个算法步骤，请必须规划并调用 `literature_rag_search` 工具以节省时间。\n"
                                f"【当前用户勾选的全局文献清单】:\n{file_list_str}\n"
                            )

                    # 初始化 Agent 状态
                    agent_app = build_graph()
                    initial_state = {
                        "current_function": st.session_state.current_function,
                        "task_input": enhanced_prompt,
                        "chat_history": chat_history_str,
                        "plan": [],
                        "planned_tools": [],
                        "current_step_index": 0,
                        "retry_count": 0,
                        "replan_count": 0,
                        "step_history": [],
                        "evaluation_result": {},
                        "final_answer": ""
                    }

                    final_output = ""
                    process_logs = []

                    for output in agent_app.stream(initial_state):
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

                    status.update(label="任务执行完毕！点击查看执行详情", state="complete", expanded=True)
                    
                    final_answer_display = f"### 执行结果\n{final_output}" if final_output else "未获取到有效结果。"
                    st.markdown(render_markdown_with_images(final_answer_display))

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