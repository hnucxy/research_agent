import streamlit as st
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config.settings import Settings
from graph.graph_builder import build_graph
from ui.config import UPLOAD_DIR, FUNC_MAP
from ui.session import save_chat
from utils.file_utils import get_file_hash, is_file_duplicate, register_file
from utils.document_parser import parse_pdf_to_markdown

def render_chat_page():
    # 渲染标题
    func_name = FUNC_MAP.get(st.session_state.current_function, "未知")

    # 1. 创建左右分栏，比例设为 3:1 (聊天区宽，文献区窄)
    chat_col, doc_col = st.columns([3, 1], gap="large")

    # 左侧：聊天主界面
    with chat_col:
        st.title(f"🤖 智能科研助手 - {func_name}")
        st.caption(f"当前会话 ID: `{st.session_state.current_chat_id}`")
        
        # 渲染当前选中的历史对话
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "process_logs" in msg:
                    with st.expander("查看历史执行过程"):
                        for log in msg["process_logs"]:
                            st.markdown(log)
                st.markdown(msg["content"])

    # 右侧：文献管理 (仅在 c, d 功能下渲染)
    with doc_col:
        if st.session_state.current_function in ["c", "d"]:
            chat_upload_dir = os.path.join(UPLOAD_DIR, st.session_state.current_chat_id)
            
            with st.expander("📁 文献上传与管理", expanded=False):
                st.write("📄 **上传文献**")
                
                # 【修改点 1】：扩展 type 限制，加入 "pdf"
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
                        
                        # 【修改点 2】：全局文件哈希去重拦截
                        file_hash = get_file_hash(file_bytes)
                        if is_file_duplicate(file_hash):
                            st.warning(f"⚠️ 文件 `{uploaded_file.name}` 已存在于全局库中，跳过解析与向量化。")
                            continue

                        # 获取文件后缀和基础名
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        file_path = os.path.join(chat_upload_dir, f"{base_name}.md")
                        
                        # 【修改点 3】：分流处理 MD 和 PDF
                        if file_ext == "pdf":
                            with st.spinner(f"正在使用 PyMuPDF 解析 `{uploaded_file.name}` 并提取图片..."):
                                text_content = parse_pdf_to_markdown(file_bytes, chat_upload_dir, base_name)
                        else:
                            text_content = file_bytes.decode("utf-8", errors="ignore")
                        
                        # 将最终获得的 Markdown 内容写入本地保存
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(text_content)
                        
                        # 向量化存入 ChromaDB
                        chunks = text_splitter.split_text(text_content)
                        Chroma.from_texts(
                            texts=chunks,
                            embedding=embeddings,
                            collection_name=st.session_state.current_chat_id,
                            persist_directory="./chroma_db"
                        )
                        
                        # 记录哈希到全局字典中
                        register_file(file_hash, file_path)
                        success_count += 1
                        
                    if success_count > 0:
                        st.success(f"成功上传、解析并向量化 {success_count} 份文献！")
                
                st.divider() # 分割线
                
                # 动态列出当前会话已存在的文献
                available_files = os.listdir(chat_upload_dir) if os.path.exists(chat_upload_dir) else []
                if available_files:
                    st.write("📚 **当前可用文献**")
                    for f_name in available_files:
                        # 内部再用小比例分隔文件名和删除按钮
                        col1, col2 = st.columns([4, 1])
                        col1.markdown(f"- `{f_name}`")
                        
                        btn_key = f"del_{st.session_state.current_chat_id}_{f_name}"
                        if col2.button("🗑️", key=btn_key, help="删除此文献"):
                            file_to_delete = os.path.join(chat_upload_dir, f_name)
                            if os.path.exists(file_to_delete):
                                os.remove(file_to_delete)
                            st.rerun()

    # 底部聊天输入框
    placeholder_text = "请输入你的科研需求..."
    if prompt := st.chat_input(placeholder_text):

        # 立即展示用户输入，并存入状态与当前文件
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat(st.session_state.current_chat_id, st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        # 助手响应区域
        with st.chat_message("assistant"):
            with st.status("Agent 正在思考与执行...", expanded=True) as status:
                try:
                    # 提取并格式化历史对话
                    history_msgs = []
                    for m in st.session_state.messages[:-1]:
                        role_name = "用户" if m["role"] == "user" else "助手"
                        content = m.get("content", "")
                        history_msgs.append(f"[{role_name}]: {content}")

                    chat_history_str = "\n".join(history_msgs) if history_msgs else "无历史对话"

                    enhanced_prompt = prompt
                    
                    # 只在功能c和d，并且拥有文件时，才给 Agent 注入强制阅读清单
                    if st.session_state.current_function in ["c", "d"]:
                        chat_upload_dir = os.path.join(UPLOAD_DIR, st.session_state.current_chat_id)
                        if os.path.exists(chat_upload_dir):
                            available_files = os.listdir(chat_upload_dir)
                            if available_files:
                                # 构建文件路径清单
                                file_list_str = "\n".join([f"- 文件名: {f}, 绝对路径: {os.path.join(chat_upload_dir, f).replace(chr(92), '/')}" for f in available_files])
                                
                                # 修改系统隐式提示，引入双轨路由指导
                                enhanced_prompt += (
                                    f"\n\n【系统隐式提示】：用户已在当前会话上传了文献。请根据用户需求在以下两个工具中选择：\n"
                                    f"1. 如果用户要求对文献进行全局性概括、总结核心贡献等，请规划并调用 `literature_read`（全文阅读工具）。必须将下方的“绝对路径”填入 file_path 参数中。\n"
                                    f"2. 如果用户询问文献中的具体细节、特定指标、或者定位某个算法步骤，请必须规划并调用 `literature_rag_search` 工具以节省时间。\n"
                                    f"【当前可用文献清单】:\n{file_list_str}\n"
                                )
                    

                    # 初始化 Agent
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

                    # 监听 LangGraph 的状态流转
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
                                    if match:
                                        parsed_text = match.group(1).strip()
                                    else:
                                        parsed_text = last_log.strip()

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

                    # 在外部正式展示最终汇总的科研结果
                    
                    final_answer_display = f"### 执行结果\n{final_output}" if final_output else "未获取到有效结果。"
                    st.markdown(final_answer_display)

                    # 覆盖写入
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer_display,
                        "process_logs": process_logs
                    })
                    save_chat(st.session_state.current_chat_id, st.session_state.messages)

                except Exception as e:
                    status.update(label="执行过程中发生错误", state="error", expanded=True)
                    st.error(f"Error: {str(e)}")
    