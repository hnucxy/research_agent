import streamlit as st
import json
import os
import re
import shutil
from datetime import datetime
from graph.graph_builder import build_graph

# 配置与辅助函数 (多会话管理)
HISTORY_DIR = "chat_history"
UPLOAD_DIR = "uploads"

# 定义功能映射表
FUNC_MAP = {
    "a": "文献检索",
    "b": "学术内容撰写",
    "c": "文献阅读",
    "d": "功能四"
}

# 确保历史记录文件夹存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 确保文件上传文件夹存在
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


def get_chat_files():
    """获取所有对话文件，按时间倒序排列（最新的在最上面）"""
    files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    return files


def load_chat(chat_id):
    """根据 ID 加载对应对话的 JSON 文件"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_chat(chat_id, messages):
    """将对话保存到对应 ID 的 JSON 文件"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def delete_chat(chat_id):
    """删除指定 ID 的对话文件"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # 同步删除该会话隔离的文献文件夹
    chat_upload_dir = os.path.join(UPLOAD_DIR, chat_id)
    if os.path.exists(chat_upload_dir):
        shutil.rmtree(chat_upload_dir)



def init_new_chat(func_code):
    """初始化一个全新的对话，ID 格式：时间戳_功能代码"""
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{func_code}"
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []
    st.session_state.current_function = func_code



# 页面初始化与状态管理

st.set_page_config(page_title="科研 Agent", page_icon="🤖", layout="wide")

# 初始化 session_state
if "current_function" not in st.session_state:
    st.session_state.current_function = None  # None 表示处于主页

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# 侧边栏：对话管理 UI

with st.sidebar:
    st.title("💬 会话管理")

    # 返回主页按钮
    if st.button("🏠 返回主页", use_container_width=True):
        st.session_state.current_function = None
        st.session_state.current_chat_id = None
        st.rerun()

    # 新建当前功能对话（仅在非主页时显示）
    if st.session_state.current_function is not None:
        if st.button(f"➕ 新建【{FUNC_MAP[st.session_state.current_function]}】对话", use_container_width=True):
            init_new_chat(st.session_state.current_function)
            st.rerun()

    # 删除当前对话按钮
    if st.session_state.current_chat_id is not None:
        if st.button("🗑️ 删除当前对话", type="primary", use_container_width=True):
            delete_chat(st.session_state.current_chat_id)
            # 删除后返回主页
            st.session_state.current_function = None
            st.session_state.current_chat_id = None
            st.rerun()

    st.divider()
    st.subheader("历史记录")

    # 渲染历史对话列表
    for file in get_chat_files():
        chat_id = file.replace(".json", "")
        chat_msgs = load_chat(chat_id)

        # 解析功能模块后缀以显示在标题中
        parts = chat_id.split("_")
        func_code_in_file = parts[-1] if len(parts) >= 3 else "a"
        func_name = FUNC_MAP.get(func_code_in_file, "未知功能")

        # 提取第一条用户消息作为标题，限制长度
        title = "新对话 (空)"
        for m in chat_msgs:
            if m["role"] == "user":
                title = m["content"][:10] + "..." if len(m["content"]) > 10 else m["content"]
                break

        # 判断是否是当前激活的对话
        is_active = (chat_id == st.session_state.current_chat_id)
        btn_label = f"▶ [{func_name}] {title}" if is_active else f"📝 [{func_name}] {title}"

        # 点击历史记录，切换页面和会话
        if st.button(btn_label, key=chat_id, use_container_width=True, disabled=is_active):
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = chat_msgs
            st.session_state.current_function = func_code_in_file
            st.rerun()


# 主界面渲染与交互核心

if st.session_state.current_function is None:

    # 视图 A: 导航主页

    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>科研助手agent系统</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>请选择您要使用的科研助手功能</p>", unsafe_allow_html=True)
    st.write("---")

    # 留空一些间距
    st.write("")
    st.write("")

    col1, col2, col3, col4 = st.columns([1, 4, 4, 1], gap="large")  # 增加 gap 让卡片之间更有呼吸感

    with col2:
        with st.container(border=True):
            st.subheader("📚 1. 文献检索")
            st.caption("输入关键词或主题，快速检索并总结相关领域的前沿学术论文与文献。")
            if st.button("进入检索", key="btn_a", use_container_width=True, type="primary"):
                init_new_chat("a")
                st.rerun()

        st.write("")  # 垂直间距

        with st.container(border=True):
            st.subheader("📖 3. 文献阅读")
            st.caption("上传预先转换好的 Markdown 格式文献，大模型将基于文献内容与您进行深度对话与总结。")
            if st.button("创建会话", key="btn_c", use_container_width=True, type="primary"):
                init_new_chat("c")
                st.rerun()

    with col3:
        with st.container(border=True):
            st.subheader("✍️ 2. 学术内容撰写")
            st.caption("根据已有文献资料和您的研究大纲，辅助撰写结构化的学术内容。")
            if st.button("进入撰写", key="btn_b", use_container_width=True, type="primary"):
                init_new_chat("b")
                st.rerun()

        st.write("")  # 垂直间距

        with st.container(border=True):
            st.subheader("⚙️ 4. 功能四")
            st.caption("TODO")
            if st.button("创建会话", key="btn_d", use_container_width=True):
                init_new_chat("d")
                st.rerun()

else:

    # 视图 B: 对话交互页
    func_name = FUNC_MAP.get(st.session_state.current_function, "未知")
    st.title(f"🤖 智能科研助手 - {func_name}")
    st.caption(f"当前会话 ID: `{st.session_state.current_chat_id}`")

    # 限定只有功能c和d允许上传和查阅文献
    if st.session_state.current_function in ["c", "d"]:
        chat_upload_dir = os.path.join(UPLOAD_DIR, st.session_state.current_chat_id)
        os.makedirs(chat_upload_dir, exist_ok=True)
        
        with st.sidebar:
            st.divider()
            st.subheader("📄 文献上传")
            # 开启 accept_multiple_files=True 支持多文献同时上传
            uploaded_files = st.file_uploader("上传您的 Markdown 文献", type=["md"], accept_multiple_files=True)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(chat_upload_dir, uploaded_file.name)
                    # 如果文件不存在则保存
                    if not os.path.exists(file_path):
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                st.success(f"成功上传 {len(uploaded_files)} 份文献！")
            
            # 动态列出当前会话已存在的文献
            available_files = os.listdir(chat_upload_dir) if os.path.exists(chat_upload_dir) else []
            if available_files:
                st.write("📚 **当前会话可用文献：**")
                
                # 遍历当前可用的文件，逐个渲染带有删除按钮的列表项
                for f_name in available_files:
                    col1, col2 = st.columns([5, 1])
                    col1.markdown(f"- `{f_name}`")
                    
                    # 动态生成唯一的按钮 key，防止多个组件冲突
                    btn_key = f"del_{st.session_state.current_chat_id}_{f_name}"
                    
                    if col2.button("🗑️", key=btn_key, help="从当前会话中删除此文献"):
                        file_to_delete = os.path.join(chat_upload_dir, f_name)
                        if os.path.exists(file_to_delete):
                            os.remove(file_to_delete)
                        st.rerun()  # 删除后强制刷新页面，更新列表

    # 渲染当前选中的历史对话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "process_logs" in msg:
                with st.expander("查看历史执行过程"):
                    for log in msg["process_logs"]:
                        st.markdown(log)
            st.markdown(msg["content"])

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
                                
                                enhanced_prompt += (
                                    f"\n\n【系统隐式提示】：用户已在当前会话上传了文献，请根据用户需求，优先规划并调用 `literature_read` 工具来分析以下文献。\n"
                                    f"【当前可用文献清单】:\n{file_list_str}\n"
                                    f"【重要注意】：调用文献工具时，必须准确将上述清单中的“绝对路径”填入 file_path 参数中。如果需要比较多篇文献，请依次规划多个工具调用步骤。"
                                )
                    

                    # 初始化 Agent
                    agent_app = build_graph()
                    initial_state = {
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