
import streamlit as st
import json
import os
import re
from datetime import datetime
from graph.graph_builder import build_graph


# 配置与辅助函数 (多会话管理)

HISTORY_DIR = "chat_history"

# 确保历史记录文件夹存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


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


def init_new_chat():
    """初始化一个全新的对话"""
    # 使用时间戳作为唯一标识符 ID
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []


# 1. 页面初始化与状态管理

st.set_page_config(page_title="科研 Agent", page_icon="🤖", layout="wide")

# 初始化 session_state
if "current_chat_id" not in st.session_state:
    files = get_chat_files()
    if files:
        # 如果有历史记录，默认加载最新的一个
        latest_id = files[0].replace(".json", "")
        st.session_state.current_chat_id = latest_id
        st.session_state.messages = load_chat(latest_id)
    else:
        # 没有任何记录时，新建对话
        init_new_chat()


# 2. 侧边栏：对话管理 UI

with st.sidebar:
    st.title("💬 会话管理")

    # 新建对话按钮
    if st.button("➕ 新建对话", use_container_width=True):
        init_new_chat()
        st.rerun()

    # 删除当前对话按钮
    if st.button("🗑️ 删除当前对话", type="primary", use_container_width=True):
        delete_chat(st.session_state.current_chat_id)
        # 删除后尝试加载上一条记录，否则新建空记录
        files = get_chat_files()
        if files:
            st.session_state.current_chat_id = files[0].replace(".json", "")
            st.session_state.messages = load_chat(st.session_state.current_chat_id)
        else:
            init_new_chat()
        st.rerun()

    st.divider()
    st.subheader("历史记录")

    # 渲染历史对话列表
    for file in get_chat_files():
        chat_id = file.replace(".json", "")
        chat_msgs = load_chat(chat_id)

        # 提取第一条用户消息作为标题，限制长度
        title = "新对话 (空)"
        for m in chat_msgs:
            if m["role"] == "user":
                title = m["content"][:12] + "..." if len(m["content"]) > 12 else m["content"]
                break

        # 判断是否是当前激活的对话，并添加特殊样式标识
        is_active = (chat_id == st.session_state.current_chat_id)
        btn_label = f"▶ {title}" if is_active else f"📝 {title}"

        # 如果点击了某条历史记录，则切换过去
        if st.button(btn_label, key=chat_id, use_container_width=True, disabled=is_active):
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = chat_msgs
            st.rerun()


# 3. 主界面渲染与交互核心

st.title("🤖 智能科研助手")
st.caption(f"当前会话 ID: `{st.session_state.current_chat_id}`")

# 渲染当前选中的历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "process_logs" in msg:
            with st.expander("查看历史执行过程"):
                for log in msg["process_logs"]:
                    st.markdown(log)
        st.markdown(msg["content"])

# 底部聊天输入框
if prompt := st.chat_input("请输入你的科研需求（例如：调研最近关于大模型在医学领域的应用论文...）"):

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
                # 遍历除最后一条（刚刚输入的当前prompt）之外的所有消息
                for m in st.session_state.messages[:-1]:
                    role_name = "用户" if m["role"] == "user" else "助手"
                    # 这里过滤掉嵌套的 process_logs，只保留纯文本对话内容
                    content = m.get("content", "")
                    history_msgs.append(f"[{role_name}]: {content}")

                chat_history_str = "\n".join(history_msgs) if history_msgs else "无历史对话"


                # 初始化 Agent
                agent_app = build_graph()
                initial_state = {
                    "task_input": prompt,
                    "chat_history": chat_history_str, #删掉未使用的长程记忆，传入历史记录
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
                process_logs = [] #收集日志

                # 监听 LangGraph 的状态流转
                for output in agent_app.stream(initial_state):
                    for node_name, state_update in output.items():

                        if node_name == "planner":
                            # 1. 网页渲染文本
                            st.write("🧠 **规划器 (Planner)** 制定了新计划：")
                            # 2. 将纯文本追加到日志列表
                            process_logs.append("🧠 **规划器 (Planner)** 制定了新计划：")

                            plans = state_update.get('plan', [])
                            tools = state_update.get('planned_tools', [])
                            for i, (p, t) in enumerate(zip(plans, tools)):
                                # 拼装好纯文本字符串
                                log_str = f"**Step {i + 1}**: {p} `[Tool: {t}]`"

                                # 用纯文本字符串去渲染网页
                                st.info(log_str)

                                # 最后把纯文本字符串存入日志
                                process_logs.append(f"  - {log_str}")  # 收集plan日志

                        elif node_name == "executor":
                            step_history = state_update.get('step_history', [])
                            if step_history:
                                st.write("🛠️ **执行器 (Executor)** 完成操作：")
                                st.code(step_history[-1], language="text")
                                process_logs.append(f"🛠️ **执行器操作**:\n```text\n{step_history[-1]}\n```") # 收集executor日志
                                # 提取最后一次结果

                                last_log = step_history[-1]

                                # 使用 re.DOTALL 确保匹配 Result: 之后的所有内容（包含换行符）
                                match = re.search(r"Result:\s*(.*)", last_log, flags=re.DOTALL)
                                if match:
                                    parsed_text = match.group(1).strip()
                                else:
                                    parsed_text = last_log.strip()

                                # 清洗掉可能附带的工具前缀
                                parsed_text = re.sub(r"^【.*?执行结果】:\s*", "", parsed_text).strip()

                                # 【关键防御】：只有解析出非空内容时，才更新 final_output
                                # 这样就算遇到空输出的异常情况，也不会把之前正确拿到的结果覆盖掉
                                if parsed_text:
                                    final_output = parsed_text

                        elif node_name == "evaluator":
                            eval_res = state_update.get('evaluation_result', {})
                            passed = eval_res.get('passed', False)
                            fb = eval_res.get('feedback', '')
                            if passed:
                                log_str = f"✅ **评估 (Evaluator)**: 步骤通过！(反馈: {fb})"
                                st.success(f"✅ **评估 (Evaluator)**: 步骤通过！(反馈: {fb})")
                            else:
                                log_str = f"⚠️ **评估 (Evaluator)**: 未通过，触发修正重试。(反馈: {fb})"
                                st.warning(f"⚠️ **评估 (Evaluator)**: 未通过，触发修正重试。(反馈: {fb})")
                            process_logs.append(log_str) #收集evaluator日志

                        # 实时缓存历史记录，用于最后总结
                        # if "step_history" in state_update:
                        #     final_output = "\n\n---\n\n".join(state_update["step_history"])

                status.update(label="任务执行完毕！点击查看执行详情", state="complete", expanded=True)

                # 在外部正式展示最终汇总的科研结果
                final_answer_display = f"### 调研汇总\n{final_output}" if final_output else "未获取到有效结果。"
                st.markdown(final_answer_display)

                # 将最终结果存入状态并【覆盖写入当前 JSON 文件】
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer_display,
                    "process_logs": process_logs
                })
                save_chat(st.session_state.current_chat_id, st.session_state.messages)

            except Exception as e:
                status.update(label="执行过程中发生错误", state="error", expanded=True)
                st.error(f"Error: {str(e)}")