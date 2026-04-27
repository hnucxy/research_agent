import streamlit as st
from ui.session import get_chat_files, load_chat, delete_chat, init_new_chat
from ui.config import FUNC_MAP


def _return_home():
    st.session_state.current_function = None
    st.session_state.current_chat_id = None


def _new_chat_for_current_function():
    init_new_chat(st.session_state.current_function)


def _delete_current_chat():
    delete_chat(st.session_state.current_chat_id)
    st.session_state.current_function = None
    st.session_state.current_chat_id = None


def _switch_chat(chat_id: str, func_code: str):
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = load_chat(chat_id)
    st.session_state.current_function = func_code


def render_sidebar():
    # 侧边栏：对话管理 UI
    with st.sidebar:
        st.title("💬 会话管理")

        # 返回主页按钮
        st.button("🏠 返回主页", width="stretch", on_click=_return_home)

        # 新建当前功能对话（仅在非主页时显示）
        if st.session_state.current_function is not None:
            st.button(
                f"➕ 新建【{FUNC_MAP[st.session_state.current_function]}】对话",
                width="stretch",
                on_click=_new_chat_for_current_function,
            )

        # 删除当前对话按钮
        if st.session_state.current_chat_id is not None:
            st.button(
                "🗑️ 删除当前对话",
                type="primary",
                width="stretch",
                on_click=_delete_current_chat,
            )

        st.divider()
        st.subheader("📊 资源消耗看板")
        if "token_usage" in st.session_state:
            usage = st.session_state.token_usage
            
            st.metric(label="总 API 请求", value=f"{usage['successful_requests']} 次")
            
            col1, col2 = st.columns(2)
            col1.metric(label="⬆️ 发送 (Prompt)", value=usage['prompt_tokens'])
            col2.metric(label="⬇️ 接收 (Completion)", value=usage['completion_tokens'])
            
            st.caption(f"🤖 当前模型: `{usage['model_name']}`")
            st.caption("*(注：受缓存命中等机制影响，具体资费以云端账单为准)*")

        st.divider()
        st.subheader("历史记录")

        # 渲染历史对话列表
        for file in get_chat_files():
            chat_id = file.replace(".json", "")
            
            # 解析功能模块后缀以显示在标题中
            parts = chat_id.split("_")
            func_code_in_file = parts[-1] if len(parts) >= 3 else "a"
            
            # 如果当前已经处于某个功能下，只显示该功能的历史记录
            if st.session_state.current_function is not None:
                if func_code_in_file != st.session_state.current_function:
                    continue  # 如果功能不匹配，直接跳过，不渲染在侧边栏中

            chat_msgs = load_chat(chat_id)
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
            st.button(
                btn_label,
                key=chat_id,
                width="stretch",
                disabled=is_active,
                on_click=_switch_chat,
                args=(chat_id, func_code_in_file),
            )

        
