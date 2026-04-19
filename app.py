import streamlit as st
from config.logger import get_logger
from ui.config import init_directories
from ui.components import render_sidebar
from ui.session import init_session_state
from ui.view_home import render_home_page
from ui.view_chat import render_chat_page

logger = get_logger()

# 1. 页面级配置
st.set_page_config(page_title="科研 Agent", page_icon="🤖", layout="wide")

# 2. 初始化环境与状态
init_directories()
init_session_state()

# 3. 渲染通用侧边栏
render_sidebar()

# 4. 核心路由分发
try:
    if st.session_state.current_function is None:
        render_home_page()
    else:
        render_chat_page()
except Exception:
    logger.exception(
        "Streamlit页面渲染出现未捕获异常 | current_function=%s | chat_id=%s",
        st.session_state.get("current_function"),
        st.session_state.get("current_chat_id"),
    )
    raise
