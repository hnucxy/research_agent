import streamlit as st
from ui.session import init_new_chat

def render_home_page():
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
