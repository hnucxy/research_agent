import os

import streamlit as st

from ui.chat_panels import (
    render_document_management_panel,
    render_memory_management_panel,
)
from ui.config import UPLOAD_DIR
from ui.session import init_new_chat


def render_home_page():
    st.markdown(
        "<h1 style='text-align: center; margin-top: 50px;'>科研助手 Agent 系统</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>请选择您要使用的科研助手功能</p>",
        unsafe_allow_html=True,
    )
    st.write("---")
    st.write("")
    st.write("")

    col1, col2, col3, col4 = st.columns([1, 4, 4, 1], gap="large")

    with col2:
        with st.container(border=True):
            st.subheader("📎 1. 文献检索")
            st.caption("输入关键词或主题，快速检索并总结相关领域的前沿学术论文与文献。")
            if st.button("进入检索", key="btn_a", width="stretch", type="primary"):
                init_new_chat("a")
                st.rerun()

        st.write("")

        with st.container(border=True):
            st.subheader("📉 3. 文献阅读")
            st.caption("上传预先转换好的 Markdown 格式文献，大模型将基于文献内容与您进行深度对话与总结。")
            if st.button("创建会话", key="btn_c", width="stretch", type="primary"):
                init_new_chat("c")
                st.rerun()

    with col3:
        with st.container(border=True):
            st.subheader("✍️ 2. 学术内容撰写")
            st.caption("根据已有文献资料和您的研究大纲，辅助撰写结构化的学术内容。")
            if st.button("进入撰写", key="btn_b", width="stretch", type="primary"):
                init_new_chat("b")
                st.rerun()

        st.write("")

        with st.container(border=True):
            st.subheader("🧭 4. 论文审稿与重构")
            st.caption("基于 Author-Reviewer 辩论模型。上传原稿或直接输入，让两个 Agent 为您审查并迭代修改。")
            if st.button("创建会话", key="btn_d", width="stretch", type="primary"):
                init_new_chat("d")
                st.rerun()

    st.write("")
    st.write("---")
    st.subheader("文献与记忆管理")

    home_upload_dir = os.path.join(UPLOAD_DIR, "_home")
    render_document_management_panel(home_upload_dir)
    render_memory_management_panel()
