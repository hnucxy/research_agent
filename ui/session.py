import json
import os
import shutil
from datetime import datetime

import streamlit as st

from config.logger import get_logger
from config.settings import Settings
from ui.config import HISTORY_DIR, UPLOAD_DIR
from utils.memory_management import delete_chat_memories

logger = get_logger()


def get_chat_files():
    """获取所有对话文件, 按时间倒序排列。"""
    files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    return files


def load_chat(chat_id):
    """根据 chat_id 加载对应的对话文件。"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_chat(chat_id, messages):
    """将对话消息持久化到磁盘。"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def _safe_remove_chat_upload_dir(chat_id):
    if not chat_id or chat_id == "_global":
        return
    upload_root = os.path.abspath(UPLOAD_DIR)
    chat_upload_dir = os.path.abspath(os.path.join(UPLOAD_DIR, chat_id))
    if os.path.commonpath([upload_root, chat_upload_dir]) != upload_root:
        return
    if os.path.isdir(chat_upload_dir):
        shutil.rmtree(chat_upload_dir)


def delete_chat(chat_id):
    """删除指定对话及其会话上传目录，不删除全局向量库内容。"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

    try:
        delete_chat_memories(chat_id, actor=f"delete_chat:{chat_id}")
    except Exception as exc:
        logger.warning("删除会话记忆失败 | chat_id=%s | error=%s", chat_id, exc)
    _safe_remove_chat_upload_dir(chat_id)


def init_new_chat(func_code):
    """在当前功能下初始化一个新的对话。"""
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{func_code}"
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []
    st.session_state.current_function = func_code


def init_session_state():
    if "current_function" not in st.session_state:
        st.session_state.current_function = None

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "search_source" not in st.session_state:
        st.session_state.search_source = "arxiv"

    if "semantic_sort_by" not in st.session_state:
        st.session_state.semantic_sort_by = "relevance"

    if "token_usage" not in st.session_state:
        model_name = getattr(Settings, "MODEL_NAME", None) or "未知模型"
        st.session_state.token_usage = {
            "model_name": model_name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
        }
