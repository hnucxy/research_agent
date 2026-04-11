import os
import json
import shutil
import chromadb
from datetime import datetime
import streamlit as st
from ui.config import HISTORY_DIR, UPLOAD_DIR
from utils.token_tracker import TokenTracker
from config.settings import Settings

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

    # 同步删除 ChromaDB 中属于该会话的 Collection
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        client.delete_collection(name=chat_id)
    except Exception:
        # 如果 collection 不存在或报错，忽略即可
        pass



def init_new_chat(func_code):
    """初始化一个全新的对话，ID 格式：时间戳_功能代码"""
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{func_code}"
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []
    st.session_state.current_function = func_code


def init_session_state():
    # 初始化 session_state
    if "current_function" not in st.session_state:
        st.session_state.current_function = None  # None 表示处于主页

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "token_usage" not in st.session_state:
        # 安全获取模型名称，处理环境变量未配置的兜底情况
        model_name = getattr(Settings, "MODEL_NAME", None) or "未知模型"
        
        st.session_state.token_usage = {
            "model_name": model_name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0
        }