import os
import json

HISTORY_DIR = "chat_history"
UPLOAD_DIR = "uploads"
REGISTRY_FILE = "uploaded_files_registry.json"

FUNC_MAP = {
    "a": "文献检索",
    "b": "学术内容撰写",
    "c": "文献阅读",
    "d": "功能四"
}

def init_directories():
    # 确保历史记录文件夹存在
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    
    # 确保文件上传文件夹存在
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # 初始化哈希注册表
    if not os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)