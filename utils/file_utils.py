import hashlib
import json
import os
from ui.config import REGISTRY_FILE

def get_file_hash(file_bytes: bytes) -> str:
    """计算文件的 SHA-256 哈希值"""
    return hashlib.sha256(file_bytes).hexdigest()

def is_file_duplicate(file_hash: str) -> bool:
    """检查文件哈希是否已存在于全局注册表中"""
    if not os.path.exists(REGISTRY_FILE):
        return False
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
    return file_hash in registry

def get_file_path_from_hash(file_hash: str) -> str:
    """根据哈希值获取已有文件的本地绝对路径"""
    if not os.path.exists(REGISTRY_FILE):
        return None
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
    return registry.get(file_hash)

def register_file(file_hash: str, file_path: str):
    """将新文件的哈希与路径记录到全局注册表"""
    registry = {}
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)
    
    registry[file_hash] = file_path
    
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

def get_all_registered_files() -> dict:
    """获取全局注册表中的所有文件"""
    if not os.path.exists(REGISTRY_FILE):
        return {}
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)