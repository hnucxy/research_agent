from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma

from config.logger import get_logger
from config.settings import Settings

logger = get_logger()

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-vectorizer")
_lock = Lock()
_tasks: Dict[str, Dict[str, Any]] = {}

ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"completed", "failed"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _snapshot(task: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not task:
        return None
    return {
        key: value
        for key, value in task.items()
        if key not in {"texts", "metadatas"}
    }


def get_vectorization_task(file_hash: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _snapshot(_tasks.get(file_hash))


def list_vectorization_tasks() -> List[Dict[str, Any]]:
    with _lock:
        return [_snapshot(task) for task in _tasks.values()]


def is_vectorization_active(file_hash: str) -> bool:
    with _lock:
        task = _tasks.get(file_hash)
        return bool(task and task.get("status") in ACTIVE_STATUSES)


def _set_task_state(file_hash: str, **updates: Any) -> None:
    with _lock:
        task = _tasks.get(file_hash)
        if not task:
            return
        task.update(updates)
        task["updated_at"] = _now()


def _run_vectorization(
    file_hash: str,
    file_name: str,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    _set_task_state(file_hash, status="running", started_at=_now())
    logger.info(
        "开始后台向量化文献 | file_name=%s | file_hash=%s | vector_count=%s",
        file_name,
        file_hash,
        len(texts),
    )

    try:
        embeddings = Settings.get_embeddings()
        research_collection = Settings.get_collection_name("global_research_knowledge")
        vectorstore = Chroma(
            collection_name=research_collection,
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
    except Exception as exc:
        logger.exception(
            "后台文献向量化失败 | file_name=%s | file_hash=%s",
            file_name,
            file_hash,
        )
        _set_task_state(file_hash, status="failed", error=str(exc), finished_at=_now())
        return

    logger.info(
        "完成后台向量化文献 | file_name=%s | file_hash=%s",
        file_name,
        file_hash,
    )
    _set_task_state(file_hash, status="completed", error="", finished_at=_now())


def start_vectorization_task(
    file_hash: str,
    file_name: str,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
) -> bool:
    if not texts:
        return False

    with _lock:
        existing = _tasks.get(file_hash)
        if existing and existing.get("status") in ACTIVE_STATUSES:
            return False

        _tasks[file_hash] = {
            "file_hash": file_hash,
            "file_name": file_name,
            "status": "queued",
            "total_vectors": len(texts),
            "error": "",
            "created_at": _now(),
            "updated_at": _now(),
            "started_at": "",
            "finished_at": "",
        }

    _executor.submit(_run_vectorization, file_hash, file_name, list(texts), list(metadatas))
    return True
