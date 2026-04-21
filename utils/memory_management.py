import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.logger import get_logger
from config.settings import Settings
from ui.config import MEMORY_AUDIT_LOG_FILE
from utils.file_utils import get_file_path_from_hash, unregister_file

logger = get_logger()

MEMORY_COLLECTIONS: Dict[str, Dict[str, str]] = {
    "experience": {
        "base_name": "global_experience",
        "label": "成功经验库",
    },
    "failure": {
        "base_name": "global_failure_experience",
        "label": "失败经验库",
    },
    "research": {
        "base_name": "global_research_knowledge",
        "label": "文献知识库",
    },
}

MEMORY_COLLECTION_OPTIONS = {
    key: value["label"] for key, value in MEMORY_COLLECTIONS.items()
}


def get_memory_audit_log_path() -> str:
    return os.path.abspath(MEMORY_AUDIT_LOG_FILE)


def _get_client():
    import chromadb

    return chromadb.PersistentClient(path="./chroma_db")


def _get_collection_name(collection_key: str) -> str:
    return Settings.get_collection_name(MEMORY_COLLECTIONS[collection_key]["base_name"])


def _get_collection(collection_key: str):
    try:
        return _get_client().get_collection(name=_get_collection_name(collection_key))
    except Exception:
        return None


def _shorten_text(text: str, limit: int = 120) -> str:
    clean_text = (text or "").replace("\n", " ").strip()
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[: limit - 3] + "..."


def _build_research_rows(
    metadatas: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for metadata in metadatas or []:
        metadata = metadata or {}
        file_hash = metadata.get("file_hash") or "unknown"
        row = grouped.setdefault(
            file_hash,
            {
                "entry_id": file_hash,
                "文件名": metadata.get("file_name", ""),
                "文件路径": metadata.get("file_path", ""),
                "会话ID": metadata.get("chat_id", ""),
                "文本块数": 0,
                "图片块数": 0,
            },
        )
        if metadata.get("type") == "image":
            row["图片块数"] += 1
        else:
            row["文本块数"] += 1

    return sorted(grouped.values(), key=lambda item: item["文件名"] or item["entry_id"])


def _count_failure_types(
    metadatas: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    counter = Counter()
    for metadata in metadatas or []:
        metadata = metadata or {}
        counter[metadata.get("failure_type") or "unknown"] += 1

    return [
        {"失败类型": failure_type, "数量": count}
        for failure_type, count in counter.most_common()
    ]


def list_memory_entries(collection_key: str, limit: int = 50) -> List[Dict[str, Any]]:
    collection = _get_collection(collection_key)
    if collection is None:
        return []

    if collection_key == "research":
        snapshot = collection.get(include=["metadatas"])
        rows = _build_research_rows(snapshot.get("metadatas", []))
        return rows[:limit]

    snapshot = collection.get(limit=limit, include=["documents", "metadatas"])
    ids = snapshot.get("ids", []) or []
    documents = snapshot.get("documents", []) or []
    metadatas = snapshot.get("metadatas", []) or []

    rows: List[Dict[str, Any]] = []
    for index, entry_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        document = documents[index] if index < len(documents) else ""
        row = {
            "entry_id": entry_id,
            "内容预览": _shorten_text(document),
        }
        if collection_key == "failure":
            row["失败类型"] = metadata.get("failure_type", "")
            row["工具名"] = metadata.get("tool_name", "")
        rows.append(row)
    return rows


def get_memory_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "overall_total_vectors": 0,
        "collections": {},
    }

    for collection_key, collection_info in MEMORY_COLLECTIONS.items():
        collection = _get_collection(collection_key)
        label = collection_info["label"]
        if collection is None:
            stats["collections"][collection_key] = {"label": label, "total_vectors": 0}
            continue

        total_vectors = collection.count()
        stats["overall_total_vectors"] += total_vectors

        if collection_key == "research":
            snapshot = collection.get(include=["metadatas"])
            rows = _build_research_rows(snapshot.get("metadatas", []))
            stats["collections"][collection_key] = {
                "label": label,
                "total_vectors": total_vectors,
                "document_count": len(rows),
                "text_chunk_count": sum(row["文本块数"] for row in rows),
                "image_chunk_count": sum(row["图片块数"] for row in rows),
            }
            continue

        if collection_key == "failure":
            snapshot = collection.get(include=["metadatas"])
            stats["collections"][collection_key] = {
                "label": label,
                "total_vectors": total_vectors,
                "failure_breakdown": _count_failure_types(snapshot.get("metadatas", [])),
            }
            continue

        stats["collections"][collection_key] = {
            "label": label,
            "total_vectors": total_vectors,
        }

    return stats


def append_memory_audit_log(
    action: str,
    collection_key: str,
    entry_ids: List[str],
    deleted_entries: int,
    deleted_vectors: int,
    actor: str,
    status: str = "success",
    error: str = "",
    audit_log_path: Optional[str] = None,
) -> None:
    log_path = audit_log_path or MEMORY_AUDIT_LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "collection_key": collection_key,
        "collection_label": MEMORY_COLLECTION_OPTIONS.get(collection_key, collection_key),
        "entry_ids": entry_ids,
        "deleted_entries": deleted_entries,
        "deleted_vectors": deleted_vectors,
        "actor": actor,
        "status": status,
        "error": error,
    }
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_remove_files(file_paths: List[str]) -> int:
    workspace_root = os.path.abspath(os.getcwd())
    removed_count = 0
    for file_path in sorted({path for path in file_paths if path}):
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(workspace_root):
            logger.warning("Skip deleting file outside workspace: %s", abs_path)
            continue
        if os.path.isfile(abs_path):
            os.remove(abs_path)
            removed_count += 1
    return removed_count


def delete_memory_entries(
    collection_key: str,
    entry_ids: List[str],
    actor: str,
    action: str,
) -> Dict[str, Any]:
    normalized_ids = [str(entry_id) for entry_id in entry_ids if entry_id]
    if not normalized_ids:
        return {"deleted_entries": 0, "deleted_vectors": 0, "entry_ids": []}

    collection = _get_collection(collection_key)
    if collection is None:
        return {"deleted_entries": 0, "deleted_vectors": 0, "entry_ids": []}

    try:
        if collection_key == "research":
            deleted_entries = 0
            deleted_vectors = 0
            deleted_file_paths: List[str] = []
            deleted_hashes: List[str] = []

            for file_hash in normalized_ids:
                snapshot = collection.get(
                    where={"file_hash": file_hash},
                    include=["metadatas"],
                )
                chunk_ids = snapshot.get("ids", []) or []
                metadatas = snapshot.get("metadatas", []) or []
                if not chunk_ids:
                    registered_path = get_file_path_from_hash(file_hash)
                    if registered_path:
                        deleted_entries += 1
                        deleted_hashes.append(file_hash)
                        deleted_file_paths.append(registered_path)
                        unregister_file(file_hash)
                    continue

                deleted_vectors += len(chunk_ids)
                deleted_entries += 1
                deleted_hashes.append(file_hash)
                for metadata in metadatas:
                    metadata = metadata or {}
                    deleted_file_paths.append(metadata.get("file_path", ""))
                    deleted_file_paths.append(metadata.get("image_path", ""))

                collection.delete(where={"file_hash": file_hash})
                unregister_file(file_hash)

            _safe_remove_files(deleted_file_paths)
            result = {
                "deleted_entries": deleted_entries,
                "deleted_vectors": deleted_vectors,
                "entry_ids": deleted_hashes,
            }
        else:
            snapshot = collection.get(ids=normalized_ids, include=["metadatas"])
            existing_ids = snapshot.get("ids", []) or []
            if existing_ids:
                collection.delete(ids=existing_ids)
            result = {
                "deleted_entries": len(existing_ids),
                "deleted_vectors": len(existing_ids),
                "entry_ids": existing_ids,
            }

        append_memory_audit_log(
            action=action,
            collection_key=collection_key,
            entry_ids=result["entry_ids"],
            deleted_entries=result["deleted_entries"],
            deleted_vectors=result["deleted_vectors"],
            actor=actor,
        )
        return result
    except Exception as exc:
        append_memory_audit_log(
            action=action,
            collection_key=collection_key,
            entry_ids=normalized_ids,
            deleted_entries=0,
            deleted_vectors=0,
            actor=actor,
            status="failed",
            error=str(exc),
        )
        raise


def delete_chat_memories(chat_id: str, actor: str) -> Dict[str, Any]:
    if not chat_id:
        return {"deleted_entries": 0, "deleted_vectors": 0, "entry_ids": []}

    total_entries = 0
    total_vectors = 0
    all_entry_ids: List[str] = []

    for collection_key in ("experience", "failure"):
        collection = _get_collection(collection_key)
        if collection is None:
            continue

        try:
            snapshot = collection.get(
                where={"chat_id": chat_id},
                include=["metadatas"],
            )
            existing_ids = snapshot.get("ids", []) or []
            if existing_ids:
                collection.delete(ids=existing_ids)

            deleted_count = len(existing_ids)
            total_entries += deleted_count
            total_vectors += deleted_count
            all_entry_ids.extend(existing_ids)
            append_memory_audit_log(
                action="delete_chat_memories",
                collection_key=collection_key,
                entry_ids=existing_ids,
                deleted_entries=deleted_count,
                deleted_vectors=deleted_count,
                actor=actor,
            )
        except Exception as exc:
            append_memory_audit_log(
                action="delete_chat_memories",
                collection_key=collection_key,
                entry_ids=[],
                deleted_entries=0,
                deleted_vectors=0,
                actor=actor,
                status="failed",
                error=str(exc),
            )
            raise

    return {
        "deleted_entries": total_entries,
        "deleted_vectors": total_vectors,
        "entry_ids": all_entry_ids,
    }
