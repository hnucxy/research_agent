from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.logger import get_logger
from config.settings import Settings
from ui.config import UPLOAD_DIR
from utils.document_parser import parse_pdf_to_markdown
from utils.exceptions import DocumentParseError
from utils.file_utils import (
    get_all_registered_files,
    get_file_hash,
    get_file_path_from_hash,
    is_file_duplicate,
    register_file,
)
from utils.image_utils import extract_markdown_image_paths
from utils.memory_management import delete_memory_entries
from utils.vectorization_tasks import (
    get_vectorization_task,
    is_vectorization_active,
    list_vectorization_tasks,
    start_vectorization_task,
)

logger = get_logger()
GLOBAL_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "_global")


@dataclass(frozen=True)
class DocumentProgressEvent:
    file_name: str
    progress: Optional[int]
    message: str
    level: str = "info"


@dataclass(frozen=True)
class DocumentUploadSummary:
    success_count: int = 0
    processed_hashes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GlobalDocument:
    file_hash: str
    file_name: str
    file_path: str
    vector_status: str = ""

    @property
    def is_processing(self) -> bool:
        return self.vector_status in {"queued", "running"}

    @property
    def is_failed(self) -> bool:
        return self.vector_status == "failed"


ProgressCallback = Callable[[DocumentProgressEvent], None]


def get_library_file_path(file_hash: str, file_name: str) -> str:
    base_name = os.path.splitext(file_name)[0]
    return os.path.join(GLOBAL_UPLOAD_DIR, file_hash, f"{base_name}.md")


def _get_vectorstore() -> Chroma:
    embeddings = Settings.get_embeddings()
    research_collection = Settings.get_collection_name("global_research_knowledge")
    return Chroma(
        collection_name=research_collection,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )


def _get_vector_snapshot(vectorstore: Chroma, file_hash: str) -> dict:
    try:
        return vectorstore._collection.get(
            where={"file_hash": file_hash},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.warning("检查文献向量记录失败 | file_hash=%s | error=%s", file_hash, exc)
        return {}


def _has_vector_record(snapshot: dict) -> bool:
    return bool(snapshot and snapshot.get("ids"))


def _rebuild_markdown_from_vectors(snapshot: dict) -> str:
    documents = snapshot.get("documents", []) or []
    metadatas = snapshot.get("metadatas", []) or []
    text_chunks = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        if (metadata or {}).get("type") == "image":
            continue
        if document and not str(document).startswith("image://"):
            text_chunks.append(str(document).strip())
    return "\n\n".join(chunk for chunk in text_chunks if chunk)


def _copy_markdown_bundle_to_library(source_path: str, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    source_dir = os.path.dirname(os.path.abspath(source_path))
    target_dir = os.path.dirname(os.path.abspath(target_path))
    source_image_dir = os.path.join(source_dir, "images")
    target_image_dir = os.path.join(target_dir, "images")
    if os.path.isdir(source_image_dir) and source_image_dir != target_image_dir:
        shutil.copytree(source_image_dir, target_image_dir, dirs_exist_ok=True)

    with open(source_path, "r", encoding="utf-8") as file:
        text_content = file.read()

    source_rel = os.path.relpath(source_dir, os.getcwd()).replace("\\", "/")
    target_rel = os.path.relpath(target_dir, os.getcwd()).replace("\\", "/")
    source_abs = source_dir.replace("\\", "/")
    target_abs = target_dir.replace("\\", "/")
    text_content = text_content.replace(
        f"{source_rel}/images/", f"{target_rel}/images/"
    )
    text_content = text_content.replace(
        f"{source_abs}/images/",
        f"{target_abs}/images/",
    )
    with open(target_path, "w", encoding="utf-8") as file:
        file.write(text_content)
    return text_content


def _write_text_file(file_path: str, text_content: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_content)


def _emit(
    callback: Optional[ProgressCallback],
    file_name: str,
    progress: Optional[int],
    message: str,
    level: str = "info",
) -> None:
    if callback:
        callback(DocumentProgressEvent(file_name, progress, message, level))


def _build_vector_payload(
    text_content: str,
    library_file_path: str,
    file_name: str,
    file_hash: str,
    active_chat_id: str,
    text_splitter: RecursiveCharacterTextSplitter,
) -> tuple[list[str], list[dict]]:
    docs_to_insert: list[str] = []
    metadatas: list[dict] = []

    for chunk in text_splitter.split_text(text_content):
        docs_to_insert.append(chunk)
        metadatas.append(
            {
                "chat_id": active_chat_id,
                "file_name": file_name,
                "file_path": os.path.abspath(library_file_path),
                "file_hash": file_hash,
                "type": "text",
            }
        )
        for img_path in extract_markdown_image_paths(
            chunk, document_path=library_file_path
        ):
            if os.path.exists(img_path):
                docs_to_insert.append(f"image://{img_path}")
                metadatas.append(
                    {
                        "chat_id": active_chat_id,
                        "file_name": file_name,
                        "file_path": os.path.abspath(library_file_path),
                        "file_hash": file_hash,
                        "type": "image",
                        "image_path": img_path,
                        "context": chunk,
                    }
                )

    return docs_to_insert, metadatas


def process_uploaded_documents(
    uploaded_files: Iterable,
    chat_upload_dir: str,
    active_chat_id: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> DocumentUploadSummary:
    os.makedirs(chat_upload_dir, exist_ok=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    vectorstore = _get_vectorstore()
    success_count = 0
    processed_hashes: list[str] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = get_file_hash(file_bytes)
        file_ext = uploaded_file.name.split(".")[-1].lower()
        base_name = os.path.splitext(uploaded_file.name)[0]
        file_path = os.path.join(chat_upload_dir, f"{base_name}.md")
        library_file_path = get_library_file_path(file_hash, uploaded_file.name)
        text_content = ""

        try:
            logger.info(
                "开始处理上传文献 | file_name=%s | file_hash=%s",
                uploaded_file.name,
                file_hash,
            )
            _emit(progress_callback, uploaded_file.name, 0, f"准备处理 `{uploaded_file.name}`")

            vector_snapshot = _get_vector_snapshot(vectorstore, file_hash)
            old_path = (
                get_file_path_from_hash(file_hash)
                if is_file_duplicate(file_hash)
                else None
            )
            can_reuse_duplicate = bool(
                (old_path and os.path.exists(old_path))
                or _has_vector_record(vector_snapshot)
            )

            if can_reuse_duplicate:
                logger.info(
                    "检测到重复文献，跳过重复解析 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )
                _emit(
                    progress_callback,
                    uploaded_file.name,
                    20,
                    f"检测到重复文献，正在复用 `{uploaded_file.name}` 的解析结果",
                    "warning",
                )
                if old_path and os.path.exists(old_path):
                    text_content = _copy_markdown_bundle_to_library(
                        old_path, library_file_path
                    )
                    register_file(file_hash, library_file_path)
                elif _has_vector_record(vector_snapshot):
                    text_content = _rebuild_markdown_from_vectors(vector_snapshot)
                    if text_content:
                        _write_text_file(library_file_path, text_content)
                        register_file(file_hash, library_file_path)
                    else:
                        logger.warning(
                            "重复文献只有向量记录但无法还原文本 | file_name=%s | file_hash=%s",
                            uploaded_file.name,
                            file_hash,
                        )
                        _emit(
                            progress_callback,
                            uploaded_file.name,
                            None,
                            f"`{uploaded_file.name}` 只有向量记录但无法还原文本，已跳过。",
                            "warning",
                        )
                        continue

                with open(library_file_path, "r", encoding="utf-8") as file:
                    text_content = file.read()
                _write_text_file(file_path, text_content)

                if _has_vector_record(vector_snapshot):
                    logger.info(
                        "检测到文献已存在向量记录，跳过向量化 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    _emit(
                        progress_callback,
                        uploaded_file.name,
                        100,
                        f"`{uploaded_file.name}` 处理完成",
                    )
                    processed_hashes.append(file_hash)
                    success_count += 1
                    continue
            elif file_ext == "pdf":
                _emit(
                    progress_callback,
                    uploaded_file.name,
                    15,
                    f"正在解析 `{uploaded_file.name}` 并提取图片，请勿刷新或切换页面。",
                    "warning",
                )
                os.makedirs(os.path.dirname(library_file_path), exist_ok=True)
                try:
                    text_content = parse_pdf_to_markdown(
                        pdf_bytes=file_bytes,
                        output_dir=os.path.dirname(library_file_path),
                        base_name=base_name,
                        file_name=uploaded_file.name,
                        file_hash=file_hash,
                    )
                except DocumentParseError as exc:
                    logger.warning(
                        "PDF解析失败 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    _emit(
                        progress_callback,
                        uploaded_file.name,
                        None,
                        f"解析被跳过: {str(exc)}",
                        "error",
                    )
                    continue
                except Exception as exc:
                    logger.exception(
                        "PDF解析出现未知错误 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    _emit(
                        progress_callback,
                        uploaded_file.name,
                        None,
                        f"未知错误导致解析失败: {str(exc)}",
                        "error",
                    )
                    continue
            else:
                _emit(
                    progress_callback,
                    uploaded_file.name,
                    15,
                    f"正在读取 `{uploaded_file.name}` 并准备向量化，请稍候。",
                )
                logger.info(
                    "开始处理 Markdown 文献 | file_name=%s | file_hash=%s",
                    uploaded_file.name,
                    file_hash,
                )
                text_content = file_bytes.decode("utf-8", errors="ignore")

            _emit(
                progress_callback,
                uploaded_file.name,
                45,
                f"正在整理 `{uploaded_file.name}` 的文本内容",
            )
            _write_text_file(library_file_path, text_content)
            _write_text_file(file_path, text_content)

            docs_to_insert, metadatas = _build_vector_payload(
                text_content=text_content,
                library_file_path=library_file_path,
                file_name=uploaded_file.name,
                file_hash=file_hash,
                active_chat_id=active_chat_id,
                text_splitter=text_splitter,
            )

            if docs_to_insert:
                _emit(
                    progress_callback,
                    uploaded_file.name,
                    70,
                    f"`{uploaded_file.name}` 已加入后台向量化队列，可继续使用当前页面。",
                )
                try:
                    register_file(file_hash, library_file_path)
                    if not is_vectorization_active(file_hash):
                        start_vectorization_task(
                            file_hash=file_hash,
                            file_name=uploaded_file.name,
                            texts=docs_to_insert,
                            metadatas=metadatas,
                        )
                except Exception as exc:
                    logger.exception(
                        "文献向量化失败 | file_name=%s | file_hash=%s",
                        uploaded_file.name,
                        file_hash,
                    )
                    _emit(
                        progress_callback,
                        uploaded_file.name,
                        None,
                        f"向量化失败: {str(exc)}",
                        "error",
                    )
                    continue

            register_file(file_hash, library_file_path)
            processed_hashes.append(file_hash)
            logger.info(
                "完成上传文献处理 | file_name=%s | file_hash=%s",
                uploaded_file.name,
                file_hash,
            )
            _emit(
                progress_callback,
                uploaded_file.name,
                100,
                f"`{uploaded_file.name}` 处理完成",
            )
            success_count += 1
        finally:
            pass

    return DocumentUploadSummary(
        success_count=success_count,
        processed_hashes=tuple(processed_hashes),
    )


def list_global_documents() -> list[GlobalDocument]:
    documents: list[GlobalDocument] = []
    unique_md_files: dict[str, dict] = {}
    for file_hash, file_path in get_all_registered_files().items():
        if file_path.endswith(".md") and os.path.exists(file_path):
            unique_md_files[file_path] = {
                "file_name": os.path.basename(file_path),
                "file_hash": file_hash,
            }

    for file_path, file_meta in unique_md_files.items():
        file_hash = file_meta["file_hash"]
        vector_task = get_vectorization_task(file_hash)
        documents.append(
            GlobalDocument(
                file_hash=file_hash,
                file_name=file_meta["file_name"],
                file_path=file_path,
                vector_status=(vector_task or {}).get("status", ""),
            )
        )
    return documents


def list_completed_vectorization_notices() -> list[dict]:
    notices = []
    for task in list_vectorization_tasks():
        if task and task.get("status") in {"completed", "failed"}:
            notices.append(task)
    return notices


def list_images_for_documents(selected_files: Iterable[dict]) -> list[str]:
    all_images: list[str] = []
    for file_meta in selected_files:
        try:
            with open(file_meta["path"], "r", encoding="utf-8") as file:
                md_content = file.read()
            img_paths = extract_markdown_image_paths(
                md_content, document_path=file_meta["path"]
            )
            for img_path in img_paths:
                if os.path.exists(img_path) and img_path not in all_images:
                    all_images.append(img_path)
        except Exception:
            continue
    return all_images


def delete_global_document(file_hash: str, actor: str) -> dict:
    return delete_memory_entries(
        collection_key="research",
        entry_ids=[file_hash],
        actor=actor,
        action="single_delete_document_selector",
    )

