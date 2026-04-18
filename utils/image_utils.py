import base64
import os
import re
from typing import Iterable, List, Optional


MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def resolve_local_image_path(image_path: str, document_path: Optional[str] = None) -> str:
    """Resolve a markdown image path to an existing local path when possible."""
    if not image_path:
        return ""

    candidate = image_path.strip().strip("<>").split("?", 1)[0]
    if candidate.startswith(("http://", "https://", "data:")):
        return candidate

    normalized = os.path.normpath(candidate)
    if os.path.exists(normalized):
        return normalized

    if document_path:
        relative_to_doc = os.path.normpath(
            os.path.join(os.path.dirname(document_path), candidate)
        )
        if os.path.exists(relative_to_doc):
            return relative_to_doc

    return normalized if os.path.isabs(normalized) else ""


def extract_markdown_image_paths(
    markdown_text: str, document_path: Optional[str] = None
) -> List[str]:
    if not markdown_text:
        return []

    image_paths: List[str] = []
    seen = set()
    for _, raw_path in MARKDOWN_IMAGE_PATTERN.findall(markdown_text):
        resolved_path = resolve_local_image_path(raw_path, document_path=document_path)
        if (
            resolved_path
            and not resolved_path.startswith(("http://", "https://", "data:"))
            and os.path.exists(resolved_path)
            and resolved_path not in seen
        ):
            seen.add(resolved_path)
            image_paths.append(resolved_path)
    return image_paths


def unique_existing_image_paths(image_paths: Iterable[str]) -> List[str]:
    unique_paths: List[str] = []
    seen = set()
    for image_path in image_paths:
        if (
            image_path
            and os.path.exists(image_path)
            and image_path not in seen
        ):
            seen.add(image_path)
            unique_paths.append(image_path)
    return unique_paths


def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        payload = base64.b64encode(image_file.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png"
    if ext in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif ext == ".webp":
        mime_type = "image/webp"
    elif ext == ".gif":
        mime_type = "image/gif"

    return f"data:{mime_type};base64,{payload}"


def build_markdown_image_gallery(
    image_paths: Iterable[str], title: str = "**相关图表**"
) -> str:
    resolved_paths = unique_existing_image_paths(image_paths)
    if not resolved_paths:
        return ""

    markdown_lines = [title]
    markdown_lines.extend(f"![命中文献图片]({image_path})" for image_path in resolved_paths)
    return "\n".join(markdown_lines)


def append_image_gallery_to_markdown(
    markdown_text: str, image_paths: Iterable[str], title: str = "**相关图表**"
) -> str:
    base_text = markdown_text or ""
    existing_images = set(extract_markdown_image_paths(base_text))
    missing_images = [
        image_path
        for image_path in unique_existing_image_paths(image_paths)
        if image_path not in existing_images
    ]
    if not missing_images:
        return base_text

    gallery_markdown = build_markdown_image_gallery(missing_images, title=title)
    if not gallery_markdown:
        return base_text

    if not base_text:
        return gallery_markdown
    return f"{base_text}\n\n{gallery_markdown}"
