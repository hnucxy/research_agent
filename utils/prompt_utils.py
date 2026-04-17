import os
from typing import List, Optional


def build_resource_context(
    selected_files: Optional[List[dict]] = None,
    selected_image_path: Optional[str] = None,
    search_source: Optional[str] = None,
    semantic_sort_by: Optional[str] = None,
) -> str:
    blocks = []

    if search_source:
        search_lines = [f"- search_source: {search_source}"]
        if search_source == "semantic_scholar":
            search_lines.append(
                f"- semantic_sort_by: {semantic_sort_by or 'relevance'}"
            )
            search_lines.append(
                "- 约束: 外部论文检索时使用 `semantic_scholar_search`。"
            )
        elif search_source == "arxiv":
            search_lines.append(
                "- 约束: 外部论文检索时使用 `arxiv_search`。"
            )

        blocks.append("[检索偏好]\n" + "\n".join(search_lines))

    if selected_files:
        file_lines = []
        for file_info in selected_files:
            file_name = file_info.get("name", "未知文献")
            file_path = file_info.get("path", "")
            abs_path = os.path.abspath(file_path).replace("\\", "/") if file_path else ""
            file_lines.append(f"- {file_name}: {abs_path}")

        blocks.append(
            "[已选文献]\n"
            + "\n".join(file_lines)
            + "\n文献整体总结或对比优先使用 `literature_read`, 具体事实和定位优先使用 `literature_rag_search`。"
        )

    if selected_image_path:
        abs_image_path = os.path.abspath(selected_image_path).replace("\\", "/")
        blocks.append(
            "[已选图片]\n"
            f"- {abs_image_path}\n"
            "如果任务依赖图片内容, 优先使用 `generate` 进行直接的多模态分析。"
        )

    return "\n\n".join(blocks) if blocks else "无"
