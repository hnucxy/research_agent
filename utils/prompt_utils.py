import os
from datetime import date
from typing import List, Optional


def build_resource_context(
    selected_files: Optional[List[dict]] = None,
    selected_image_path: Optional[str] = None,
    search_source: Optional[str] = None,
    semantic_sort_by: Optional[str] = None,
    semantic_year_filter: Optional[str] = None,
) -> str:
    blocks = []

    if search_source:
        today = date.today()
        search_lines = [
            f"- current_date: {today.isoformat()}",
            f"- current_year: {today.year}",
            f"- search_source: {search_source}",
            "- 时间理解: 用户说“近三年”时按当前年份向前换算为连续年份范围；“近几年”未说明数量时按近三年处理。",
        ]
        if search_source == "semantic_scholar":
            search_lines.append(
                f"- semantic_sort_by: {semantic_sort_by or 'relevance'}"
            )
            if semantic_year_filter:
                search_lines.append(f"- semantic_year_filter: {semantic_year_filter}")
            search_lines.append(
                "- 约束: 外部论文检索时使用 `semantic_scholar_search`。"
            )
        elif search_source == "arxiv":
            search_lines.append("- 约束: 外部论文检索时使用 `arxiv_search`。")

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
            + "\n整体总结或对比优先使用 `literature_read`，具体事实、定位与图表命中优先使用 `literature_rag_search`。"
            + "\n如果调用 `literature_rag_search`，可在工具参数中传入这些 `file_paths` 以限制检索范围。"
        )

    if selected_image_path:
        abs_image_path = os.path.abspath(selected_image_path).replace("\\", "/")
        blocks.append(
            "[已选图片]\n"
            f"- {abs_image_path}\n"
            "如果任务依赖图片内容，优先使用 `generate` 进行直接的多模态分析。"
        )

    return "\n\n".join(blocks) if blocks else "无"
