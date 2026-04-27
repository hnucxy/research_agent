import os
import re
from datetime import date
from typing import List, Optional


_CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def extract_year_filter(text: str, current_year: Optional[int] = None) -> str:
    if not text:
        return ""

    if current_year is None:
        current_year = date.today().year

    text = str(text)
    if re.search(r"(不限|不限制|不限定|无需|不要).{0,8}(年份|年代|时间|年限)", text):
        return ""

    range_match = re.search(
        r"((?:19|20)\d{2})\s*年?\s*(?:-|~|—|到|至)\s*((?:19|20)\d{2})\s*年?",
        text,
    )
    if range_match:
        start, end = range_match.groups()
        return f"{start}-{end}" if int(start) <= int(end) else ""

    recent_match = re.search(
        r"(?:近|最近|过去|过去的)\s*([一二两三四五六七八九十]|\d{1,2})\s*年",
        text,
    )
    if recent_match:
        count = _parse_year_count(recent_match.group(1))
        if count:
            start_year = current_year - count + 1
            return f"{start_year}-{current_year}"

    if re.search(r"(?:近|最近|过去|过去的)\s*几年", text):
        return f"{current_year - 2}-{current_year}"

    year_match = re.search(r"((?:19|20)\d{2})\s*年", text)
    if year_match:
        return year_match.group(1)

    return ""


def _parse_year_count(value: str) -> int:
    if value.isdigit():
        return int(value)
    return _CHINESE_NUMBER_MAP.get(value, 0)


def build_resource_context(
    selected_files: Optional[List[dict]] = None,
    selected_image_path: Optional[str] = None,
    search_source: Optional[str] = None,
    semantic_sort_by: Optional[str] = None,
    semantic_year_filter: Optional[str] = None,
    user_task: Optional[str] = None,
) -> str:
    blocks = []

    if search_source:
        today = date.today()
        extracted_year_filter = extract_year_filter(user_task or "", today.year)
        search_lines = [
            f"- current_date: {today.isoformat()}",
            f"- current_year: {today.year}",
            f"- search_source: {search_source}",
            "- 年份过滤策略: 只有用户原始请求明确包含年份或时间范围时，才允许添加年份限制；未指定时禁止默认添加“近几年/近三年”等限制。",
        ]
        if extracted_year_filter:
            search_lines.append(f"- extracted_year_filter: {extracted_year_filter}")
        else:
            search_lines.append(
                "- extracted_year_filter: none（用户未指定年份限制，规划和工具参数都不要添加年份过滤）"
            )
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
