import os
from typing import List, Optional


def build_resource_context(
    selected_files: Optional[List[dict]] = None,
    selected_image_path: Optional[str] = None,
) -> str:
    blocks = []

    if selected_files:
        file_lines = []
        for file_info in selected_files:
            file_name = file_info.get("name", "未知文献")
            file_path = file_info.get("path", "")
            abs_path = os.path.abspath(file_path).replace("\\", "/") if file_path else ""
            file_lines.append(f"- {file_name}: {abs_path}")

        blocks.append(
            "【已选文献】\n"
            + "\n".join(file_lines)
            + "\n文献整体总结/对比优先使用 `literature_read`；具体细节、指标、步骤定位优先使用 `literature_rag_search`。"
        )

    if selected_image_path:
        abs_image_path = os.path.abspath(selected_image_path).replace("\\", "/")
        blocks.append(
            "【已选图片】\n"
            f"- {abs_image_path}\n"
            "若任务需要结合图表内容回答，优先分配 `generate` 直接进行多模态分析。"
        )

    return "\n\n".join(blocks) if blocks else "无"
