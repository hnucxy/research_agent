import base64
import os
import re

import streamlit as st

from utils.image_utils import resolve_local_image_path


HORIZONTAL_RULE_PATTERN = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def normalize_markdown_for_display(text: str) -> str:
    if not text:
        return text

    lines = text.replace("\r\n", "\n").split("\n")
    normalized_lines = []
    in_code_fence = False

    for line in lines:
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            normalized_lines.append(line)
            continue

        if not in_code_fence and HORIZONTAL_RULE_PATTERN.match(line):
            if normalized_lines and normalized_lines[-1].strip():
                normalized_lines.append("")
            normalized_lines.append(line.strip())
            normalized_lines.append("")
            continue

        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def render_markdown_with_images(text: str) -> str:
    if not text:
        return text

    text = normalize_markdown_for_display(text)
    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

    def replace_image(match):
        alt_text = match.group(1)
        img_path = resolve_local_image_path(match.group(2))
        if img_path and not img_path.startswith("http") and os.path.exists(img_path):
            try:
                with open(img_path, "rb") as file:
                    b64 = base64.b64encode(file.read()).decode("utf-8")
                ext = img_path.split(".")[-1].lower()
                mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
                return f"![{alt_text}](data:{mime};base64,{b64})"
            except Exception:
                return match.group(0)
        return match.group(0)

    return re.sub(pattern, replace_image, text)


def render_rich_markdown(text: str):
    st.markdown(render_markdown_with_images(text), unsafe_allow_html=True)


def render_chat_messages(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "process_logs" in msg:
                with st.expander("查看历史执行过程"):
                    for log in msg["process_logs"]:
                        render_rich_markdown(log)
            render_rich_markdown(msg["content"])
