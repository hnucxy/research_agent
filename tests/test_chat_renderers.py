import sys
import types


sys.modules.setdefault("streamlit", types.SimpleNamespace())

from ui.chat_renderers import normalize_markdown_for_display


def test_normalize_markdown_for_display_breaks_setext_heading_pattern():
    markdown = "URL: https://example.org/paper\n---\n标题: A Paper"

    normalized = normalize_markdown_for_display(markdown)

    assert normalized == "URL: https://example.org/paper\n\n---\n\n标题: A Paper"


def test_normalize_markdown_for_display_preserves_code_fences():
    markdown = "```text\nURL: https://example.org/paper\n---\n```"

    normalized = normalize_markdown_for_display(markdown)

    assert normalized == markdown
