import html
import re
from difflib import SequenceMatcher


RELAXED_INTENT_RULES = {
    "language_polish": [
        "润色",
        "润饰",
        "语言优化",
        "学术化",
        "学术表达",
        "改写",
        "重写表述",
        "精炼",
        "简化表达",
        "语法",
        "病句",
        "通顺",
        "口语",
        "口语化",
        "书面化",
        "措辞",
    ],
    "logic_restructure": [
        "重构逻辑",
        "逻辑优化",
        "重组结构",
        "结构优化",
        "梳理结构",
        "调整结构",
        "提升连贯",
        "增强连贯",
        "段落衔接",
        "论述顺序",
    ],
}

STRICT_INTENT_RULES = {
    "evidence_enhancement": [
        "补充论据",
        "补强论据",
        "补充证据",
        "增加证据",
        "补充数据",
        "增加数据",
        "补充定量",
        "增加定量",
        "补充引用",
        "增加引用",
        "补充文献",
        "增加文献",
        "事实核查",
        "核查事实",
        "核实数据",
        "完善结论",
        "补强结论",
        "实验结论",
        "实验结果",
        "补充实验",
        "补充指标",
        "增加指标",
    ]
}


def classify_review_intent(user_prompt: str) -> dict:
    text = (user_prompt or "").lower()

    for intent, keywords in STRICT_INTENT_RULES.items():
        if any(keyword in text for keyword in keywords):
            return {
                "task_intent": intent,
                "review_mode": "strict",
                "review_focus": "重点审查事实支撑、定量数据、引用依据与结论是否得到原稿支持。",
            }

    for intent, keywords in RELAXED_INTENT_RULES.items():
        if any(keyword in text for keyword in keywords):
            focus = (
                "重点优化学术表达、语法准确性与风格一致性。"
                if intent == "language_polish"
                else "重点优化结构层次、段落衔接与论证连贯性。"
            )
            return {
                "task_intent": intent,
                "review_mode": "relaxed",
                "review_focus": focus,
            }

    return {
        "task_intent": "general_revision",
        "review_mode": "relaxed",
        "review_focus": "默认以学术表达、清晰度和结构合理性为主，不因缺乏新增数据而直接否决。",
    }


def _sentence_tokens(text: str) -> list[str]:
    if not text:
        return []

    tokens: list[str] = []
    parts = re.split(r"(\n+)", text)
    sentence_pattern = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?|\S+", re.S)

    for part in parts:
        if not part:
            continue
        if "\n" in part:
            tokens.append(part)
            continue
        tokens.extend(
            [segment for segment in sentence_pattern.findall(part) if segment.strip()]
        )
    return tokens


def _format_token(token: str) -> str:
    if "\n" in token:
        return token.replace("\n", "<br>")
    return html.escape(token)


def build_diff_markup(previous_text: str, revised_text: str) -> str:
    previous_tokens = _sentence_tokens(previous_text or "")
    revised_tokens = _sentence_tokens(revised_text or "")
    matcher = SequenceMatcher(a=previous_tokens, b=revised_tokens)

    fragments: list[str] = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        old_chunk = "".join(_format_token(token) for token in previous_tokens[a0:a1])
        new_chunk = "".join(_format_token(token) for token in revised_tokens[b0:b1])

        if opcode == "equal":
            fragments.append(new_chunk)
        elif opcode == "delete":
            fragments.append(
                f"<del style='background:#fde7e9;color:#b42318;padding:0 2px;'>{old_chunk}</del>"
            )
        elif opcode == "insert":
            fragments.append(
                f"<span style='background:#d9f2dd;color:#166534;padding:0 2px;'>{new_chunk}</span>"
            )
        elif opcode == "replace":
            if old_chunk:
                fragments.append(
                    f"<del style='background:#fde7e9;color:#b42318;padding:0 2px;'>{old_chunk}</del>"
                )
            if new_chunk:
                fragments.append(
                    f"<span style='background:#d9f2dd;color:#166534;padding:0 2px;'>{new_chunk}</span>"
                )

    return "".join(fragments) if fragments else html.escape(revised_text or "")
