from utils.reviewer_support import build_diff_markup, classify_review_intent


def test_classify_review_intent_relaxed_language():
    result = classify_review_intent("请帮我润色语言，消除口语化表达。")
    assert result["task_intent"] == "language_polish"
    assert result["review_mode"] == "relaxed"


def test_classify_review_intent_strict_evidence():
    result = classify_review_intent("请补充论据并完善实验结论。")
    assert result["task_intent"] == "evidence_enhancement"
    assert result["review_mode"] == "strict"


def test_build_diff_markup_contains_add_and_delete_markup():
    diff = build_diff_markup("模型效果较好。", "该模型在实验中表现更稳定。")
    assert "<del" in diff
    assert "background:#d9f2dd" in diff
