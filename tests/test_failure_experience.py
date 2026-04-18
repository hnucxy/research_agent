from utils.failure_experience import (
    build_failure_record,
    classify_failure_type,
    should_record_failure,
)


def test_classify_failure_type_recognizes_no_result():
    failure_type = classify_failure_type("Arxiv 搜索出错: no results found")
    assert failure_type == "search_no_result"


def test_classify_failure_type_recognizes_parameter_error():
    failure_type = classify_failure_type("RAG 工具参数解析失败，请确保输入的是合法 JSON。")
    assert failure_type == "tool_parameter_error"


def test_should_record_failure_requires_repeated_parameter_error():
    assert not should_record_failure("tool_parameter_error", retry_count=1, evaluator_action="retry_step")
    assert should_record_failure("tool_parameter_error", retry_count=2, evaluator_action="retry_step")


def test_build_failure_record_returns_advice_for_arxiv_no_results():
    record = build_failure_record(
        task_input="找一篇关于 extremely niche hardware token 的论文",
        current_step="搜索 arXiv 论文",
        tool_name="arxiv_search",
        step_result="No results found.",
        feedback="请调整关键词",
        retry_count=2,
        evaluator_action="retry_step",
    )

    assert record is not None
    assert record["failure_type"] == "search_no_result"
    assert "broader English terms" in record["avoidance_advice"]
