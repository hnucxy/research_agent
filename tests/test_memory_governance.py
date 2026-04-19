import json
from pathlib import Path

from utils.memory_governance import (
    _build_research_rows,
    _count_failure_types,
    append_memory_audit_log,
)


def test_build_research_rows_groups_chunks_by_file_hash():
    rows = _build_research_rows(
        [
            {
                "file_hash": "hash-a",
                "file_name": "paper-a.md",
                "file_path": "/tmp/paper-a.md",
                "chat_id": "chat-1",
                "type": "text",
            },
            {
                "file_hash": "hash-a",
                "file_name": "paper-a.md",
                "file_path": "/tmp/paper-a.md",
                "chat_id": "chat-1",
                "type": "image",
            },
            {
                "file_hash": "hash-b",
                "file_name": "paper-b.md",
                "file_path": "/tmp/paper-b.md",
                "chat_id": "chat-2",
                "type": "text",
            },
        ]
    )

    assert len(rows) == 2
    assert rows[0]["entry_id"] == "hash-a"
    assert rows[0]["文本块数"] == 1
    assert rows[0]["图片块数"] == 1


def test_count_failure_types_returns_frequency_table():
    breakdown = _count_failure_types(
        [
            {"failure_type": "search_no_result"},
            {"failure_type": "tool_parameter_error"},
            {"failure_type": "search_no_result"},
        ]
    )

    assert breakdown[0] == {"失败类型": "search_no_result", "数量": 2}
    assert breakdown[1] == {"失败类型": "tool_parameter_error", "数量": 1}


def test_append_memory_audit_log_writes_json_line():
    audit_dir = Path("tests_tmp_runtime") / "memory_governance"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "memory_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()

    append_memory_audit_log(
        action="single_delete",
        collection_key="experience",
        entry_ids=["id-1"],
        deleted_entries=1,
        deleted_vectors=1,
        actor="chat:test",
        audit_log_path=str(audit_path),
    )

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["action"] == "single_delete"
    assert payload["collection_key"] == "experience"
    assert payload["deleted_entries"] == 1
    assert payload["actor"] == "chat:test"
