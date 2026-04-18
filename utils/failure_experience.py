from typing import Dict, Iterable, List, Optional, Sequence

from config.settings import Settings


NO_RESULT_PATTERNS = (
    "\u672a\u627e\u5230\u76f8\u5173\u8bba\u6587",
    "\u672a\u68c0\u7d22\u5230\u76f8\u5173\u5185\u5bb9",
    "no results",
    "no result",
    "no papers found",
)

PARAMETER_ERROR_PATTERNS = (
    "\u53c2\u6570\u89e3\u6790\u5931\u8d25",
    "\u53c2\u6570\u4e0d\u80fd\u4e3a\u7a7a",
    "\u7f3a\u5c11\u5fc5\u586b\u53c2\u6570",
    "\u5408\u6cd5 json",
    "json",
    "json decode",
    "invalid json",
    "missing required",
)

NETWORK_ERROR_PATTERNS = (
    "timed out",
    "timeout",
    "connection",
    "http error",
    "url error",
    "api",
)


def classify_failure_type(step_result: str, feedback: str = "") -> str:
    text = f"{step_result}\n{feedback}".lower()
    if any(pattern in text for pattern in NO_RESULT_PATTERNS):
        return "search_no_result"
    if any(pattern in text for pattern in PARAMETER_ERROR_PATTERNS):
        return "tool_parameter_error"
    if any(pattern in text for pattern in NETWORK_ERROR_PATTERNS):
        return "tool_or_api_error"
    return "generic_failure"


def build_avoidance_advice(tool_name: str, failure_type: str) -> str:
    if failure_type == "search_no_result":
        if tool_name == "arxiv_search":
            return (
                "Avoid niche or over-constrained arXiv keywords. Rewrite the query with 2-5 "
                "broader English terms, drop long phrases, and retry once before changing strategy."
            )
        return (
            "Broaden the retrieval query, keep only core concepts, and avoid stacking too many "
            "constraints in a single search."
        )
    if failure_type == "tool_parameter_error":
        return (
            "Use the tool's exact JSON schema, keep field names unchanged, and avoid wrapping the "
            "payload with explanation text or markdown fences."
        )
    if failure_type == "tool_or_api_error":
        return (
            "Treat this as an external dependency failure. Retry with simpler parameters and, if it "
            "persists, switch to a fallback tool or report the limitation explicitly."
        )
    return (
        "Do not repeat the same failing step unchanged. Adjust the query, parameters, or tool choice "
        "before retrying."
    )


def should_record_failure(
    failure_type: str, retry_count: int, evaluator_action: str
) -> bool:
    if evaluator_action == "replan":
        return True
    if failure_type in {"search_no_result", "tool_parameter_error"} and retry_count >= 2:
        return True
    if failure_type == "tool_or_api_error" and retry_count >= 2:
        return True
    return False


def build_failure_record(
    task_input: str,
    current_step: str,
    tool_name: str,
    step_result: str,
    feedback: str,
    retry_count: int,
    evaluator_action: str,
) -> Optional[Dict[str, str]]:
    failure_type = classify_failure_type(step_result, feedback)
    if not should_record_failure(failure_type, retry_count, evaluator_action):
        return None

    advice = build_avoidance_advice(tool_name, failure_type)
    record = {
        "failure_type": failure_type,
        "tool_name": tool_name,
        "task_input": task_input or "",
        "current_step": current_step or "",
        "step_result": step_result or "",
        "feedback": feedback or "",
        "retry_count": str(retry_count),
        "evaluator_action": evaluator_action or "",
        "avoidance_advice": advice,
    }
    return record


def format_failure_record(record: Dict[str, str]) -> str:
    return (
        "[Failure Memory]\n"
        f"Tool: {record.get('tool_name', '')}\n"
        f"Failure Type: {record.get('failure_type', '')}\n"
        f"User Task: {record.get('task_input', '')}\n"
        f"Plan Step: {record.get('current_step', '')}\n"
        f"Observed Failure: {record.get('step_result', '')}\n"
        f"Evaluator Feedback: {record.get('feedback', '')}\n"
        f"Attempts Before Capture: {record.get('retry_count', '')}\n"
        f"Avoidance Advice: {record.get('avoidance_advice', '')}"
    )


def store_failure_record(record: Dict[str, str], embeddings) -> None:
    from langchain_chroma import Chroma

    record_text = format_failure_record(record)
    vectorstore = Chroma(
        collection_name=Settings.get_collection_name("global_failure_experience"),
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    existing_docs = vectorstore.similarity_search_with_score(record_text, k=1)
    if existing_docs:
        _, score = existing_docs[0]
        if score < 0.12:
            return

    vectorstore.add_texts(
        texts=[record_text],
        metadatas=[
            {
                "tool_name": record.get("tool_name", ""),
                "failure_type": record.get("failure_type", ""),
            }
        ],
    )


def format_failure_warnings(
    documents: Sequence, preferred_tools: Optional[Iterable[str]] = None
) -> str:
    if not documents:
        return "无"

    preferred = set(preferred_tools or [])
    warnings: List[str] = []
    seen = set()

    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        tool_name = metadata.get("tool_name", "")
        if preferred and tool_name and tool_name not in preferred:
            continue

        advice_line = ""
        for line in (document.page_content or "").splitlines():
            if line.startswith("Avoidance Advice:"):
                advice_line = line.replace("Avoidance Advice:", "").strip()
                break

        if not advice_line:
            continue

        warning = f"- [{tool_name or 'unknown_tool'}] {advice_line}"
        if warning not in seen:
            seen.add(warning)
            warnings.append(warning)

    return "\n".join(warnings) if warnings else "无"


def retrieve_failure_warnings(
    query: str, embeddings, preferred_tools: Optional[Iterable[str]] = None, k: int = 4
) -> str:
    from langchain_chroma import Chroma

    if not query:
        return "无"

    vectorstore = Chroma(
        collection_name=Settings.get_collection_name("global_failure_experience"),
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    documents = vectorstore.similarity_search(query, k=k)
    return format_failure_warnings(documents, preferred_tools=preferred_tools)
