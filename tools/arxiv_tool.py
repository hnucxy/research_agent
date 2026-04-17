import json
import re

import arxiv

from config.logger import get_logger
from .base import BaseTool

logger = get_logger()


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = (
        "用于搜索 arXiv 论文, 返回标题、作者、发布日期、摘要和链接。"
    )
    prompt_spec = (
        '输出 JSON: {"query":"英文检索词","max_results":5,"sort_by":"relevance|submitted_date"}。'
        "使用简洁的英文关键词, 默认 max_results=5。"
    )

    def run(self, params: str) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
        except json.JSONDecodeError:
            return f"Arxiv 搜索出错: JSON 参数不合法。原始内容: {params}"

        query = (args.get("query") or "").strip()
        max_results = args.get("max_results", 5)
        sort_str = (args.get("sort_by") or "relevance").strip()

        if not query:
            return "Arxiv 搜索出错: 缺少必填参数 `query`。"

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 20))

        logger.info(
            "    [Tool] 正在访问 arXiv 搜索: %s (max: %s, sort: %s)...",
            query,
            max_results,
            sort_str,
        )

        sort_criterion = arxiv.SortCriterion.Relevance
        if sort_str == "submitted_date":
            sort_criterion = arxiv.SortCriterion.SubmittedDate

        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion,
            )

            results = []
            for paper in client.results(search):
                results.append(
                    f"标题: {paper.title}\n"
                    f"作者: {', '.join(author.name for author in paper.authors)}\n"
                    f"发布日期: {paper.published.date()}\n"
                    f"摘要: {paper.summary}\n"
                    f"URL: {paper.entry_id}\n"
                    "---"
                )

            return "\n".join(results) if results else "未找到相关论文。"
        except Exception as exc:
            return f"Arxiv 搜索出错: {str(exc)}"
