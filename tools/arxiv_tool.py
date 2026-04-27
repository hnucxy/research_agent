import json
import re

import arxiv

from config.logger import get_logger
from tools.base import BaseTool

logger = get_logger()


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = "用于搜索 arXiv 论文，返回标题、作者、发布日期、摘要和链接。"
    prompt_spec = (
        '输出 JSON: {"query":"英文检索词","max_results":5,'
        '"sort_by":"relevance|submitted_date","year_start":"YYYY","year_end":"YYYY"}。'
        " 使用简洁的英文关键词，默认 max_results=5。"
        " 如用户要求“2025年”“近三年”等时间范围，请根据 current_year 换算后填写可选的 "
        "`year_start` 和 `year_end`；不要把年份限制手写进 query。"
    )

    def run(self, params: str, config: dict | None = None) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
        except json.JSONDecodeError:
            return f"Arxiv 搜索出错: invalid JSON. 原始内容: {params}"

        query = (args.get("query") or "").strip()
        max_results = args.get("max_results", 5)
        sort_str = (args.get("sort_by") or "relevance").strip()
        year_range = self._normalize_year_range(args)

        if not query:
            return "Arxiv 搜索出错: missing required field `query`."
        if year_range is None:
            return (
                "Arxiv 搜索出错: invalid `year_start` or `year_end`. "
                "Use four-digit years and ensure year_start <= year_end."
            )

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 20))
        if year_range:
            year_start, year_end = year_range
            query = (
                f"{query} AND "
                f"submittedDate:[{year_start}01010000 TO {year_end}12312359]"
            )

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
            logger.info(
                "    [Tool] arXiv request URL: %s",
                client._format_url(search, 0, client.page_size),
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

            return "\n".join(results) if results else "未找到相关论文。No results found."
        except Exception as exc:
            return f"Arxiv 搜索出错: {str(exc)}"

    @staticmethod
    def _normalize_year_range(args: dict) -> tuple[str, str] | None:
        year = args.get("year") or args.get("year_filter")
        year_start = args.get("year_start")
        year_end = args.get("year_end")

        if year and not (year_start or year_end):
            year = str(year).strip()
            if re.fullmatch(r"\d{4}", year):
                year_start = year_end = year
            elif re.fullmatch(r"\d{4}-\d{4}", year):
                year_start, year_end = year.split("-", 1)

        if year_start in (None, "") and year_end in (None, ""):
            return ()
        if year_start in (None, ""):
            year_start = year_end
        if year_end in (None, ""):
            year_end = year_start

        year_start = str(year_start).strip()
        year_end = str(year_end).strip()
        if not re.fullmatch(r"\d{4}", year_start) or not re.fullmatch(
            r"\d{4}", year_end
        ):
            return None

        start_int = int(year_start)
        end_int = int(year_end)
        if start_int > end_int or start_int < 1900 or end_int > 2100:
            return None

        return year_start, year_end
