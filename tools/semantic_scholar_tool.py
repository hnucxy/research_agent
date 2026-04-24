import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config.logger import get_logger
from config.settings import Settings
from tools.base import BaseTool

logger = get_logger()


class SemanticScholarSearchTool(BaseTool):
    name = "semantic_scholar_search"
    description = (
        "用于搜索 Semantic Scholar 论文，返回标题、作者、年份、摘要、总引用量、重要引用量和链接。"
    )
    prompt_spec = (
        'Output JSON: {"query":"English search terms","max_results":5,'
        '"sort_by":"relevance|citation_count|most_influential|recency",'
        '"year":"2020-2024"}。'
        " Use concise English keywords. Default max_results=5. "
        "If the frontend selected a Semantic Scholar sort order or year filter, "
        "strictly follow it. The year filter supports YYYY, YYYY-YYYY, YYYY-, or -YYYY."
    )

    SORT_MAPPING = {
        "relevance": None,
        "citation_count": "citationCount:desc",
        "most_influential": "influentialCitationCount:desc",
        "recency": "publicationDate:desc",
    }

    FIELDS = "title,authors,year,citationCount,influentialCitationCount,url,abstract"

    def run(self, params: str) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
        except json.JSONDecodeError:
            return f"Semantic Scholar 搜索出错: invalid JSON. 原始内容: {params}"

        query = (args.get("query") or "").strip()
        max_results = args.get("max_results", 5)
        sort_by = (args.get("sort_by") or "relevance").strip()
        year_filter = self._normalize_year_filter(args.get("year") or args.get("year_filter"))

        if not query:
            return "Semantic Scholar 搜索出错: missing required field `query`."

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 20))

        if sort_by not in self.SORT_MAPPING:
            sort_by = "relevance"

        if year_filter is None:
            return (
                "Semantic Scholar 搜索出错: invalid `year`. "
                "Use YYYY, YYYY-YYYY, YYYY-, or -YYYY."
            )

        api_key = Settings.SEMANTIC_SCHOLAR_API_KEY
        if not api_key:
            return "Semantic Scholar 搜索出错: missing `SEMANTIC_SCHOLAR_API_KEY`."

        params_dict = {
            "query": query,
            "limit": max_results,
            "fields": self.FIELDS,
        }
        mapped_sort = self.SORT_MAPPING.get(sort_by)
        if mapped_sort:
            params_dict["sort"] = mapped_sort
        if year_filter:
            params_dict["year"] = year_filter

        headers = {"Authorization": f"Bearer {api_key}"}
        logger.info(
            "    [Tool] 正在访问 Semantic Scholar 搜索: %s (max: %s, sort: %s, year: %s)...",
            query,
            max_results,
            sort_by,
            year_filter or "any",
        )

        try:
            data = self._fetch_data(params_dict, headers)
        except HTTPError as exc:
            should_retry_without_sort = mapped_sort is not None and exc.code == 400
            if should_retry_without_sort:
                fallback_params = dict(params_dict)
                fallback_params.pop("sort", None)
                try:
                    data = self._fetch_data(fallback_params, headers)
                except (HTTPError, URLError) as retry_exc:
                    return f"Semantic Scholar 搜索出错: {self._format_request_error(retry_exc)}"
            else:
                return f"Semantic Scholar 搜索出错: {self._format_request_error(exc)}"
        except URLError as exc:
            return f"Semantic Scholar 搜索出错: {self._format_request_error(exc)}"

        papers = data.get("data", [])
        if not papers:
            return "未找到相关论文。No results found. 建议尝试更宽泛的关键词。"

        if sort_by == "citation_count":
            papers = sorted(
                papers,
                key=lambda paper: paper.get("citationCount") or 0,
                reverse=True,
            )
        elif sort_by == "most_influential":
            papers = sorted(
                papers,
                key=lambda paper: paper.get("influentialCitationCount") or 0,
                reverse=True,
            )
        elif sort_by == "recency":
            papers = sorted(
                papers,
                key=lambda paper: paper.get("year") or 0,
                reverse=True,
            )

        results = []
        for paper in papers:
            authors = ", ".join(
                author.get("name", "未知")
                for author in paper.get("authors", [])
                if author.get("name")
            ) or "未知"
            results.append(
                f"标题: {paper.get('title', 'N/A')}\n"
                f"作者: {authors}\n"
                f"年份: {paper.get('year', 'N/A')}\n"
                f"总引用量: {paper.get('citationCount', 'N/A')}\n"
                f"重要引用量: {paper.get('influentialCitationCount', 'N/A')}\n"
                f"摘要: {paper.get('abstract', 'N/A')}\n"
                f"URL: {paper.get('url', 'N/A')}\n"
                "---"
            )

        return "\n".join(results)

    def _fetch_data(self, params_dict: dict, headers: dict) -> dict:
        query_string = urlencode(params_dict)
        request = Request(
            f"{Settings.SEMANTIC_SCHOLAR_BASE_URL}?{query_string}",
            headers=headers,
            method="GET",
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _normalize_year_filter(value) -> str | None:
        if value is None:
            return ""
        if isinstance(value, int):
            value = str(value)
        value = str(value).strip()
        if not value:
            return ""
        if re.fullmatch(r"\d{4}", value):
            return value
        if re.fullmatch(r"\d{4}-\d{4}", value):
            start, end = value.split("-", 1)
            return value if int(start) <= int(end) else None
        if re.fullmatch(r"\d{4}-", value) or re.fullmatch(r"-\d{4}", value):
            return value
        return None

    @staticmethod
    def _format_request_error(exc: Exception) -> str:
        if isinstance(exc, HTTPError):
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            return f"{exc}. 详情: {detail}" if detail else str(exc)
        return str(exc)
