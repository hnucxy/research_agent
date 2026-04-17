import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config.logger import get_logger
from config.settings import Settings
from .base import BaseTool

logger = get_logger()


class SemanticScholarSearchTool(BaseTool):
    name = "semantic_scholar_search"
    description = (
        "用于搜索 Semantic Scholar 论文, 返回标题、作者、年份、摘要、总引用量、重要引用量和链接。"
    )
    prompt_spec = (
        '输出 JSON: {"query":"英文检索词","max_results":5,"sort_by":"relevance|citation_count|most_influential|recency"}。'
        "使用简洁的英文关键词, 默认 max_results=5。"
        "当前端选择了 Semantic Scholar 排序方式时, 必须严格遵循。"
    )

    SORT_MAPPING = {
        "relevance": None,
        "citation_count": "citationCount:desc",
        "most_influential": "influentialCitationCount:desc",
        "recency": "publicationDate:desc",
    }

    FIELDS = (
        "title,authors,year,citationCount,influentialCitationCount,url,abstract"
    )

    def run(self, params: str) -> str:
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            args = json.loads(clean_params)
        except json.JSONDecodeError:
            return (
                "Semantic Scholar 搜索出错: JSON 参数不合法。"
                f"原始内容: {params}"
            )

        query = (args.get("query") or "").strip()
        max_results = args.get("max_results", 5)
        sort_by = (args.get("sort_by") or "relevance").strip()

        if not query:
            return "Semantic Scholar 搜索出错: 缺少必填参数 `query`。"

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 20))

        if sort_by not in self.SORT_MAPPING:
            sort_by = "relevance"

        api_key = Settings.SEMANTIC_SCHOLAR_API_KEY
        if not api_key:
            return (
                "Semantic Scholar 搜索出错: 缺少 "
                "`SEMANTIC_SCHOLAR_API_KEY`。"
            )

        params_dict = {
            "query": query,
            "limit": max_results,
            "fields": self.FIELDS,
        }
        mapped_sort = self.SORT_MAPPING.get(sort_by)
        if mapped_sort:
            params_dict["sort"] = mapped_sort

        headers = {"Authorization": f"Bearer {api_key}"}
        logger.info(
            "    [Tool] 正在访问 Semantic Scholar 搜索: %s (max: %s, sort: %s)...",
            query,
            max_results,
            sort_by,
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
            return "未找到相关论文。建议尝试更宽泛的关键词。"

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
    def _format_request_error(exc: Exception) -> str:
        if isinstance(exc, HTTPError):
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            return f"{exc}. 详情: {detail}" if detail else str(exc)
        return str(exc)
