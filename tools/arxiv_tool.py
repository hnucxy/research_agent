import arxiv
import json
import re
from .base import BaseTool
from config.logger import get_logger

logger = get_logger()

class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = (
        "用于搜索 arXiv 论文，只返回标题、作者、日期、摘要和链接，不提供全文。"
    )
    prompt_spec = (
        "输出 JSON："
        '{"query":"英文检索词","max_results":5,"sort_by":"relevance|submitted_date"}。'
        "query 必须简洁、英文、可用 AND/OR 组合；默认 max_results=5。"
    )

    def run(self, params: str) -> str:
        # 清理可能残留的 Markdown 代码块标记
        clean_params = params.strip()
        clean_params = re.sub(r"^```[a-zA-Z]*\n", "", clean_params)
        clean_params = re.sub(r"\n```$", "", clean_params)

        try:
            # 解析 JSON 参数
            args = json.loads(clean_params)
            query = args.get("query", "")
            max_results = args.get("max_results", 5)
            sort_str = args.get("sort_by", "relevance")

            if not query:
                return "Arxiv 搜索出错: 未提供有效的 query 参数。"

            # print(f"    [Tool] 正在访问 arXiv 搜索: {query} (max: {max_results}, sort: {sort_str})...")
            logger.info("    [Tool] 正在访问 arXiv 搜索: %s (max: %s, sort: %s)...", query, max_results, sort_str)

            # 映射排序参数
            sort_criterion = arxiv.SortCriterion.Relevance
            if sort_str == "submitted_date":
                sort_criterion = arxiv.SortCriterion.SubmittedDate

            # 构造搜索客户端
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )

            results = []
            for r in client.results(search):
                results.append(
                    f"Title: {r.title}\n"
                    f"Authors: {', '.join([a.name for a in r.authors])}\n"
                    f"Published: {r.published.date()}\n"
                    f"Summary: {r.summary}\n"
                    f"URL: {r.entry_id}\n"
                    "---"
                )

            return "\n".join(results) if results else "未找到相关论文。建议简化搜索关键词，或去除多余的限定条件。"

        except json.JSONDecodeError:
            return f"Arxiv 搜索出错: 参数解析失败，请确保大模型输出的是合法的 JSON 字符串。收到的原始内容: {params}"
        except Exception as e:
            return f"Arxiv 搜索出错: {str(e)}"
