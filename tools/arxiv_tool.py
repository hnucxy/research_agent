import arxiv
import json
import re
from .base import BaseTool


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    # description = "用于搜索 arXiv 上的科研论文。输入参数为搜索关键词。"
    # description = "用于搜索 arXiv 上的科研论文。输入参数必须是提炼后的【英文】搜索关键词（如: LLM AND medical diagnosis），不要输入长句或中文。"

    description = (
        "用于搜索 arXiv 上的科研论文。输入参数必须是一个合法的 JSON 字符串，包含以下字段：\n"
        "- query: (必填) 提炼后的【英文】搜索关键词。请使用双引号确保精确匹配（如: \"large language model\" AND \"medical\"）。如果用户需要特定年份，可以适当加入年份关键词。\n"
        "- max_results: (可选) 返回的文献数量，默认为 5。如果用户要求广泛调研，可以设置为 10 或 20。\n"
        "- sort_by: (可选) 排序方式。可选值为 'relevance' (相关性优先，默认) 或 'submitted_date' (最新提交时间优先)。"
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

            print(f"    [Tool] 正在访问 arXiv 搜索: {query} (max: {max_results}, sort: {sort_str})...")

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