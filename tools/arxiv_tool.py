import arxiv
from .base import BaseTool


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = "用于搜索 arXiv 上的科研论文。输入参数为搜索关键词。"

    def run(self, query: str) -> str:
        print(f"    [Tool] 正在访问 arXiv 搜索: {query} ...")
        try:
            # 构造搜索客户端
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=3,  # 为了演示速度，只取前3篇
                sort_by=arxiv.SortCriterion.SubmittedDate  # 按提交时间排序，找最新的
            )

            results = []
            for r in client.results(search):
                results.append(
                    f"Title: {r.title}\n"
                    f"Authors: {', '.join([a.name for a in r.authors])}\n"
                    f"Published: {r.published.date()}\n"
                    f"Summary: {r.summary[:200]}...\n"  # 截断摘要
                    f"URL: {r.entry_id}\n"
                    "---"
                )

            return "\n".join(results) if results else "未找到相关论文。"

        except Exception as e:
            return f"Arxiv 搜索出错: {str(e)}"