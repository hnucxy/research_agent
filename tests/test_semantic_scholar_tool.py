import os
import sys
import unittest
from unittest.mock import patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from tools.semantic_scholar_tool import SemanticScholarSearchTool


class SemanticScholarToolTests(unittest.TestCase):
    def setUp(self):
        self.tool = SemanticScholarSearchTool()

    @patch("tools.semantic_scholar_tool.Settings.SEMANTIC_SCHOLAR_API_KEY", "test-key")
    @patch.object(SemanticScholarSearchTool, "_fetch_data")
    def test_maps_most_influential_sort_and_formats_output(self, mock_fetch):
        mock_fetch.return_value = {
            "data": [
                {
                    "title": "A Paper",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "year": 2024,
                    "citationCount": 123,
                    "influentialCitationCount": 45,
                    "url": "https://example.org/paper",
                    "abstract": "Important findings.",
                }
            ]
        }

        result = self.tool.run(
            '{"query":"multi-agent","max_results":5,"sort_by":"most_influential"}'
        )

        self.assertIn("A Paper", result)
        self.assertIn("重要引用量: 45", result)
        self.assertEqual(
            mock_fetch.call_args.args[0]["sort"],
            "influentialCitationCount:desc",
        )

    def test_requires_query(self):
        result = self.tool.run('{"max_results":5}')
        self.assertIn("缺少必填参数 `query`", result)

    def test_requires_api_key(self):
        with patch(
            "tools.semantic_scholar_tool.Settings.SEMANTIC_SCHOLAR_API_KEY", None
        ):
            result = self.tool.run('{"query":"multi-agent"}')
        self.assertIn("SEMANTIC_SCHOLAR_API_KEY", result)


if __name__ == "__main__":
    unittest.main()
