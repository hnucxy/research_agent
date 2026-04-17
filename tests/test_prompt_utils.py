import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.prompt_utils import build_resource_context


class PromptUtilsTests(unittest.TestCase):
    def test_includes_semantic_search_preferences(self):
        context = build_resource_context(
            search_source="semantic_scholar",
            semantic_sort_by="citation_count",
        )

        self.assertIn("search_source: semantic_scholar", context)
        self.assertIn("semantic_sort_by: citation_count", context)
        self.assertIn("semantic_scholar_search", context)

    def test_includes_arxiv_constraint(self):
        context = build_resource_context(search_source="arxiv")
        self.assertIn("search_source: arxiv", context)
        self.assertIn("arxiv_search", context)


if __name__ == "__main__":
    unittest.main()
