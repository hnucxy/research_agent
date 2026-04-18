import os
import sys
import shutil

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def workspace_tmp_path(request):
    base_dir = os.path.join(PROJECT_ROOT, "tests_tmp_runtime", request.node.name)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    try:
        yield base_dir
    finally:
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
