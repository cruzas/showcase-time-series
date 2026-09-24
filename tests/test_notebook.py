"""The notebook is valid and imports only what exists in src/."""

import json
import re

from tests.conftest import REPO_ROOT

NOTEBOOK = REPO_ROOT / "notebooks" / "sales_forecasting.ipynb"


def test_notebook_is_valid_json_with_code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in nb["cells"])


def test_notebook_src_imports_resolve():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in nb["cells"])
    for module, names in re.findall(r"from (src\.\w+) import (.+)", source):
        mod = __import__(module, fromlist=["_"])
        for name in names.split(","):
            assert hasattr(mod, name.strip()), f"{module}.{name.strip()}"
