import ast
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MODULE_PATH = (
    Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/moe/ep_moe/layer.py"
)
OPTIONAL_SYMBOL = "m_grouped_w4a8_gemm_nt_contiguous_hipc"


def test_contiguous_hipc_symbol_is_imported_as_an_optional_capability():
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)

    unconditional_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "deepgemm"
    ]
    assert all(
        alias.name != OPTIONAL_SYMBOL
        for node in unconditional_imports
        for alias in node.names
    )

    guarded_import = any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "deepgemm"
        and any(alias.name == OPTIONAL_SYMBOL for alias in statement.names)
        for node in tree.body
        if isinstance(node, ast.Try)
        for statement in node.body
    )
    assert guarded_import
    assert "SGLANG_USE_W4A8_CONTIGUOUS_HIPC requires deepgemm" in source
