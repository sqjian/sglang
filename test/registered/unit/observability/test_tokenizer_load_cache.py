import ast
import asyncio
import copy
import unittest
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/tokenizer_control_mixin.py"
TOKENIZER_MANAGER_SOURCE = SOURCE.with_name("tokenizer_manager.py")
OUTPUT_PROCESSOR_SOURCE = SOURCE.with_name("scheduler_output_processor_mixin.py")


class _GetLoadsReqInput:
    def __init__(self, include=None, dp_rank=None):
        self.include = include
        self.dp_rank = dp_rank


def _load_complete_cached_loads():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "complete_cached_loads"
    )
    function_source = dedent("\n".join(source.splitlines()[function.lineno - 1 : function.end_lineno]))
    namespace = {"Any": object, "copy": copy}
    exec(
        compile(
            "from __future__ import annotations\n" + function_source,
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["complete_cached_loads"]


def _load_get_loads():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    mixin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TokenizerControlMixin")
    function = next(node for node in mixin.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_loads")
    function_source = dedent("\n".join(source.splitlines()[function.lineno - 1 : function.end_lineno]))
    namespace = {
        "GetLoadsReqInput": _GetLoadsReqInput,
        "GetLoadsReqOutput": object,
        "List": list,
        "Optional": object,
        "complete_cached_loads": _load_complete_cached_loads(),
    }
    exec(
        compile(
            "from __future__ import annotations\n" + function_source,
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["get_loads"]


class TestTokenizerLoadCache(unittest.TestCase):
    def test_complete_cache_is_sorted_and_copied(self):
        complete_cached_loads = _load_complete_cached_loads()
        rank_one = SimpleNamespace(dp_rank=1, num_running_reqs=2)
        rank_zero = SimpleNamespace(dp_rank=0, num_running_reqs=1)

        result = complete_cached_loads({1: rank_one, 0: rank_zero}, dp_size=2)

        self.assertEqual([load.dp_rank for load in result], [0, 1])
        self.assertEqual([load.num_running_reqs for load in result], [1, 2])
        self.assertIsNot(result[0], rank_zero)

    def test_incomplete_or_non_contiguous_cache_falls_back(self):
        complete_cached_loads = _load_complete_cached_loads()

        self.assertIsNone(complete_cached_loads({0: SimpleNamespace(dp_rank=0)}, dp_size=2))
        self.assertIsNone(
            complete_cached_loads(
                {
                    0: SimpleNamespace(dp_rank=0),
                    2: SimpleNamespace(dp_rank=2),
                },
                dp_size=2,
            )
        )

    def test_endpoint_uses_complete_cache_without_fanout(self):
        get_loads = _load_get_loads()
        cached = {rank: SimpleNamespace(dp_rank=rank, num_running_reqs=rank + 1) for rank in range(2)}

        async def unexpected_fanout(_request):
            raise AssertionError("complete cache must not fan out")

        manager = SimpleNamespace(
            scheduler_load_cache=cached,
            server_args=SimpleNamespace(dp_size=2),
            get_loads_communicator=unexpected_fanout,
            auto_create_handle_loop=lambda: None,
        )

        result = asyncio.run(get_loads(manager, include=["all"]))

        self.assertEqual([load.dp_rank for load in result], [0, 1])
        self.assertIsNot(result[0], cached[0])

    def test_scheduler_output_refreshes_full_cache_contract(self):
        tokenizer_source = TOKENIZER_MANAGER_SOURCE.read_text(encoding="utf-8")
        output_source = OUTPUT_PROCESSOR_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            "self.scheduler_load_cache[recv_obj.load.dp_rank] = recv_obj.load",
            tokenizer_source,
        )
        self.assertIn('GetLoadsReqInput(include=["all"])', output_source)


if __name__ == "__main__":
    unittest.main()
