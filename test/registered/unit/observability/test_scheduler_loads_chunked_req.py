import ast
import time
import unittest
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/observability/scheduler_metrics_mixin.py"
)


class _DisaggregationMode:
    PREFILL = "prefill"
    DECODE = "decode"


class _GetLoadsReqInput:
    def __init__(self, include=None):
        self.include = include


class _GetLoadsReqOutput(SimpleNamespace):
    pass


def _load_get_loads():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    mixin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SchedulerMetricsMixin"
    )
    function = next(
        node
        for node in mixin.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_loads"
    )
    function_source = dedent(
        "\n".join(source.splitlines()[function.lineno - 1 : function.end_lineno])
    )
    namespace = {
        "DisaggregationMode": _DisaggregationMode,
        "GetLoadsReqInput": _GetLoadsReqInput,
        "GetLoadsReqOutput": _GetLoadsReqOutput,
        "time": time,
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


def _scheduler(disaggregation_mode, chunked_req, running_reqs=None):
    return SimpleNamespace(
        running_batch=SimpleNamespace(reqs=running_reqs or []),
        chunked_req=chunked_req,
        disaggregation_mode=disaggregation_mode,
        waiting_queue=[],
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            pending_queue=[],
            queue=[],
        ),
        disagg_decode_prealloc_queue=SimpleNamespace(
            queue=[],
            retracted_queue=[],
        ),
        disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
        get_pool_stats=lambda: SimpleNamespace(get_kv_token_stats=lambda: (0, 0.0)),
        max_total_num_tokens=1024,
        dp_rank=0,
        stats=SimpleNamespace(
            gen_throughput=0.0,
            cache_hit_rate=0.0,
            utilization=-1.0,
        ),
        max_running_requests=4,
    )


class TestSchedulerLoadsChunkedReq(unittest.TestCase):
    def test_prefill_chunked_request_is_reported_as_running(self):
        get_loads = _load_get_loads()

        loads = get_loads(
            _scheduler(_DisaggregationMode.PREFILL, object()),
            _GetLoadsReqInput(include=["core"]),
        )

        self.assertEqual(loads.num_running_reqs, 1)
        self.assertEqual(loads.num_waiting_reqs, 0)

    def test_decode_does_not_double_count_unrelated_chunked_state(self):
        get_loads = _load_get_loads()

        loads = get_loads(
            _scheduler(_DisaggregationMode.DECODE, object()),
            _GetLoadsReqInput(include=["core"]),
        )

        self.assertEqual(loads.num_running_reqs, 0)

    def test_prefill_does_not_double_count_chunked_request_in_running_batch(self):
        get_loads = _load_get_loads()
        chunked_req = object()

        loads = get_loads(
            _scheduler(
                _DisaggregationMode.PREFILL,
                chunked_req,
                running_reqs=[chunked_req],
            ),
            _GetLoadsReqInput(include=["core"]),
        )

        self.assertEqual(loads.num_running_reqs, 1)


if __name__ == "__main__":
    unittest.main()
