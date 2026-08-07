import ast
from pathlib import Path
from types import SimpleNamespace

import torch


_REPO_ROOT = Path(__file__).resolve().parents[4]
_EAGLE_INFO_V2 = _REPO_ROOT / "python/sglang/srt/speculative/eagle_info_v2.py"


class _ForwardModeValue:
    def __init__(self, name):
        self.name = name

    def is_idle(self):
        return self.name == "idle"


class _IndexedKernel:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, _grid):
        return self.kernel


def _load_prepare_for_v2_verify(namespace):
    tree = ast.parse(_EAGLE_INFO_V2.read_text(encoding="utf-8"))
    mixin = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EagleVerifyInputV2Mixin"
    )
    method = next(
        node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_for_v2_verify"
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(_EAGLE_INFO_V2), "exec"), namespace)
    return namespace["prepare_for_v2_verify"]


def test_hisparse_verify_slots_bind_after_cache_locs_are_assigned():
    expected_locs = torch.tensor([11, 12, 13, 14], dtype=torch.int64)

    for use_hcu_kernel in (True, False):
        events = []

        def assign_locs(*args):
            out_cache_loc = args[4]
            out_cache_loc.copy_(expected_locs)
            events.append("assign")

        class Coordinator:
            def supports_hisparse_draft_slots(self):
                return True

            def prepare_verify_slots_spec_v2(self, *, verify_cache_locs, **_kwargs):
                torch.testing.assert_close(verify_cache_locs, expected_locs)
                events.append("bind")

        forward_mode = SimpleNamespace(
            IDLE=_ForwardModeValue("idle"),
            TARGET_VERIFY=_ForwardModeValue("target_verify"),
        )
        forward_batch = SimpleNamespace(init_new=lambda batch, _runner: batch)
        namespace = {
            "CaptureHiddenMode": SimpleNamespace(NULL="null", FULL="full"),
            "ForwardBatch": forward_batch,
            "ForwardMode": forward_mode,
            "assign_extend_cache_locs": _IndexedKernel(assign_locs),
            "get_global_server_args": lambda: SimpleNamespace(enable_mamba_extra_buffer=lambda: False),
            "hcu_assign_extend_cache_locs": assign_locs,
            "next_power_of_2": lambda value: value,
            "torch": torch,
        }
        prepare_for_v2_verify = _load_prepare_for_v2_verify(namespace)

        coordinator = Coordinator()
        batch = SimpleNamespace(
            forward_mode=_ForwardModeValue("decode"),
            input_ids=torch.tensor([101, 102, 103, 104]),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            seq_lens_sum=4,
            hisparse_coordinator=coordinator,
            reqs=[],
        )
        req_to_token_pool = SimpleNamespace(req_to_token=torch.full((1, 32), -1, dtype=torch.int64))
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
                graph_runner=None,
                attn_backend=SimpleNamespace(init_forward_metadata=lambda _batch: events.append("metadata")),
            )
        )
        verify_input = SimpleNamespace(
            draft_token=torch.tensor([201, 202, 203, 204]),
            draft_token_num=4,
            use_sglang_assign_extend_cache_locs=use_hcu_kernel,
        )

        prepare_for_v2_verify(verify_input, req_to_token_pool, batch, target_worker)

        assert events == ["assign", "bind", "metadata"]
