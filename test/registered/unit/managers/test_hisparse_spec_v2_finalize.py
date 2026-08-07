import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _stub_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_hisparse_coordinator(monkeypatch):
    class Placeholder:
        pass

    class DeviceModule:
        Event = object

    for package in (
        "sglang",
        "sglang.jit_kernel",
        "sglang.srt",
        "sglang.srt.managers",
        "sglang.srt.mem_cache",
    ):
        module = _stub_module(monkeypatch, package)
        module.__path__ = []

    _stub_module(
        monkeypatch,
        "sglang.jit_kernel.hisparse",
        load_cache_to_device_buffer_dsv4_mla=lambda *args, **kwargs: None,
        load_cache_to_device_buffer_mla=lambda *args, **kwargs: None,
    )
    _stub_module(monkeypatch, "sglang.srt.managers.schedule_batch", Req=Placeholder)
    _stub_module(
        monkeypatch,
        "sglang.srt.mem_cache.hisparse_memory_pool",
        DeepSeekV4HiSparseTokenToKVPoolAllocator=Placeholder,
        DeepSeekV4SingleKVPoolHost=Placeholder,
        HiSparseNSATokenToKVPool=Placeholder,
        HiSparseTokenToKVPoolAllocator=Placeholder,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.mem_cache.memory_pool_host",
        MLATokenToKVPoolHost=Placeholder,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.mem_cache.memory_pool",
        ReqToTokenPool=Placeholder,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.utils",
        get_device_module=lambda: DeviceModule,
    )

    vendor_root = Path(__file__).resolve().parents[4]
    source = vendor_root / "python/sglang/srt/managers/hisparse_coordinator.py"
    spec = importlib.util.spec_from_file_location(
        "hisparse_coordinator_under_test", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HiSparseCoordinator


def test_spec_v2_short_finalize_avoids_scalar_extraction(monkeypatch):
    class RejectScalarExtraction(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError(
                    "short spec-v2 finalize must not extract GPU scalars"
                )
            return func(*args, **(kwargs or {}))

    coordinator_type = _load_hisparse_coordinator(monkeypatch)
    coordinator = coordinator_type.__new__(coordinator_type)
    coordinator.is_dsv4_hisparse = False
    coordinator.device_buffer_size = 16
    coordinator._pending_draft_extend_backup = None
    coordinator._skip_first_backup = [False] * 8

    mapping = torch.zeros(32, dtype=torch.int64)
    verify_cache_locs = torch.arange(4, 12, dtype=torch.int64)
    mapping[verify_cache_locs] = torch.arange(100, 108, dtype=torch.int64)
    coordinator.token_to_kv_pool_allocator = SimpleNamespace(
        full_to_hisparse_device_index_mapping=mapping
    )

    with RejectScalarExtraction():
        coordinator.finalize_accepted_tokens_spec_v2(
            req_pool_indices=torch.tensor([1, 3], dtype=torch.int64),
            req_pool_indices_cpu=[1, 3],
            seq_lens=torch.tensor([3, 6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3, 6], dtype=torch.int64),
            verify_cache_locs=verify_cache_locs,
            accept_index=torch.tensor(
                [[0, 2, -1, -1], [4, 7, -1, -1]], dtype=torch.int64
            ),
        )

    assert torch.equal(
        mapping[verify_cache_locs],
        torch.tensor([100, 0, 102, 0, 104, 0, 0, 107], dtype=torch.int64),
    )
    assert coordinator._skip_first_backup[1]
    assert coordinator._skip_first_backup[3]


def test_spec_v2_long_finalize_keeps_transactional_fallback(monkeypatch):
    coordinator_type = _load_hisparse_coordinator(monkeypatch)
    coordinator = coordinator_type.__new__(coordinator_type)
    coordinator.is_dsv4_hisparse = False
    coordinator.device_buffer_size = 16
    coordinator.finalize_accepted_tokens = Mock()

    verify_cache_locs = torch.arange(4, dtype=torch.int64)
    coordinator.finalize_accepted_tokens_spec_v2(
        req_pool_indices=torch.tensor([2], dtype=torch.int64),
        req_pool_indices_cpu=[2],
        seq_lens=torch.tensor([14], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([14], dtype=torch.int64),
        verify_cache_locs=verify_cache_locs,
        accept_index=torch.tensor([[0, -1, -1, -1]], dtype=torch.int64),
    )

    call = coordinator.finalize_accepted_tokens.call_args.kwargs
    assert torch.equal(call["accepted_cache_locs"], verify_cache_locs[:1])
    assert torch.equal(call["draft_cache_locs"], verify_cache_locs)
    assert torch.equal(call["num_correct_drafts"], torch.tensor([0]))
    assert torch.equal(call["num_correct_drafts_cpu"], torch.tensor([0]))
    assert torch.equal(call["accepted_token_positions"], torch.tensor([14]))
