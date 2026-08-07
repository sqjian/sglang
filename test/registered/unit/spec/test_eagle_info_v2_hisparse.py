import ast
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_CONFIG = _REPO_ROOT / "python/sglang/srt/configs/model_config.py"
_SCHEDULER_DP_ATTN_MIXIN = (
    _REPO_ROOT / "python/sglang/srt/managers/scheduler_dp_attn_mixin.py"
)
_EAGLE_DRAFT_GRAPH_RUNNER = (
    _REPO_ROOT / "python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py"
)
_EAGLE_WORKER = _REPO_ROOT / "python/sglang/srt/speculative/eagle_worker.py"
_EAGLE_INFO = _REPO_ROOT / "python/sglang/srt/speculative/eagle_info.py"
_EAGLE_INFO_V2 = _REPO_ROOT / "python/sglang/srt/speculative/eagle_info_v2.py"
_EAGLE_WORKER_V2 = _REPO_ROOT / "python/sglang/srt/speculative/eagle_worker_v2.py"


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
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EagleVerifyInputV2Mixin"
    )
    method = next(
        node
        for node in mixin.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_for_v2_verify"
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


def _load_eagle_draft_slice_single(namespace):
    tree = ast.parse(_EAGLE_INFO.read_text(encoding="utf-8"))
    eagle_draft_input = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EagleDraftInput"
    )
    method = next(
        node
        for node in eagle_draft_input.body
        if isinstance(node, ast.FunctionDef) and node.name == "slice_single"
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
    exec(compile(module, str(_EAGLE_INFO), "exec"), namespace)
    return namespace["slice_single"]


def _load_module_function(path, function_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def _load_class_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
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
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def test_hisparse_disables_cross_step_mtp_topk_index_sharing():
    is_mtp_index_share_enabled = _load_module_function(
        _MODEL_CONFIG,
        "is_mtp_index_share_enabled",
        {},
    )
    hf_config = SimpleNamespace(index_share_for_mtp_iteration=True)

    assert is_mtp_index_share_enabled(hf_config)
    assert not is_mtp_index_share_enabled(hf_config, enable_hisparse=True)


def test_hisparse_draft_graph_does_not_require_cross_step_seed():
    can_run = _load_class_method(
        _EAGLE_DRAFT_GRAPH_RUNNER,
        "EAGLEDraftCudaGraphRunner",
        "can_run",
        {
            "torch": torch,
            "is_mtp_index_share_enabled": (
                lambda config, *, enable_hisparse=False: bool(
                    config.index_share_for_mtp_iteration and not enable_hisparse
                )
            ),
        },
    )
    hf_config = SimpleNamespace(index_share_for_mtp_iteration=True)
    runner = SimpleNamespace(
        require_mlp_tp_gather=False,
        disable_padding=True,
        graphs={1: object()},
        max_bs=1,
        require_mlp_sync=False,
        enable_mtp_index_share=True,
        mtp_index_share_topk=2,
        topk=1,
        model_runner=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
    )
    active_mode = SimpleNamespace(is_idle=lambda: False)
    forward_batch = SimpleNamespace(
        batch_size=1,
        forward_mode=active_mode,
        spec_info=SimpleNamespace(mtp_topk_indices=None),
        hisparse_coordinator=object(),
    )

    assert can_run(runner, forward_batch)

    forward_batch.hisparse_coordinator = None
    assert not can_run(runner, forward_batch)
    forward_batch.spec_info.mtp_topk_indices = torch.zeros((1, 2), dtype=torch.int32)
    assert can_run(runner, forward_batch)


def test_hisparse_dp_graph_sync_does_not_require_cross_step_seed():
    captured = {}

    def prepare_mlp_sync_batch_raw(batch, **kwargs):
        captured.update(kwargs)
        return batch

    prepare_mlp_sync_batch = _load_class_method(
        _SCHEDULER_DP_ATTN_MIXIN,
        "SchedulerDPAttnMixin",
        "prepare_mlp_sync_batch",
        {
            "is_mtp_index_share_enabled": (
                lambda config, *, enable_hisparse=False: bool(
                    config.index_share_for_mtp_iteration and not enable_hisparse
                )
            ),
            "prepare_mlp_sync_batch_raw": prepare_mlp_sync_batch_raw,
            "require_mlp_tp_gather": lambda _server_args: False,
        },
    )
    server_args = SimpleNamespace(
        disaggregation_mode="decode",
        enable_dp_attention=False,
        dp_size=32,
        disable_cuda_graph=False,
        disable_overlap_schedule=False,
        enable_hisparse=True,
        speculative_eagle_topk=1,
    )
    scheduler = SimpleNamespace(
        server_args=server_args,
        attn_tp_size=1,
        attn_cp_size=1,
        tp_group=object(),
        get_idle_batch=lambda: None,
        offload_tags=set(),
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_share_for_mtp_iteration=True)
        ),
    )

    prepare_mlp_sync_batch(scheduler, None)

    assert captured["mtp_index_share_for_topk1"] is False


def test_hisparse_draft_extend_does_not_publish_cross_step_seed():
    capture_mtp_topk_indices = _load_class_method(
        _EAGLE_WORKER_V2,
        "EagleDraftWorker",
        "_capture_mtp_topk_indices",
        {
            "is_mtp_index_share_enabled": (
                lambda config, *, enable_hisparse=False: bool(
                    config.index_share_for_mtp_iteration and not enable_hisparse
                )
            ),
            "torch": torch,
        },
    )
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(index_share_for_mtp_iteration=True)
            )
        )
    )
    next_draft_input = SimpleNamespace(mtp_topk_indices=None)
    forward_batch = SimpleNamespace(
        topk_indices=torch.tensor([[10, 20]], dtype=torch.int32),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        hisparse_coordinator=object(),
        extend_seq_lens=torch.tensor([1], dtype=torch.int32),
    )

    capture_mtp_topk_indices(worker, next_draft_input, forward_batch)

    assert next_draft_input.mtp_topk_indices is None


def test_hisparse_skips_unreplayable_draft_extend_graph_capture():
    captures = []

    def capture_draft_extend(worker):
        captures.append(worker)
        return "captured"

    init_cuda_graphs = _load_class_method(
        _EAGLE_WORKER,
        "EAGLEWorker",
        "init_cuda_graphs",
        {
            "EAGLEDraftCudaGraphRunner": object,
            "EAGLEDraftExtendCudaGraphRunner": capture_draft_extend,
            "EAGLEDraftNpuGraphRunner": object,
            "_is_npu": False,
            "get_available_gpu_memory": lambda *_args: 8.0,
            "log_info_on_rank0": lambda *_args: None,
            "logger": object(),
            "time": SimpleNamespace(perf_counter=lambda: 0.0),
        },
    )

    def make_worker(hisparse_coordinator):
        return SimpleNamespace(
            server_args=SimpleNamespace(disable_cuda_graph=False),
            speculative_num_steps=1,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    hisparse_coordinator=hisparse_coordinator,
                ),
                device="cuda",
            ),
            draft_extend_attn_backend=object(),
            device="cuda",
            gpu_id=0,
        )

    hisparse_worker = make_worker(object())
    init_cuda_graphs(hisparse_worker)
    assert hisparse_worker.cuda_graph_runner_for_draft_extend is None
    assert captures == []

    dense_worker = make_worker(None)
    init_cuda_graphs(dense_worker)
    assert dense_worker.cuda_graph_runner_for_draft_extend == "captured"
    assert captures == [dense_worker]


def test_hisparse_v2_skips_unreplayable_draft_extend_graph_capture():
    captures = []

    def capture_draft_extend(worker):
        captures.append(worker)
        return "captured"

    init_cuda_graphs = _load_class_method(
        _EAGLE_WORKER_V2,
        "EagleDraftWorker",
        "init_cuda_graphs",
        {
            "EAGLEDraftCudaGraphRunner": object,
            "EAGLEDraftExtendCudaGraphRunner": capture_draft_extend,
            "EAGLEDraftExtendNpuGraphRunner": object,
            "EAGLEDraftNpuGraphRunner": object,
            "TRTLLMMLABackend": type("TRTLLMMLABackend", (), {}),
            "TritonAttnBackend": type("TritonAttnBackend", (), {}),
            "_is_cuda": True,
            "_is_hcu": False,
            "_is_hip": False,
            "_is_musa": False,
            "_is_npu": False,
            "_is_nsa_attn_backend": lambda _backend: True,
            "get_available_gpu_memory": lambda *_args: 8.0,
            "log_info_on_rank0": lambda *_args: None,
            "logger": object(),
            "time": SimpleNamespace(perf_counter=lambda: 0.0),
        },
    )

    def make_worker(hisparse_coordinator):
        return SimpleNamespace(
            server_args=SimpleNamespace(
                disable_cuda_graph=False,
                model_impl="auto",
                speculative_attention_mode="prefill",
            ),
            speculative_num_steps=1,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    hisparse_coordinator=hisparse_coordinator,
                ),
                device="cuda",
            ),
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            device="cuda",
            gpu_id=0,
        )

    hisparse_worker = make_worker(object())
    init_cuda_graphs(hisparse_worker)
    assert hisparse_worker.cuda_graph_runner_for_draft_extend is None
    assert captures == []

    dense_worker = make_worker(None)
    init_cuda_graphs(dense_worker)
    assert dense_worker.cuda_graph_runner_for_draft_extend == "captured"
    assert captures == [dense_worker]


def test_hisparse_draft_slice_preserves_mtp_topk_seed():
    slice_single = _load_eagle_draft_slice_single(
        {
            "EagleDraftInput": lambda **kwargs: SimpleNamespace(**kwargs),
            "FutureIndices": lambda **kwargs: SimpleNamespace(**kwargs),
            "_slice_optional_tensor": (
                lambda value, index: None if value is None else value[index]
            ),
        }
    )
    draft_input = SimpleNamespace(
        topk_p=torch.tensor([[0.1], [0.2]]),
        topk_index=torch.tensor([[1], [2]]),
        hidden_states=torch.tensor([[3.0], [4.0]]),
        capture_hidden_mode="last",
        bonus_tokens=torch.tensor([5, 6]),
        mtp_topk_indices=torch.tensor([[101, 102], [201, 202]], dtype=torch.int32),
        future_indices=None,
        new_seq_lens=torch.tensor([7, 8]),
        verify_done=None,
        num_correct_drafts=torch.tensor([1, 2]),
        num_accept_tokens=torch.tensor([2, 3]),
    )

    sliced = slice_single(draft_input, 1)

    torch.testing.assert_close(
        sliced.mtp_topk_indices,
        torch.tensor([[201, 202]], dtype=torch.int32),
    )


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
            "get_global_server_args": lambda: SimpleNamespace(
                enable_mamba_extra_buffer=lambda: False
            ),
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
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.full((1, 32), -1, dtype=torch.int64)
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
                graph_runner=None,
                attn_backend=SimpleNamespace(
                    init_forward_metadata=lambda _batch: events.append("metadata")
                ),
            )
        )
        verify_input = SimpleNamespace(
            draft_token=torch.tensor([201, 202, 203, 204]),
            draft_token_num=4,
            use_sglang_assign_extend_cache_locs=use_hcu_kernel,
        )

        prepare_for_v2_verify(verify_input, req_to_token_pool, batch, target_worker)

        assert events == ["assign", "bind", "metadata"]
