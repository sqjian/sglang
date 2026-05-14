import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import (  # noqa: E402
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator  # noqa: E402
from sglang.srt.mem_cache.hisparse_memory_pool import (  # noqa: E402
    HiSparseTokenToKVPoolAllocator,
)

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestHiSparseDecodeAdmission(unittest.TestCase):
    def test_allocator_oom_preserves_mapping(self):
        allocator = HiSparseTokenToKVPoolAllocator.__new__(
            HiSparseTokenToKVPoolAllocator
        )
        allocator.page_size = 4
        allocator.device = "cpu"
        allocator.full_to_hisparse_device_index_mapping = torch.zeros(
            16, dtype=torch.int64
        )
        allocated_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
        allocator.full_to_hisparse_device_index_mapping[allocated_indices] = (
            torch.tensor([4, 5, 6])
        )
        original_mapping = allocator.full_to_hisparse_device_index_mapping.clone()
        allocator.hisparse_attn_allocator = SimpleNamespace(
            alloc=lambda need_size: None
        )
        allocator.free_hisparse_indices = MagicMock()

        self.assertIsNone(allocator.alloc_device_buffer(allocated_indices, 8))
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping, original_mapping
            )
        )
        allocator.free_hisparse_indices.assert_not_called()

    def test_coordinator_estimates_mtp_extra_page_capacity(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mem_pool_device = SimpleNamespace(page_size=64)
        coordinator.device_buffer_size = 4096
        coordinator.padded_buffer_size = 4160
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_allocated_len=8192,
            sampling_params=SimpleNamespace(max_new_tokens=8192),
        )

        self.assertEqual(
            coordinator.estimate_device_buffer_alloc_size(req, True),
            (4160, 4096),
        )

        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(9000, dtype=torch.int64).view(1, -1)
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_to_hisparse_device_index_mapping=torch.zeros(9000, dtype=torch.int64),
            hisparse_attn_allocator=SimpleNamespace(available_size=lambda: 4159),
        )
        self.assertFalse(coordinator.can_admit_request_direct(req, True))

        coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator = (
            SimpleNamespace(available_size=lambda: 4160)
        )
        self.assertTrue(coordinator.can_admit_request_direct(req, True))

    def test_hisparse_pending_admission_preserves_fifo(self):
        class DummyScheduler(SchedulerDisaggregationDecodeMixin):
            pass

        scheduler = DummyScheduler()
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        scheduler.tree_cache = MagicMock()
        scheduler.stream_output = MagicMock()
        attempts = []
        blocked = {"a"}

        def try_admit(req, require_spec_extra_page=False):
            attempts.append(req.rid)
            return req.rid not in blocked

        scheduler.hisparse_coordinator = SimpleNamespace(
            try_admit_request_direct=try_admit,
            request_finished=MagicMock(),
        )
        req_a = SimpleNamespace(rid="a", finished_reason=None, return_logprob=False)
        req_b = SimpleNamespace(rid="b", finished_reason=None, return_logprob=False)

        self.assertEqual(scheduler._admit_hisparse_transferred_reqs([req_a, req_b]), [])
        self.assertEqual(attempts, ["a"])
        self.assertEqual(
            [req.rid for req in scheduler.hisparse_admit_pending_queue],
            ["a", "b"],
        )

        blocked.clear()
        self.assertEqual(
            [req.rid for req in scheduler._drain_hisparse_admit_pending_queue()],
            ["a", "b"],
        )
        self.assertEqual(list(scheduler.hisparse_admit_pending_queue), [])

    def test_decode_queue_limit_rejects_with_503(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        existing_req = SimpleNamespace(rid="existing")
        rejected_req = SimpleNamespace(
            rid="new", return_logprob=False, finished_reason=None
        )
        queue.queue = []
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.transfer_queue = SimpleNamespace(queue=[])
        queue.scheduler = SimpleNamespace(
            max_queued_requests=1,
            waiting_queue=[existing_req],
            running_batch=SimpleNamespace(reqs=[]),
            stream_output=MagicMock(),
        )

        self.assertTrue(queue._abort_if_decode_queue_full(rejected_req))
        self.assertEqual(rejected_req.finished_reason.status_code, 503)
        queue.scheduler.stream_output.assert_called_once_with([rejected_req], False)


if __name__ == "__main__":
    unittest.main()
