import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_output_processor_mixin import (  # noqa: E402
    SchedulerOutputProcessorMixin,
)

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestSchedulerOutputProcessor(unittest.TestCase):
    def test_prefill_converts_batched_logprob_tensors(self):
        logits_output = SimpleNamespace(
            next_token_logprobs=None,
            input_token_logprobs=None,
            input_top_logprobs_val=torch.tensor([[[-1.0, -2.0]]]),
            input_top_logprobs_idx=torch.tensor([[[7, 8]]]),
            input_token_ids_logprobs_val=torch.tensor([[[-3.0, -4.0]]]),
            next_token_top_logprobs_val=torch.tensor([[1.0, 2.0]]),
            next_token_top_logprobs_idx=torch.tensor([[3, 4]]),
            next_token_token_ids_logprobs_val=torch.tensor([[5.0, 6.0]]),
        )
        result = SimpleNamespace(
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            next_draft_input=None,
            logits_output=logits_output,
            next_token_ids=torch.empty(0, dtype=torch.int64),
            extend_input_len_per_req=[],
            extend_logprob_start_len_per_req=[],
        )
        batch = SimpleNamespace(
            return_logprob=True,
            reqs=[],
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        scheduler = SimpleNamespace(
            is_generation=True,
            enable_hisparse=False,
            stream_output=MagicMock(),
            report_prefill_stats=MagicMock(),
        )

        SchedulerOutputProcessorMixin.process_batch_result_prefill(scheduler, batch, result)

        self.assertEqual(logits_output.next_token_top_logprobs_val, [[1.0, 2.0]])
        self.assertEqual(logits_output.next_token_top_logprobs_idx, [[3, 4]])
        self.assertEqual(logits_output.input_top_logprobs_val, [[[-1.0, -2.0]]])
        self.assertEqual(logits_output.input_top_logprobs_idx, [[[7, 8]]])
        self.assertEqual(logits_output.input_token_ids_logprobs_val, [[[-3.0, -4.0]]])
        self.assertEqual(logits_output.next_token_token_ids_logprobs_val, [[5.0, 6.0]])


class _FakeMambaPool:
    def __init__(self):
        self.freed_indices = []

    def free(self, indices):
        self.freed_indices.append(indices.clone())


class TestSchedulerOutputProcessorFinishedReq(unittest.TestCase):
    def _new_scheduler(self, tree_cache):
        scheduler = SimpleNamespace()
        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        scheduler.tree_cache = tree_cache
        scheduler.enable_hisparse = False
        scheduler.maybe_collect_routed_experts = MagicMock()
        scheduler.maybe_collect_indexer_topk = MagicMock()
        scheduler.maybe_collect_customized_info = MagicMock()
        return scheduler

    def _new_finished_req(self, *, req_pool_idx, mamba_pool_idx=None):
        req = SimpleNamespace()
        req.rid = "finished-req"
        req.req_pool_idx = req_pool_idx
        req.mamba_pool_idx = mamba_pool_idx
        req.multimodal_inputs = None
        req.session = None
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False
        req.time_stats = SimpleNamespace(set_completion_time=MagicMock())
        req.finished = MagicMock(return_value=True)
        return req

    def test_finished_req_without_req_pool_idx_releases_mamba_slot(self):
        mamba_pool = _FakeMambaPool()
        tree_cache = SimpleNamespace(
            supports_mamba=MagicMock(return_value=True),
            req_to_token_pool=SimpleNamespace(mamba_pool=mamba_pool),
        )
        scheduler = self._new_scheduler(tree_cache)
        req = self._new_finished_req(
            req_pool_idx=None, mamba_pool_idx=torch.tensor(7, dtype=torch.int64)
        )

        SchedulerOutputProcessorMixin._handle_finished_req(
            scheduler, req, 0, logits_output=None
        )

        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(len(mamba_pool.freed_indices), 1)
        torch.testing.assert_close(
            mamba_pool.freed_indices[0], torch.tensor([7], dtype=torch.int64)
        )
        tree_cache.supports_mamba.assert_called_once()
        req.time_stats.set_completion_time.assert_called_once()

    def test_finished_req_without_any_pool_idx_keeps_duplicate_kv_release_guard(self):
        tree_cache = SimpleNamespace(supports_mamba=MagicMock(return_value=True))
        scheduler = self._new_scheduler(tree_cache)
        req = self._new_finished_req(req_pool_idx=None, mamba_pool_idx=None)

        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release_kv_cache:
            SchedulerOutputProcessorMixin._handle_finished_req(
                scheduler, req, 0, logits_output=None
            )

        release_kv_cache.assert_not_called()
        tree_cache.supports_mamba.assert_not_called()
        req.time_stats.set_completion_time.assert_called_once()

    def test_finished_req_with_req_pool_idx_uses_existing_release_path(self):
        tree_cache = MagicMock()
        scheduler = self._new_scheduler(tree_cache)
        scheduler.enable_hisparse = True
        scheduler.hisparse_coordinator = SimpleNamespace(request_finished=MagicMock())
        req = self._new_finished_req(req_pool_idx=3)

        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release_kv_cache:
            SchedulerOutputProcessorMixin._handle_finished_req(
                scheduler, req, 0, logits_output=None
            )

        scheduler.hisparse_coordinator.request_finished.assert_called_once_with(req)
        release_kv_cache.assert_called_once_with(req, tree_cache)
        req.time_stats.set_completion_time.assert_called_once()


if __name__ == "__main__":
    unittest.main()
