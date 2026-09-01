import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodeAdmissionCapacity(unittest.TestCase):
    def _new_queue(
        self,
        *,
        physical_capacity: int,
        logical_capacity: int,
        hisparse_capacity: int | None = None,
    ) -> DecodePreallocQueue:
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.max_total_num_tokens = physical_capacity
        queue.logical_max_total_num_tokens = logical_capacity
        queue.scheduler = SimpleNamespace(
            enable_hisparse=hisparse_capacity is not None,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    max_token_pool_size=hisparse_capacity,
                )
            ),
            output_streamer=MagicMock(),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        return queue

    @staticmethod
    def _new_req(
        input_len: int,
        *,
        output_len: int = 0,
        rebootstrap: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            rid="capacity-test",
            origin_input_ids=[1] * input_len,
            output_ids=[2] * output_len,
            pd_rebootstrap_in_progress=rebootstrap,
            return_logprob=False,
            finished_reason=None,
        )

    def test_dcp1_keeps_physical_capacity_boundary(self):
        queue = self._new_queue(
            physical_capacity=16,
            logical_capacity=16,
        )

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(self._new_req(16)))
        rejected = self._new_req(17)
        self.assertTrue(queue._check_if_req_exceed_kv_capacity(rejected))
        self.assertIsInstance(rejected.finished_reason, FINISH_ABORT)

    def test_dcp8_admits_above_physical_through_logical_boundary(self):
        queue = self._new_queue(
            physical_capacity=16,
            logical_capacity=128,
        )

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(self._new_req(17)))
        self.assertFalse(queue._check_if_req_exceed_kv_capacity(self._new_req(128)))

    def test_dcp8_rejects_above_logical_capacity(self):
        queue = self._new_queue(
            physical_capacity=16,
            logical_capacity=128,
        )
        rejected = self._new_req(129)

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(rejected))
        self.assertIn("129 > 128", rejected.finished_reason.message)
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [rejected], False
        )

    def test_rebootstrap_uses_logical_capacity_for_full_recomputed_prefix(self):
        queue = self._new_queue(
            physical_capacity=16,
            logical_capacity=128,
        )

        self.assertFalse(
            queue._check_if_req_exceed_kv_capacity(
                self._new_req(100, output_len=28, rebootstrap=True)
            )
        )
        rejected = self._new_req(100, output_len=29, rebootstrap=True)
        self.assertTrue(queue._check_if_req_exceed_kv_capacity(rejected))
        self.assertIn("129 > 128", rejected.finished_reason.message)

    def test_hisparse_keeps_host_backed_capacity(self):
        queue = self._new_queue(
            physical_capacity=16,
            logical_capacity=128,
            hisparse_capacity=256,
        )

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(self._new_req(256)))
        self.assertTrue(queue._check_if_req_exceed_kv_capacity(self._new_req(257)))

    def test_hybrid_swa_fallback_scales_the_existing_physical_limit(self):
        allocator = MagicMock()
        allocator.get_kvcache.return_value = object()
        transfer_queue = MagicMock()
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_size=1),
            tp_worker=SimpleNamespace(
                is_hybrid_swa=True,
                model_runner=SimpleNamespace(swa_max_total_num_tokens=12),
            ),
            dcp_logical_token_capacity=MagicMock(side_effect=lambda value: value * 8),
        )

        with (
            envs.SGLANG_DISAGG_STAGING_BUFFER.override(False),
            patch(
                "sglang.srt.disaggregation.decode.is_mla_backend",
                return_value=False,
            ),
            patch.object(
                DecodePreallocQueue,
                "_init_kv_manager",
                return_value=MagicMock(),
            ),
        ):
            queue = DecodePreallocQueue(
                req_to_token_pool=MagicMock(),
                token_to_kv_pool_allocator=allocator,
                draft_token_to_kv_pool=None,
                req_to_metadata_buffer_idx_allocator=MagicMock(),
                metadata_buffers=MagicMock(),
                scheduler=scheduler,
                transfer_queue=transfer_queue,
                tree_cache=MagicMock(),
                gloo_group=MagicMock(),
                tp_rank=0,
                tp_size=8,
                dp_size=4,
                gpu_id=0,
                bootstrap_port=8999,
                max_total_num_tokens=16,
                logical_max_total_num_tokens=128,
                pp_rank=0,
                num_reserved_decode_tokens=0,
                transfer_backend=MagicMock(),
            )

        self.assertEqual(queue.max_total_num_tokens, 12)
        self.assertEqual(queue.logical_max_total_num_tokens, 96)
        scheduler.dcp_logical_token_capacity.assert_called_once_with(12)

    def test_swa_tail_capacity_remains_a_separate_physical_gate(self):
        queue = self._new_queue(
            physical_capacity=16,
            logical_capacity=128,
        )
        queue._uses_swa_tail_prealloc.return_value = True
        queue._prealloc_required_tokens = MagicMock(return_value=(64, 65))
        queue.token_to_kv_pool_allocator = SimpleNamespace(size_swa=64)
        rejected = self._new_req(100)

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(rejected))
        self.assertIn("65 > 64", rejected.finished_reason.message)


if __name__ == "__main__":
    unittest.main()
