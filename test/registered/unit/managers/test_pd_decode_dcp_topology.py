import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDDecodeDCPTopology(unittest.TestCase):
    def _new_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.server_args = SimpleNamespace(enable_dp_attention=True, dp_size=4)
        scheduler.require_mlp_sync = True
        scheduler.ps = SimpleNamespace(
            pp_size=1,
            attn_tp_size=8,
            attn_cp_size=1,
            attn_dcp_size=8,
            tp_rank=0,
        )
        scheduler.tp_group = SimpleNamespace(ranks=list(range(32)))
        return scheduler

    def test_logical_kv_capacity_scales_physical_pool_once_for_dcp8(self):
        scheduler = self._new_scheduler()
        scheduler.max_total_num_tokens = 147_584

        with get_parallel().override(attn_dcp_size=8):
            self.assertEqual(scheduler.logical_max_total_num_tokens, 1_180_672)

    def test_logical_kv_capacity_keeps_dcp1_unchanged(self):
        scheduler = self._new_scheduler()
        scheduler.max_total_num_tokens = 147_584

        with get_parallel().override(attn_dcp_size=1):
            self.assertEqual(scheduler.logical_max_total_num_tokens, 147_584)

    def test_stepinfo_sync_is_disabled_by_default(self):
        self.assertFalse(envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.default)

        scheduler = self._new_scheduler()
        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(False),
            patch("torch.distributed.get_world_size") as get_world_size,
            patch("torch.distributed.new_group") as new_group,
        ):
            scheduler._init_pd_decode_stepinfo_sync()

        self.assertFalse(scheduler._enable_pd_decode_stepinfo_sync)
        self.assertIsNone(scheduler.dp_scheduler_cpu_group)
        self.assertIsNone(scheduler.get_pd_decode_step_context())
        get_world_size.assert_not_called()
        new_group.assert_not_called()

    def test_stepinfo_sync_opt_in_allows_dcp_attention_tp(self):
        scheduler = self._new_scheduler()
        cpu_group = object()
        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(True),
            patch("torch.distributed.get_world_size", return_value=32),
            patch("torch.distributed.new_group", return_value=cpu_group) as new_group,
        ):
            scheduler._init_pd_decode_stepinfo_sync()

        self.assertTrue(scheduler._enable_pd_decode_stepinfo_sync)
        self.assertIs(scheduler.dp_scheduler_cpu_group, cpu_group)
        new_group.assert_called_once()

    def test_allows_dcp_attention_tp(self):
        Scheduler._validate_pd_decode_dp_sync_parallel_sizes(
            SimpleNamespace(pp_size=1, attn_tp_size=8, attn_cp_size=1)
        )

    def test_rejects_pipeline_parallelism(self):
        with self.assertRaisesRegex(RuntimeError, "pp_size=1"):
            Scheduler._validate_pd_decode_dp_sync_parallel_sizes(
                SimpleNamespace(pp_size=2, attn_tp_size=8, attn_cp_size=1)
            )

    def test_rejects_context_parallelism(self):
        with self.assertRaisesRegex(RuntimeError, "attn_cp_size=1"):
            Scheduler._validate_pd_decode_dp_sync_parallel_sizes(
                SimpleNamespace(pp_size=1, attn_tp_size=8, attn_cp_size=2)
            )


if __name__ == "__main__":
    unittest.main()
