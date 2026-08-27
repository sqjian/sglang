import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDDecodeDCPTopology(unittest.TestCase):
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
