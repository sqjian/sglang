import unittest
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_output_processor_mixin import (  # noqa: E402
    DisaggregationMode,
    SchedulerOutputProcessorMixin,
)

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSchedulerOutputProcessor(unittest.TestCase):
    def _new_scheduler(self):
        scheduler = SchedulerOutputProcessorMixin()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.enable_hisparse = True
        scheduler.hisparse_coordinator = MagicMock()
        scheduler.server_args = MagicMock()
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.tree_cache = MagicMock()
        scheduler.stream_output = MagicMock()
        return scheduler

    def _new_finished_req(self):
        req = MagicMock()
        req.finished.return_value = True
        req.time_stats = MagicMock()
        return req

    def test_prebuilt_finish_releases_hisparse_before_kv_cache(self):
        scheduler = self._new_scheduler()
        req = self._new_finished_req()
        batch = MagicMock()
        batch.reqs = [req]
        batch.return_logprob = False

        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release_kv_cache:
            calls = MagicMock()
            calls.attach_mock(
                scheduler.hisparse_coordinator.request_finished, "request_finished"
            )
            calls.attach_mock(release_kv_cache, "release_kv_cache")

            scheduler.process_batch_result_prebuilt(batch)

        req.check_finished.assert_called_once()
        req.time_stats.set_quick_finish_time.assert_called_once()
        calls.assert_has_calls(
            [
                call.request_finished(req),
                call.release_kv_cache(req, scheduler.tree_cache),
            ]
        )
        scheduler.stream_output.assert_called_once_with(batch.reqs, False)


if __name__ == "__main__":
    unittest.main()
