import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from sglang.srt.distributed import parallel_state


class PipelineP2PPrewarmTest(unittest.TestCase):
    def test_model_runner_prewarms_after_group_init_and_before_memory_measurement(self):
        model_runner = (
            Path(parallel_state.__file__).parents[1]
            / "model_executor"
            / "model_runner.py"
        )
        source = model_runner.read_text(encoding="utf-8")
        method_start = source.index("    def init_torch_distributed(self):")
        method_end = source.index(
            "    def init_shared_mooncake_transfer_engine", method_start
        )
        method = source[method_start:method_end]

        self.assertLess(
            method.index("initialize_model_parallel("),
            method.index("prewarm_pp_p2p()"),
        )
        self.assertLess(
            method.index("prewarm_pp_p2p()"),
            method.index("pre_model_load_memory = get_available_gpu_memory("),
        )

    def test_middle_stage_receives_then_sends_to_adjacent_stages(self):
        pp_group = Mock(
            device="cuda:0",
            device_group=object(),
            is_first_rank=False,
            is_last_rank=False,
            rank_in_group=1,
            ranks=[0, 8, 16, 24],
        )
        dummy = object()
        operations = Mock()

        with (
            patch.object(parallel_state, "get_pp_group", return_value=pp_group),
            patch.object(parallel_state.torch, "zeros", return_value=dummy),
            patch.object(
                parallel_state.torch.distributed,
                "recv",
                side_effect=lambda *args, **kwargs: operations.recv(*args, **kwargs),
            ),
            patch.object(
                parallel_state.torch.distributed,
                "send",
                side_effect=lambda *args, **kwargs: operations.send(*args, **kwargs),
            ),
            patch.object(parallel_state.torch, "get_device_module") as device_module,
        ):
            parallel_state.prewarm_pp_p2p()

        self.assertEqual(
            operations.mock_calls,
            [
                call.recv(dummy, src=0, group=pp_group.device_group),
                call.send(dummy, dst=16, group=pp_group.device_group),
            ],
        )
        device_module.return_value.synchronize.assert_called_once_with()

    def test_edge_stages_only_contact_their_existing_neighbor(self):
        for rank_in_group, first, last, expected_operation in (
            (0, True, False, "send"),
            (3, False, True, "recv"),
        ):
            with self.subTest(rank_in_group=rank_in_group):
                pp_group = Mock(
                    device="cuda:0",
                    device_group=object(),
                    is_first_rank=first,
                    is_last_rank=last,
                    rank_in_group=rank_in_group,
                    ranks=[0, 8, 16, 24],
                )
                with (
                    patch.object(parallel_state, "get_pp_group", return_value=pp_group),
                    patch.object(parallel_state.torch, "zeros", return_value=object()),
                    patch.object(parallel_state.torch.distributed, "recv") as recv,
                    patch.object(parallel_state.torch.distributed, "send") as send,
                    patch.object(parallel_state.torch, "get_device_module"),
                ):
                    parallel_state.prewarm_pp_p2p()

                self.assertEqual(recv.call_count, expected_operation == "recv")
                self.assertEqual(send.call_count, expected_operation == "send")


if __name__ == "__main__":
    unittest.main()
