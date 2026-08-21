import unittest
from unittest.mock import patch

from sglang.srt.distributed import parallel_state
from sglang.srt.environ import envs


class PyNcclDisableTest(unittest.TestCase):
    def test_environment_switch_overrides_default_and_explicit_enable(self):
        for requested in (None, True):
            with self.subTest(requested=requested):
                with (
                    envs.SGLANG_DISABLE_PYNCCL.override(True),
                    patch.object(parallel_state, "GroupCoordinator") as coordinator,
                ):
                    parallel_state.init_model_parallel_group(
                        [[0, 1]],
                        local_rank=0,
                        backend="nccl",
                        use_pynccl=requested,
                    )
                self.assertFalse(coordinator.call_args.kwargs["use_pynccl"])

    def test_pynccl_remains_enabled_without_environment_switch(self):
        with (
            envs.SGLANG_DISABLE_PYNCCL.override(False),
            patch.object(parallel_state, "GroupCoordinator") as coordinator,
        ):
            parallel_state.init_model_parallel_group(
                [[0, 1]],
                local_rank=0,
                backend="nccl",
                use_pynccl=True,
            )
        self.assertTrue(coordinator.call_args.kwargs["use_pynccl"])


if __name__ == "__main__":
    unittest.main()
