"""CPU coverage for the Rust PD bootstrap lifecycle handoff."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.disagg_service import start_rust_disagg_service
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestRustPdBootstrap(CustomTestCase):
    def test_non_prefill_role_is_noop_without_loading_extension(self):
        args = SimpleNamespace(disaggregation_mode="decode")
        with patch.dict(sys.modules, {"sglang.srt.server._core": None}):
            self.assertIsNone(start_rust_disagg_service(args))

    def test_prefill_role_constructs_native_server(self):
        created = []

        class FakeBootstrapServer:
            def __init__(self, host, port):
                created.append((host, port))

        extension = types.ModuleType("sglang.srt.server._core")
        extension.BootstrapServer = FakeBootstrapServer
        args = SimpleNamespace(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            disaggregation_bootstrap_port=8998,
            host="127.0.0.1",
            node_rank=0,
        )

        with patch.dict(sys.modules, {"sglang.srt.server._core": extension}):
            handle = start_rust_disagg_service(args)

        self.assertIsInstance(handle, FakeBootstrapServer)
        self.assertEqual(created, [("127.0.0.1", 8998)])

    def test_scheduler_retains_handle_only_on_rust_host_rank(self):
        handle = object()

        class FakeScheduler:
            server_args = object()

            def __init__(self, hosts_rust_server):
                self.hosts_rust_server = hosts_rust_server

            def _hosts_rust_server(self):
                return self.hosts_rust_server

        host = FakeScheduler(True)
        with patch(
            "sglang.srt.managers.scheduler.start_rust_disagg_service",
            return_value=handle,
        ) as start:
            Scheduler.maybe_init_disagg_bootstrap_server(host)
        start.assert_called_once_with(host.server_args)
        self.assertIs(host.disagg_bootstrap_server, handle)

        non_host = FakeScheduler(False)
        with patch("sglang.srt.managers.scheduler.start_rust_disagg_service") as start:
            Scheduler.maybe_init_disagg_bootstrap_server(non_host)
        start.assert_not_called()
        self.assertIsNone(non_host.disagg_bootstrap_server)


if __name__ == "__main__":
    unittest.main()
