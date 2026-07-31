"""Real two-GPU Mooncake PD coverage with Rust ingress on both roles."""

import json
import os
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    is_rust_server_built,
)

register_cuda_ci(est_time=500, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationRustServer(PDDisaggregationServerBase):
    """This class is intentionally not skip-guarded.

    It is the authoritative gate: a missing Rust extension, GPU, or Mooncake
    runtime is a failed prerequisite rather than a silently reduced test.
    """

    capture_per_side_logs = True
    extra_prefill_env = {"SGLANG_RUST_SERVER": "1"}
    extra_decode_env = {"SGLANG_RUST_SERVER": "1"}

    @classmethod
    def setUpClass(cls):
        if not is_rust_server_built():
            raise RuntimeError(
                "authoritative Rust PD E2E requires the built "
                "sglang.srt.server._core extension"
            )

        cls._old_backend = os.environ.get("SGLANG_TEST_PD_DISAGG_BACKEND")
        os.environ["SGLANG_TEST_PD_DISAGG_BACKEND"] = "mooncake"
        base_initialized = False
        try:
            super().setUpClass()
            base_initialized = True
            cls.model = os.environ.get(
                "SGLANG_TEST_MODEL",
                DEFAULT_MODEL_NAME_FOR_TEST,
            )
            if cls.transfer_backend[-1] != "mooncake":
                raise RuntimeError(
                    f"authoritative Rust PD E2E requires Mooncake, got "
                    f"{cls.transfer_backend[-1]}"
                )
            cls.launch_all()
        except BaseException:
            if base_initialized:
                super().tearDownClass()
            cls._restore_backend_env()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            cls._restore_backend_env()

    @classmethod
    def _restore_backend_env(cls):
        if cls._old_backend is None:
            os.environ.pop("SGLANG_TEST_PD_DISAGG_BACKEND", None)
        else:
            os.environ["SGLANG_TEST_PD_DISAGG_BACKEND"] = cls._old_backend

    def test_scalar_generate_via_router(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["text"])
        self.assertIsNotNone(result["meta_info"]["finish_reason"])

    def test_batch_generate_preserves_count_and_order(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "rid": ["first", "second"],
                "text": [
                    "The capital of France is",
                    "The capital of Japan is",
                ],
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )

        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        self.assertEqual(len(results), 2)
        self.assertEqual(
            [result["meta_info"]["id"] for result in results],
            ["first", "second"],
        )
        self.assertTrue(all(result["text"] for result in results))

    def test_stream_generate_finishes_once(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Count from one to three:",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                "stream": True,
            },
            stream=True,
            timeout=120,
        )

        self.assertEqual(response.status_code, 200, response.text)
        chunks = []
        done_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]":
                done_count += 1
                continue
            chunks.append(json.loads(payload))

        self.assertTrue(chunks)
        self.assertTrue(any(chunk["text"] for chunk in chunks))
        self.assertEqual(done_count, 1)
        terminal = [
            chunk for chunk in chunks if chunk["meta_info"]["finish_reason"] is not None
        ]
        self.assertEqual(len(terminal), 1)

    def test_backend_health_after_pd_warmup(self):
        for name, process, url in (
            ("prefill", self.process_prefill, self.prefill_url),
            ("decode", self.process_decode, self.decode_url),
        ):
            assert_process_healthy(self, name, process, url, "/health_generate")

    def test_missing_bootstrap_returns_400_without_crashing_scheduler(self):
        stdout_offset = self._decode_stdout_buf.tell()
        stderr_offset = self._decode_stderr_buf.tell()

        response = requests.post(
            self.decode_url + "/generate",
            json={
                "text": "This request intentionally omits bootstrap metadata.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
            timeout=30,
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("without bootstrap room id", response.text)
        assert_process_healthy(
            self,
            "decode",
            self.process_decode,
            self.decode_url,
            "/health_generate",
        )
        new_logs = (
            self._decode_stdout_buf.getvalue()[stdout_offset:]
            + self._decode_stderr_buf.getvalue()[stderr_offset:]
        )
        self.assertNotIn("Traceback (most recent call last)", new_logs)


if __name__ == "__main__":
    unittest.main()
