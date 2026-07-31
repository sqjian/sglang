"""Positional ABI checks for Rust scheduler ingress headers."""

import unittest

import msgspec

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.utils import msgpack_decode_explained
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

RUST_HEADER_FIELDS = (
    "rid",
    "http_worker_ipc",
    "input_text",
    "input_ids",
    "input_embeds",
    "mm_inputs",
    "token_type_ids",
    "sampling_params",
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
    "token_ids_logprob",
    "stream",
    "return_sampling_mask",
    "return_hidden_states",
    "return_routed_experts",
    "routed_experts_start_len",
    "return_indexer_topk",
    "session_id",
    "session_params",
    "lora_id",
    "custom_logit_processor",
    "positional_embed_overrides",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "bootstrap_pair_key",
    "decode_tp_size",
    "routed_dp_rank",
    "disagg_prefill_dp_rank",
)


class TestRustHeaderAbi(CustomTestCase):
    def test_python_struct_prefix_matches_rust_encoder(self):
        self.assertEqual(
            TokenizedGenerateReqInput.__struct_fields__[: len(RUST_HEADER_FIELDS)],
            RUST_HEADER_FIELDS,
        )

    def test_rust_shaped_header_decodes_pd_fields(self):
        header = msgspec.msgpack.encode(
            [
                "TokenizedGenerateReqInput",
                "rid-1",
                None,
                "hello",
                None,
                None,
                None,
                None,
                SamplingParams(temperature=0.0, max_new_tokens=8),
                False,
                -1,
                0,
                None,
                True,
                False,
                True,
                False,
                0,
                False,
                None,
                None,
                None,
                None,
                None,
                "10.0.0.1",
                8998,
                2**63 - 1,
                "pair",
                2,
                1,
                0,
            ]
        )

        request = msgpack_decode_explained(header)

        self.assertIsInstance(request, TokenizedGenerateReqInput)
        self.assertEqual(request.rid, "rid-1")
        self.assertTrue(request.stream)
        self.assertFalse(request.return_sampling_mask)
        self.assertTrue(request.return_hidden_states)
        self.assertEqual(request.bootstrap_host, "10.0.0.1")
        self.assertEqual(request.bootstrap_port, 8998)
        self.assertEqual(request.bootstrap_room, 2**63 - 1)
        self.assertEqual(request.bootstrap_pair_key, "pair")
        self.assertEqual(request.decode_tp_size, 2)
        self.assertEqual(request.routed_dp_rank, 1)
        self.assertEqual(request.disagg_prefill_dp_rank, 0)

    def test_short_header_uses_python_defaults(self):
        header = msgspec.msgpack.encode(
            [
                "TokenizedGenerateReqInput",
                "rid-2",
                None,
                None,
                None,
                None,
                None,
                None,
                SamplingParams(),
                False,
                -1,
                0,
                None,
                False,
            ]
        )

        request = msgpack_decode_explained(header)

        self.assertFalse(request.return_sampling_mask)
        self.assertIsNone(request.bootstrap_host)
        self.assertIsNone(request.bootstrap_room)
        self.assertIsNone(request.disagg_prefill_dp_rank)


if __name__ == "__main__":
    unittest.main()
