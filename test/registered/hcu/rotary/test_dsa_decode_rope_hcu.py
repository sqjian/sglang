import unittest

import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.utils import is_hcu
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=10, suite="stage-b-test-1-hcu-small")


@unittest.skipUnless(is_hcu(), "Requires HCU")
class TestDSADecodeRotaryEmbeddingHCU(unittest.TestCase):
    def test_tp8_decode_shape_matches_native(self) -> None:
        rope = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=4096,
            base=5000000,
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).cuda()
        positions = torch.tensor([17], dtype=torch.int64, device="cuda")
        query = torch.randn(1, 8 * 64, dtype=torch.bfloat16, device="cuda")
        key = torch.randn(1, 64, dtype=torch.bfloat16, device="cuda")

        expected_query, expected_key = rope.forward_native(
            positions, query.clone(), key.clone()
        )
        actual_query, actual_key = rope.forward_cuda(
            positions, query.clone(), key.clone()
        )

        torch.testing.assert_close(actual_query, expected_query, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual_key, expected_key, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
