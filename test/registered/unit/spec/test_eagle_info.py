import unittest

import torch

from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_draft_extend_input():
    return EagleDraftExtendInput(
        hidden_states=torch.arange(12, dtype=torch.float32).reshape(6, 2),
        num_correct_drafts=torch.tensor([1, 0, 2], dtype=torch.int32),
        num_accept_tokens=torch.tensor([2, 1, 3], dtype=torch.int32),
        num_accept_tokens_cpu=[2, 1, 3],
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long),
        seq_lens=torch.tensor([22, 31, 43], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([22, 31, 43], dtype=torch.int32),
        req_pool_indices=torch.tensor([5, 6, 7], dtype=torch.int64),
        positions=torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.long),
        bonus_tokens=torch.tensor([210, 211, 212], dtype=torch.int32),
    )


class TestEagleDraftExtendInput(unittest.TestCase):
    def test_slice_single_preserves_flat_accepted_token_fields(self):
        spec_info = _make_draft_extend_input()

        sliced = spec_info.slice_single(2)

        self.assertIsInstance(sliced, EagleDraftExtendInput)
        self.assertEqual(sliced.num_accept_tokens_cpu, [3])
        self.assertEqual(sliced.num_correct_drafts.tolist(), [2])
        self.assertEqual(sliced.num_accept_tokens.tolist(), [3])
        self.assertEqual(sliced.input_ids.tolist(), [13, 14, 15])
        self.assertEqual(sliced.positions.tolist(), [103, 104, 105])
        self.assertEqual(
            sliced.hidden_states.tolist(), [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]
        )
        self.assertEqual(sliced.seq_lens.tolist(), [43])
        self.assertEqual(sliced.seq_lens_cpu.tolist(), [43])
        self.assertEqual(sliced.req_pool_indices.tolist(), [7])
        self.assertEqual(sliced.bonus_tokens.tolist(), [212])

    def test_merge_batch_concatenates_request_and_flat_token_fields(self):
        left = _make_draft_extend_input().slice_single(0)
        right = _make_draft_extend_input().slice_single(2)

        left.merge_batch(right)

        self.assertEqual(left.num_accept_tokens_cpu, [2, 3])
        self.assertEqual(left.num_correct_drafts.tolist(), [1, 2])
        self.assertEqual(left.num_accept_tokens.tolist(), [2, 3])
        self.assertEqual(left.input_ids.tolist(), [10, 11, 13, 14, 15])
        self.assertEqual(left.positions.tolist(), [100, 101, 103, 104, 105])
        self.assertEqual(
            left.hidden_states.tolist(),
            [[0.0, 1.0], [2.0, 3.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        )
        self.assertEqual(left.seq_lens.tolist(), [22, 43])
        self.assertEqual(left.seq_lens_cpu.tolist(), [22, 43])
        self.assertEqual(left.req_pool_indices.tolist(), [5, 7])
        self.assertEqual(left.bonus_tokens.tolist(), [210, 212])

    def test_filter_batch_selects_requested_requests_and_token_spans(self):
        spec_info = _make_draft_extend_input()

        spec_info.filter_batch(
            torch.tensor([2, 0], dtype=torch.int64), has_been_filtered=False
        )

        self.assertEqual(spec_info.num_accept_tokens_cpu, [3, 2])
        self.assertEqual(spec_info.num_correct_drafts.tolist(), [2, 1])
        self.assertEqual(spec_info.num_accept_tokens.tolist(), [3, 2])
        self.assertEqual(spec_info.input_ids.tolist(), [13, 14, 15, 10, 11])
        self.assertEqual(spec_info.positions.tolist(), [103, 104, 105, 100, 101])
        self.assertEqual(spec_info.seq_lens.tolist(), [43, 22])
        self.assertEqual(spec_info.seq_lens_cpu.tolist(), [43, 22])
        self.assertEqual(spec_info.req_pool_indices.tolist(), [7, 5])
        self.assertEqual(spec_info.bonus_tokens.tolist(), [212, 210])

    def test_filter_batch_uses_prefix_when_spec_info_was_already_filtered(self):
        spec_info = _make_draft_extend_input()

        spec_info.filter_batch(
            torch.tensor([9, 8], dtype=torch.int64), has_been_filtered=True
        )

        self.assertEqual(spec_info.num_accept_tokens_cpu, [2, 1])
        self.assertEqual(spec_info.num_correct_drafts.tolist(), [1, 0])
        self.assertEqual(spec_info.input_ids.tolist(), [10, 11, 12])
        self.assertEqual(spec_info.positions.tolist(), [100, 101, 102])
        self.assertEqual(spec_info.seq_lens.tolist(), [22, 31])
        self.assertEqual(spec_info.req_pool_indices.tolist(), [5, 6])


if __name__ == "__main__":
    unittest.main()
