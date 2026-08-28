import ast
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation import dcp_cache_hash_diagnostic as diagnostic
from sglang.srt.disaggregation.dcp_cache_hash_diagnostic import (
    build_owned_slot_plan,
    get_prefill_layer_hash_config,
    hash_int_sequence,
    hash_positioned_rows,
    log_dsa_cache_hash_snapshot,
    log_prefill_layer_hash_snapshot,
    should_log_cache_hash,
    xor_hex_digests,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SRT_ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def _function_calls(relative_path: str, function_name: str) -> list[ast.Call]:
    tree = ast.parse((SRT_ROOT / relative_path).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"log_dsa_cache_hash_snapshot", "log_dcp_transfer_plan"}
    ]


def _class_method_calls(
    relative_path: str,
    class_name: str,
    method_name: str,
    called_name: str,
) -> list[ast.Call]:
    tree = ast.parse((SRT_ROOT / relative_path).read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == called_name
    ]


def _class_method_attribute_calls(
    relative_path: str,
    class_name: str,
    method_name: str,
    called_name: str,
) -> list[ast.Call]:
    tree = ast.parse((SRT_ROOT / relative_path).read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == called_name
    ]


class TestDcpCacheHashDiagnostic(unittest.TestCase):
    def test_dcp1_slot_plan_keeps_every_position(self):
        global_slots = torch.tensor([64, 65, 66, 67], dtype=torch.int64)

        positions, local_slots = build_owned_slot_plan(
            global_slots, dcp_size=1, dcp_rank=0
        )

        torch.testing.assert_close(positions, torch.arange(4))
        torch.testing.assert_close(local_slots, global_slots)

    def test_dcp8_slot_plan_uses_global_owner_and_local_slot(self):
        global_slots = torch.arange(512, 524, dtype=torch.int64)

        positions, local_slots = build_owned_slot_plan(
            global_slots, dcp_size=8, dcp_rank=3
        )

        torch.testing.assert_close(positions, torch.tensor([3, 11]))
        torch.testing.assert_close(local_slots, torch.tensor([64, 65]))

    def test_partitioned_row_hashes_combine_to_full_hash(self):
        rows = torch.arange(24, dtype=torch.int32).reshape(6, 4)
        positions = torch.arange(6, dtype=torch.int64)
        full = hash_positioned_rows(rows, positions, layer_id=7)
        partitioned = [
            hash_positioned_rows(rows[rank::2], positions[rank::2], layer_id=7)
            for rank in range(2)
        ]

        self.assertEqual(xor_hex_digests(partitioned), full)

    def test_integer_sequence_hash_depends_on_order(self):
        self.assertNotEqual(hash_int_sequence([1, 2]), hash_int_sequence([2, 1]))

    def test_diagnostic_is_default_off_and_exact_length_gated(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(should_log_cache_hash(727))

        with patch.dict(
            os.environ,
            {
                "SGLANG_DEBUG_DCP_CACHE_HASH": "1",
                "SGLANG_DEBUG_DCP_CACHE_HASH_SEQ_LEN": "727",
            },
            clear=True,
        ):
            self.assertTrue(should_log_cache_hash(727))
            self.assertFalse(should_log_cache_hash(728))

    def test_invalid_length_gate_fails_fast(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_DEBUG_DCP_CACHE_HASH": "1",
                "SGLANG_DEBUG_DCP_CACHE_HASH_SEQ_LEN": "not-an-int",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                should_log_cache_hash(727)

    def test_prefill_layer_hash_gate_is_default_off_and_exact(self):
        parallel = SimpleNamespace(tp_rank=0)
        environment = {
            "SGLANG_DEBUG_PREFILL_LAYER_HASH": "1",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_SEQ_LEN": "727",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MIN_LAYER": "39",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MAX_LAYER": "51",
        }

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(diagnostic, "get_parallel", return_value=parallel),
        ):
            self.assertIsNone(
                get_prefill_layer_hash_config(
                    batch_size=1,
                    is_extend=True,
                    extend_seq_lens=[727],
                )
            )

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(diagnostic, "get_parallel", return_value=parallel),
        ):
            config = get_prefill_layer_hash_config(
                batch_size=1,
                is_extend=True,
                extend_seq_lens=[727],
            )
            self.assertEqual(
                (config.seq_len, config.min_layer, config.max_layer), (727, 39, 51)
            )
            self.assertIsNone(
                get_prefill_layer_hash_config(
                    batch_size=2,
                    is_extend=True,
                    extend_seq_lens=[727, 727],
                )
            )
            self.assertIsNone(
                get_prefill_layer_hash_config(
                    batch_size=1,
                    is_extend=False,
                    extend_seq_lens=[727],
                )
            )
            self.assertIsNone(
                get_prefill_layer_hash_config(
                    batch_size=1,
                    is_extend=True,
                    extend_seq_lens=[728],
                )
            )

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                diagnostic,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=1),
            ),
        ):
            self.assertIsNone(
                get_prefill_layer_hash_config(
                    batch_size=1,
                    is_extend=True,
                    extend_seq_lens=[727],
                )
            )

    def test_invalid_prefill_layer_hash_range_fails_fast(self):
        environment = {
            "SGLANG_DEBUG_PREFILL_LAYER_HASH": "1",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_SEQ_LEN": "727",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MIN_LAYER": "52",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MAX_LAYER": "51",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                diagnostic,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "invalid layer range"):
                get_prefill_layer_hash_config(
                    batch_size=1,
                    is_extend=True,
                    extend_seq_lens=[727],
                )

    def test_prefill_layer_hash_sparse_layers_are_exact_and_outer_only(self):
        environment = {
            "SGLANG_DEBUG_PREFILL_LAYER_HASH": "1",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_SEQ_LEN": "727",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_LAYERS": "12,24,36,48,60,72,77",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                diagnostic,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
        ):
            config = get_prefill_layer_hash_config(
                batch_size=1,
                is_extend=True,
                extend_seq_lens=[727],
            )

        self.assertEqual(config.min_layer, 12)
        self.assertEqual(config.max_layer, 77)
        self.assertEqual(
            config.layer_ids,
            frozenset({12, 24, 36, 48, 60, 72, 77}),
        )
        self.assertTrue(config.includes(48))
        self.assertFalse(config.includes(49))
        self.assertFalse(config.log_sub_layer_boundaries)

    def test_prefill_sub_layer_hash_requires_explicit_opt_in(self):
        environment = {
            "SGLANG_DEBUG_PREFILL_LAYER_HASH": "1",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_SEQ_LEN": "727",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MIN_LAYER": "17",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MAX_LAYER": "17",
            "SGLANG_DEBUG_PREFILL_SUB_LAYER_HASH": "1",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                diagnostic,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
        ):
            config = get_prefill_layer_hash_config(
                batch_size=1,
                is_extend=True,
                extend_seq_lens=[727],
            )

        self.assertTrue(config.log_sub_layer_boundaries)

    def test_prefill_mlp_hash_requires_sublayer_opt_in(self):
        environment = {
            "SGLANG_DEBUG_PREFILL_LAYER_HASH": "1",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_SEQ_LEN": "727",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MIN_LAYER": "17",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MAX_LAYER": "17",
            "SGLANG_DEBUG_PREFILL_SUB_LAYER_HASH": "1",
        }
        parallel = SimpleNamespace(tp_rank=0)
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(diagnostic, "get_parallel", return_value=parallel),
        ):
            config = get_prefill_layer_hash_config(
                batch_size=1,
                is_extend=True,
                extend_seq_lens=[727],
            )
        self.assertFalse(config.log_mlp_boundaries)

        environment["SGLANG_DEBUG_PREFILL_MLP_HASH"] = "1"
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(diagnostic, "get_parallel", return_value=parallel),
        ):
            config = get_prefill_layer_hash_config(
                batch_size=1,
                is_extend=True,
                extend_seq_lens=[727],
            )
        self.assertTrue(config.log_mlp_boundaries)

        del environment["SGLANG_DEBUG_PREFILL_SUB_LAYER_HASH"]
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(diagnostic, "get_parallel", return_value=parallel),
        ):
            with self.assertRaisesRegex(ValueError, "requires"):
                get_prefill_layer_hash_config(
                    batch_size=1,
                    is_extend=True,
                    extend_seq_lens=[727],
                )

    def test_prefill_sparse_layers_reject_ambiguous_range(self):
        environment = {
            "SGLANG_DEBUG_PREFILL_LAYER_HASH": "1",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_SEQ_LEN": "727",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_LAYERS": "12,24",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MIN_LAYER": "12",
            "SGLANG_DEBUG_PREFILL_LAYER_HASH_MAX_LAYER": "24",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                diagnostic,
                "get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                get_prefill_layer_hash_config(
                    batch_size=1,
                    is_extend=True,
                    extend_seq_lens=[727],
                )

    def test_prefill_layer_snapshot_hashes_without_tensor_contents(self):
        parallel = SimpleNamespace(
            world_rank=16,
            tp_rank=0,
            pp_rank=2,
            attn_tp_rank=0,
            attn_dp_rank=0,
            attn_dcp_size=1,
            attn_dcp_rank=0,
        )
        positions = torch.tensor([0, 1, 0], dtype=torch.int64)
        tensors = {
            "hidden_states": torch.tensor(
                [[11, 12], [13, 14], [0, 0]], dtype=torch.int32
            ),
            "residual": None,
            "topk_indices": torch.tensor([[3], [4], [0]], dtype=torch.int32),
        }

        with (
            patch.object(diagnostic, "get_parallel", return_value=parallel),
            patch.object(diagnostic.logger, "info") as info,
        ):
            log_prefill_layer_hash_snapshot(
                boundary="layer_input",
                layer_id=51,
                rid="sensitive-request-id",
                seq_len=2,
                positions=positions,
                tensors=tensors,
            )

        payloads = [json.loads(call.args[2]) for call in info.call_args_list]
        self.assertEqual(len(payloads), 3)
        by_component = {payload["component"]: payload for payload in payloads}
        self.assertEqual(by_component["hidden_states"]["row_count"], 2)
        self.assertEqual(by_component["hidden_states"]["shape"], [2, 2])
        self.assertEqual(
            by_component["hidden_states"]["positions_sha256"],
            hash_int_sequence([0, 1]),
        )
        self.assertFalse(by_component["residual"]["present"])
        self.assertTrue(by_component["topk_indices"]["present"])
        self.assertNotIn("sensitive-request-id", json.dumps(payloads))

    def test_snapshot_logs_hashes_and_counts_without_tensor_contents(self):
        parallel = SimpleNamespace(
            world_rank=1,
            tp_rank=1,
            pp_rank=0,
            attn_tp_rank=1,
            attn_dp_rank=0,
            attn_dcp_size=2,
            attn_dcp_rank=1,
        )
        pool = SimpleNamespace(
            kv_buffer=[torch.arange(16, dtype=torch.int32).reshape(8, 1, 2)],
            index_k_buffer=[torch.arange(16, dtype=torch.int32).reshape(1, 8, 1, 2)],
            start_layer=4,
        )
        environment = {
            "SGLANG_DEBUG_DCP_CACHE_HASH": "1",
            "SGLANG_DEBUG_DCP_CACHE_HASH_SEQ_LEN": "4",
        }

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(diagnostic, "get_parallel", return_value=parallel),
            patch.object(diagnostic.logger, "info") as info,
        ):
            self.assertTrue(
                log_dsa_cache_hash_snapshot(
                    stage="decode_post_receive",
                    rid="sensitive-request-id",
                    bootstrap_room=123,
                    seq_len=4,
                    global_slots=torch.tensor([4, 5, 6, 7]),
                    pool=pool,
                )
            )

        payloads = [json.loads(call.args[2]) for call in info.call_args_list]
        self.assertEqual(len(payloads), 3)
        slot_map = next(
            payload for payload in payloads if payload["record_type"] == "slot_map"
        )
        layers = {
            payload["component"]: payload
            for payload in payloads
            if payload["record_type"] == "layer_hash"
        }
        self.assertEqual(slot_map["owned_position_count"], 2)
        self.assertEqual(layers["latent_kv"]["row_count"], 2)
        self.assertEqual(layers["dsa_index_k"]["row_count"], 4)
        self.assertEqual(layers["latent_kv"]["layer_id"], 4)
        self.assertNotIn("sensitive-request-id", json.dumps(payloads))

    def test_diagnostic_hooks_cover_the_three_transfer_boundaries_once(self):
        hooks = (
            ("disaggregation/prefill.py", "send_kv_chunk"),
            ("disaggregation/mooncake/conn.py", "send_kvcache_dcp"),
            ("disaggregation/decode.py", "pop_transferred"),
        )

        for relative_path, function_name in hooks:
            with self.subTest(path=relative_path, function=function_name):
                self.assertEqual(
                    len(_function_calls(relative_path, function_name)),
                    1,
                )

        prefill_call = _function_calls("disaggregation/prefill.py", "send_kv_chunk")[0]
        decode_call = _function_calls("disaggregation/decode.py", "pop_transferred")[0]
        prefill_keywords = {
            keyword.arg: ast.unparse(keyword.value) for keyword in prefill_call.keywords
        }
        decode_keywords = {
            keyword.arg: ast.unparse(keyword.value) for keyword in decode_call.keywords
        }

        self.assertEqual(
            prefill_keywords["pool"],
            "self.token_to_kv_pool_allocator.get_kvcache()",
        )
        self.assertEqual(
            decode_keywords["global_slots"],
            "self.scheduler.req_to_token_pool.req_to_token[decode_req.req."
            "req_pool_idx, :seq_len]",
        )
        self.assertEqual(
            decode_keywords["pool"],
            "self.scheduler.token_to_kv_pool_allocator.get_kvcache()",
        )

    def test_prefill_layer_hash_hooks_cover_stage_input_and_layer_boundaries(self):
        config_calls = _class_method_calls(
            "models/deepseek_v2.py",
            "DeepseekV2Model",
            "forward",
            "get_prefill_layer_hash_config",
        )
        snapshot_calls = _class_method_calls(
            "models/deepseek_v2.py",
            "DeepseekV2Model",
            "forward",
            "log_prefill_layer_hash_snapshot",
        )

        self.assertEqual(len(config_calls), 1)
        self.assertEqual(len(snapshot_calls), 3)
        boundaries = {
            ast.literal_eval(
                next(
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "boundary"
                )
            )
            for call in snapshot_calls
        }
        self.assertEqual(
            boundaries,
            {"pp_stage_input", "layer_input", "layer_output"},
        )

        decoder_snapshot_calls = _class_method_calls(
            "models/deepseek_v2.py",
            "DeepseekV2DecoderLayer",
            "forward",
            "log_prefill_layer_hash_snapshot",
        )
        self.assertEqual(len(decoder_snapshot_calls), 5)
        decoder_boundaries = {
            ast.literal_eval(
                next(
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "boundary"
                )
            )
            for call in decoder_snapshot_calls
        }
        self.assertEqual(
            decoder_boundaries,
            {
                "after_prepare_attn",
                "after_self_attn",
                "after_prepare_mlp",
                "after_mlp",
                "after_postprocess_layer",
            },
        )

        mlp_snapshot_calls = sum(
            (
                _class_method_attribute_calls(
                    "models/deepseek_v2.py",
                    "DeepseekV2MoE",
                    method_name,
                    "_log_prefill_mlp_hash_snapshot",
                )
                for method_name in (
                    "forward_normal",
                    "_forward_experts_with_prefill_mlp_hash",
                )
            ),
            [],
        )
        mlp_boundaries = {
            ast.literal_eval(
                next(
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "boundary"
                )
            )
            for call in mlp_snapshot_calls
        }
        self.assertEqual(
            mlp_boundaries,
            {
                "mlp_after_gate",
                "mlp_after_topk",
                "mlp_after_dispatch",
                "mlp_after_expert_core",
                "mlp_after_combine",
                "mlp_after_experts",
                "mlp_after_shared_experts",
                "mlp_after_shared_add",
                "mlp_after_optional_outer_all_reduce",
                "mlp_output",
            },
        )


if __name__ == "__main__":
    unittest.main()
