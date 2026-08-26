import unittest

from sglang.srt.layers.attention.dsa_backend import _validate_dsa_dcp_launch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _valid_config(**overrides):
    config = {
        "dcp_enabled": True,
        "dcp_size": 8,
        "device_capability": (9, 0),
        "dsa_prefill_impl": "flashmla_kv",
        "dsa_decode_impl": "flashmla_kv",
        "dsa_kv_cache_store_fp8": True,
        "page_size": 64,
        "enable_prefill_cp": False,
        "enable_hisparse": False,
        "enable_hierarchical_cache": False,
        "enable_symm_mem": False,
        "speculative_algorithm": None,
        "fused_topk_enabled": False,
        "decode_cuda_graph_disabled": True,
        "dcp_comm_backend": "ag_rs",
    }
    config.update(overrides)
    return config


class TestDSADCPLaunchContract(unittest.TestCase):
    def test_supported_h20_contract(self):
        _validate_dsa_dcp_launch(**_valid_config())

    def test_dcp_disabled_is_noop(self):
        _validate_dsa_dcp_launch(
            **_valid_config(
                dcp_enabled=False,
                dcp_size=1,
                device_capability=(8, 0),
                dsa_prefill_impl="trtllm",
                dsa_decode_impl="trtllm",
                dsa_kv_cache_store_fp8=False,
                page_size=1,
                enable_prefill_cp=True,
                enable_hisparse=True,
                enable_hierarchical_cache=True,
                enable_symm_mem=True,
                speculative_algorithm="EAGLE",
                fused_topk_enabled=True,
                decode_cuda_graph_disabled=False,
                dcp_comm_backend="a2a",
            )
        )

    def test_rejects_each_out_of_scope_combination(self):
        cases = {
            "dcp size": {"dcp_size": 3},
            "SM90": {"device_capability": (10, 0)},
            "flashmla_kv": {"dsa_prefill_impl": "trtllm"},
            "FP8 KV": {"dsa_kv_cache_store_fp8": False},
            "page size 64": {"page_size": 1},
            "prefill CP": {"enable_prefill_cp": True},
            "HiSparse": {"enable_hisparse": True},
            "HiCache": {"enable_hierarchical_cache": True},
            "symmetric memory": {"enable_symm_mem": True},
            "speculative decoding": {"speculative_algorithm": "EAGLE"},
            "fused DSA top-k": {"fused_topk_enabled": True},
            "decode CUDA Graph": {"decode_cuda_graph_disabled": False},
            "ag_rs": {"dcp_comm_backend": "a2a"},
        }
        for expected, overrides in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_dsa_dcp_launch(**_valid_config(**overrides))


if __name__ == "__main__":
    unittest.main()
